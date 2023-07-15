
from typing import Callable

import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ss_baselines.av_nav.ppo.policy import CriticHead

# General helpers
def compute_grad_norm(model):
    if isinstance(model, nn.Parameter):
        return model.grad.norm(2)

    elif isinstance(model, nn.Module):
        total_norm = 0.
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** (.5)
    
    else:
        raise NotImplementedError(f"Unsupported grad norm computation of {type(model)}")

###################################
# region: Vision modules          #

# From ss_baselines/av_nav/models/visual_cnn.py
from ss_baselines.common.utils import CategoricalNet, Flatten

def conv_output_dim(dimension, padding, dilation, kernel_size, stride
):
    r"""Calculates the output height and width based on the input
    height and width to the convolution layer.

    ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    """
    assert len(dimension) == 2
    out_dimension = []
    for i in range(len(dimension)):
        out_dimension.append(
            int(
                np.floor(
                    (
                            (
                                    dimension[i]
                                    + 2 * padding[i]
                                    - dilation[i] * (kernel_size[i] - 1)
                                    - 1
                            )
                            / stride[i]
                    )
                    + 1
                )
            )
        )
    return tuple(out_dimension)

def layer_init(cnn):
    for layer in cnn:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                layer.weight, nn.init.calculate_gain("relu")
            )
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)


class VisualCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, extra_rgb, obs_center=False):
        super().__init__()
        if "rgb" in observation_space.spaces and not extra_rgb:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        self.output_size = output_size
        self.obs_center = obs_center

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (2, 2)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                self.cnn_dims = cnn_dims = conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                #  nn.ReLU(True),
                Flatten(),
                nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.ReLU(True),
            )

        layer_init(self.cnn)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            if self.obs_center:
                rgb_observations -= 0.5
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            if self.obs_center:
                depth_observations -= 0.5
            cnn_input.append(depth_observations)

        cnn_input = th.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)

"""
    Mirrors upon the default Visual Encoder for SSL reconstruction task
"""
class ReshapeLayer(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape
    
    def forward(self, x):
        B = x.shape[0]
        return x.reshape(B, *self.output_shape)
    
class VisualCNNDecoder(nn.Module):
    def __init__(self, visual_encoder):
        super().__init__()
        # Reference visual encoder that will be mirrored
        # self.visual_encoder = visual_encoder
        output_size = visual_encoder.output_size
        cnn_dims = visual_encoder.cnn_dims
        kernel_sizes = visual_encoder._cnn_layers_kernel_size
        stride_sizes = visual_encoder._cnn_layers_stride

        if visual_encoder.is_blind:
            self.cnn = nn.Sequential()
        else:
            self.cnn = nn.Sequential(
                nn.Linear(output_size, 64 * cnn_dims[0] * cnn_dims[1]),
                nn.ReLU(True),
                ReshapeLayer([64, cnn_dims[0], cnn_dims[1]]),

                nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=kernel_sizes[-1],
                    stride=stride_sizes[-1],
                    output_padding=1
                ),
                nn.ReLU(True),

                nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=kernel_sizes[-2],
                    stride=stride_sizes[-2],
                    output_padding=1
                ),
                nn.ReLU(True),

                nn.ConvTranspose2d(
                    in_channels=32,
                    out_channels=3,
                    kernel_size=kernel_sizes[-3],
                    stride=stride_sizes[-3],
                    # padding=2
                )
            )
    
    def forward(self, x):
        return self.cnn(x)

# endregion: Vision modules       #
###################################

###################################
# region: Audio modules          #

# From ss_baselines/av_nav/models/audio_cnn.py
class AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN for processing audio spectrogram features

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, audiogoal_sensor):
        super(AudioCNN, self).__init__()
        self._n_input_audio = observation_space.spaces[audiogoal_sensor].shape[2]
        self._audiogoal_sensor = audiogoal_sensor

        cnn_dims = np.array(
            observation_space.spaces[audiogoal_sensor].shape[:2], dtype=np.float32
        )

        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )

        layer_init(self.cnn)

    def forward(self, observations):
        cnn_input = []

        audio_observations = observations[self._audiogoal_sensor]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        audio_observations = audio_observations.permute(0, 3, 1, 2)
        cnn_input.append(audio_observations)

        cnn_input = th.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)

# endregion: Audio modules           #
###################################


###################################
# region: Recurrent modules       #

# From ss_baselines/av_nav/models/rnn_state_encoder.py
class RNNStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        r"""An RNN for encoding the state in RL.

        Supports masking the hidden state during various timesteps in the forward lass

        Args:
            input_size: The input size of the RNN
            hidden_size: The hidden size
            num_layers: The number of recurrent layers
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """

        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (2 if "LSTM" in self._rnn_type else 1)

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = th.cat(
                [hidden_states[0], hidden_states[1]], dim=0
            )

        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )

        return hidden_states

    def _mask_hidden(self, hidden_states, masks):
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states

        return hidden_states

    def single_forward(self, x, hidden_states, masks):
        r"""Forward for a non-sequence input
          - x: [NUM_ENVS, ...] is expected, then unsqueezed to [1, NUM_ENVS, ...] to match expectation of GRU
          - hidden_states: [1, NUM_ENVS, HIDDEN_SIZE] is expected here, then no futher reshape ... a bit weird but ok
          - masks: [NUM_ENVS, 1] is expected, then unsqueezed to [1, NUM_ENVS, ...] to match expectation of GRU
        """
        hidden_states = self._unpack_hidden(hidden_states)
        x, hidden_states = self.rnn(
            x.unsqueeze(0),
            self._mask_hidden(hidden_states, masks.unsqueeze(0)),
        )
        x = x.squeeze(0)
        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def seq_forward(self, x, hidden_states, masks):
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        n = hidden_states.size(1)
        t = int(x.size(0) / n)

        # unflatten
        x = x.view(t, n, x.size(1))
        masks = masks.view(t, n)

        # steps in sequence which have zero for any agent. Assume t=0 has
        # a zero in it.
        has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]  # handle scalar
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [t]

        hidden_states = self._unpack_hidden(hidden_states)
        outputs = []
        for i in range(len(has_zeros) - 1):
            # process steps that don't have any zeros in masks together
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, hidden_states = self.rnn(
                x[start_idx:end_idx],
                self._mask_hidden(
                    hidden_states, masks[start_idx].view(1, -1, 1)
                ),
            )

            outputs.append(rnn_scores)

        # x is a (T, N, -1) tensor
        x = th.cat(outputs, dim=0)
        x = x.view(t * n, -1)  # flatten

        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def forward(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(1):
            return self.single_forward(x, hidden_states, masks)
        else:
            return self.seq_forward(x, hidden_states, masks)

# endregion: Recurrent modules    #
###################################

#########################################
# region: Actor Critic modules          #

# Based on ss_baselines/av_nav/ppo/policy.py's AudioNavBaselineNet module
# A simplified actor critic that assumes the few following points:
## - Not blind: there is always a visual observation, and it is RGB by default
## - Not deaf: there is always an acoustic observation, and it is Spectrogram by default
## - extra_rgb: make the VisualCNN ignore the rgb observations even if they are in the observations
GRU_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES = [
    *[f"visual_encoder.cnn.{i}" for i in range(8)], # Shared arch. for GRU and PGWT
    *[f"audio_encoder.cnn.{i}" for i in range(8)], # Shared arch. for GRU and PGWT
    "action_distribution.linear", # Shared arch. for any type of agent
    "critic.fc", # Shared arch. for any type of agent
    
    "state_encoder", # Either GRU or PGWT based. Shape different based on nature of cell though.
]
class ActorCritic(nn.Module):
    def __init__(self, 
                 observation_space,
                 action_space,
                 args,
                 extra_rgb=False,
                 analysis_layers=[]):
        super().__init__()

        self.args = args

        hidden_size = args.hidden_size
        self.prev_actions = prev_actions = args.prev_actions

        if prev_actions:
            self.prev_action_embedding = nn.Linear(action_space.n, hidden_size)

        # TODO: 
        # - support for blind agent
        # - support for RGB + Depth as 4 channel dim (default SS / SAVi behavior)
        # - support for RGB CNN and Depth CNN separated (geared toward PGWT modality)
        self.visual_encoder = VisualCNN(observation_space, hidden_size, 
            extra_rgb=extra_rgb, obs_center=args.obs_center)
        self.audio_encoder = AudioCNN(observation_space, hidden_size, "spectrogram")
        
        if self.visual_encoder.is_blind:
            input_dim = hidden_size
        else:
            input_dim = hidden_size * 2

        if self.prev_actions:
            input_dim += hidden_size

        self.state_encoder = RNNStateEncoder(input_dim, hidden_size)

        self.action_distribution = CategoricalNet(hidden_size, action_space.n) # Policy fn
        self.critic = CriticHead(hidden_size) # Value fn

        self.train()

        # Layers to record for neuroscience based analysis
        self.analysis_layers = analysis_layers

        # Hooks for intermediate features storage
        if len(analysis_layers):
            self._features = {layer: th.empty(0) for layer in self.analysis_layers}
            for layer_id in analysis_layers:
                layer = dict([*self.named_modules()])[layer_id]
                layer.register_forward_hook(self.save_outputs_hook(layer_id))
        
        # SSL tasks
        if args.ssl_tasks is not None:
            self.ssl_modules = nn.ModuleDict()
            for ssl_task in args.ssl_tasks:
                if ssl_task == "rec-rgb":
                    # TODO: check that the Vis. Encoder is actually using RGB, not RGBD
                    self.ssl_modules["rec-rgb"] = VisualCNNDecoder(self.visual_encoder)
                else:
                    raise NotImplementedError(f"Unsupported ssl-task: {ssl_task}")
                # elif ssl_task == "rec-spectr":
                #     pass
    
    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, observations, rnn_hidden_states, masks, prev_actions=None):
        x1 = []

        # Extracts audio featues
        # TODO: consider having waveform data too ?
        audio_features = self.audio_encoder(observations)
        x1.append(audio_features)

        # Extracts vision features
        ## TODO: consider having
        ## - rgb and depth simulatenous input as 4 channel dim input
        ## - deparate encoders for rgb and depth, give one more modality to PGWT
        if not self.visual_encoder.is_blind:
            video_features = self.visual_encoder(observations)
            x1.append(video_features)

        if self.prev_actions:
            prev_actions *= masks
            embed_prev_actions = self.prev_action_embedding(prev_actions)
            x1.append(embed_prev_actions)
        
        x1 = th.cat(x1, dim=1)
        # Current state, current rnn hidden states, respectively
        x2, rnn_hidden_states2 = self.state_encoder(x1, rnn_hidden_states, masks)

        if th.isnan(x2).any().item():
            for key in observations:
                print(key, th.isnan(observations[key]).any().item())
            print('rnn_old', th.isnan(rnn_hidden_states).any().item())
            print('rnn_new', th.isnan(rnn_hidden_states2).any().item())
            print('mask', th.isnan(masks).any().item())
            assert True
        
        return x2, rnn_hidden_states2
    
    def act(self, observations, rnn_hidden_states, masks, deterministic=False, actions=None, prev_actions=None,
                  value_feat_detach=False, actor_feat_detach=False, ssl_tasks=None):
        features, rnn_hidden_states = self(observations, rnn_hidden_states, masks, prev_actions=prev_actions)

        # Estimate the value function
        values = self.critic(features.detach() if value_feat_detach else features)

        # Estimate the policy as distribution to sample actions from
        distribution, action_logits = self.action_distribution(features.detach() if actor_feat_detach else features)

        if actions is None:
            if deterministic:
                actions = distribution.mode()
            else:
                actions = distribution.sample()
        # TODO: maybe some assert on the 
        action_log_probs = distribution.log_probs(actions)

        distribution_entropy = distribution.entropy()

        # Additonal SSL tasks outputs
        ssl_outputs = {}
        if ssl_tasks is not None:
            for ssl_task in ssl_tasks:
                if ssl_task == "rec-rgb":
                    ssl_outputs[ssl_task] = self.ssl_modules[ssl_task](features)
                else:
                    raise NotImplementedError(f"Unsupported SSL task: {ssl_task}")

        return actions, distribution.probs, action_log_probs, action_logits, \
               distribution_entropy, values, rnn_hidden_states, ssl_outputs
    
    def get_value(self, observations, rnn_hidden_states, masks, prev_actions=None):
        features, _ = self(observations, rnn_hidden_states, masks, prev_actions=prev_actions)

        return self.critic(features)

    def get_grad_norms(self):
        modules = ["visual_encoder", "audio_encoder", "state_encoder", "action_distribution", "critic"]
        return {mod_name: compute_grad_norm(self.__getattr__(mod_name)) for mod_name in modules}

# Actor critic variant that uses Perceiver as an RNN internally
# This variant uses the same visual and audio encoder as SS baseline
# TODO: clean up the followign variant, and make the PGWT GWWM one the main.
from perceiver_gwt import Perceiver_GWT
from perceiverio_gwt import PerceiverIO_GWT

class Perceiver_GWT_ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, config, extra_rgb=False, obs_center=False):
        super().__init__()
        self.config = config

        # TODO: later, we might want to augment the RGB info with the depth.
        self.visual_encoder = VisualCNN(observation_space, config.hidden_size, extra_rgb=extra_rgb, obs_center=False)
        self.audio_encoder = AudioCNN(observation_space, config.hidden_size, "spectrogram")

        self.state_encoder = Perceiver_GWT(
            depth = config.pgwt_depth, # Our default: 4; Perceiver default: 6
            input_channels = config.hidden_size * 2, # ???
            latent_type = config.pgwt_latent_type,
            latent_learned = config.pgwt_latent_learned,
            num_latents = config.pgwt_num_latents, # Our default: 32, Perceiver: 512
            latent_dim = config.pgwt_latent_dim, # Our default: 32, Perceiver default: 512
            cross_heads = config.pgwt_cross_heads, # Default: 1
            latent_heads = config.pgwt_latent_heads, # Default: 8
            cross_dim_head = config.pgwt_cross_dim_head, # Default: 64
            latent_dim_head = config.pgwt_latent_dim_head, # Default: 64
            attn_dropout = 0.,
            ff_dropout = 0.,
            self_per_cross_attn = 1,
            weight_tie_layers = config.pgwt_weight_tie_layers, # Default: False
            # Fourier Features related
            max_freq = config.pgwt_max_freq, # Default: 10.
            num_freq_bands = config.pgwt_num_freq_bands, # Default: 6
            fourier_encode_data = config.pgwt_ff,
            input_axis = 1,
            # Modality embedding related
            mod_embed = config.pgwt_mod_embed, # Using / dimension of mdality embeddings
            hidden_size = config.hidden_size,
            use_sa = config.pgwt_use_sa
        )

        # input_dim for policy / value is num_latents * latent_dim of the Perceiver
        self.action_distribution = CategoricalNet(config.pgwt_num_latents * config.pgwt_latent_dim,
            action_space.n) # Policy fn
        self.critic = CriticHead(config.pgwt_num_latents * config.pgwt_latent_dim) # Value fn

        self.train()

    def forward(self, observations, prev_latents, masks):
        video_features = self.visual_encoder(observations)
        audio_features = self.audio_encoder(observations)

        obs_feat = th.cat([audio_features, video_features], dim=1)
        state_feat, latents = self.state_encoder(obs_feat, prev_latents, masks) # [B, num_latents, latent_dim]

        return state_feat, latents

    def act(self, observations, prev_latents, masks, deterministic=False, actions=None):
        features, latents = self(observations, prev_latents, masks)

        # Estimate the value function
        values = self.critic(features)

        # Estimate the policy as distribution to sample actions from
        distribution, action_logits = self.action_distribution(features)

        if actions is None:
            if deterministic:
                actions = distribution.mode()
            else:
                actions = distribution.sample()
        # TODO: maybe some assert on the 
        action_log_probs = distribution.log_probs(actions)

        distribution_entropy = distribution.entropy()

        return actions, distribution.probs, action_log_probs, action_logits, distribution_entropy, values, latents
    
    def get_value(self, observations, prev_latents, masks):
        features, _ = self(observations, prev_latents, masks)

        return self.critic(features)

    def get_grad_norms(self):
        modules = ["visual_encoder", "audio_encoder", "state_encoder", "action_distribution", "critic"]
        return {mod_name: compute_grad_norm(self.__getattr__(mod_name)) for mod_name in modules}

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn
    
# TODO: clean up and make this variant the main one.
from perceiver_gwt_gwwm import Perceiver_GWT_GWWM, Perceiver_GWT_AttGRU
# Layer names for recording intermediate features, for a PGWT GWWM network with a single CA, no SA, no modality embeddings
PGWT_GWWM_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES = [
    *[f"visual_encoder.cnn.{i}" for i in range(8)], # Shared arch. for GRU and PGWT
    *[f"audio_encoder.cnn.{i}" for i in range(8)], # Shared arch. for GRU and PGWT
    "action_distribution.linear", # Shared arch. for any type of agent
    "critic.fc", # Shared arch. for any type of agent

    "state_encoder", # Either GRU or PGWT based. Shape different based on nature of cell though.
    ## PGWT GWWM specific
    "state_encoder.ca", # This will have the output of the residual conn. self.ff_self(attention_value) + attention_value
    "state_encoder.ca.mha", # Note: output here is tuple (attention_value, attention_weight)
    "state_encoder.ca.ln_q",
    "state_encoder.ca.ln_kv",
    "state_encoder.ca.ff_self", # No residual connection output
    *[f"state_encoder.ca.ff_self.{i}" for i in range(4)], # [LayerNorm, Linear, GELU, Linear]
    ## TODO: add support for the SA layers too
]

class Perceiver_GWT_GWWM_ActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 config,
                 extra_rgb=False,
                 analysis_layers=[]):
        
        super().__init__()
        self.config = config

        # TODO: 
        # - support for blind agent
        # - support for RGB + Depth as 4 channel dim (default SS / SAVi behavior)
        # - support for RGB CNN and Depth CNN separated (geared toward PGWT modality)
        self.visual_encoder = VisualCNN(observation_space, config.hidden_size,
            extra_rgb=extra_rgb, obs_center=config.obs_center)
        self.audio_encoder = AudioCNN(observation_space, config.hidden_size, "spectrogram")
        
        # Override the state encoder with a custom PerceiverIO
        self.state_encoder = Perceiver_GWT_GWWM(
            input_dim = config.hidden_size,
            latent_type = config.pgwt_latent_type,
            latent_learned = config.pgwt_latent_learned,
            num_latents = config.pgwt_num_latents, # Our default: 32, Perceiver: 512
            latent_dim = config.pgwt_latent_dim, # Our default: 32, Perceiver default: 512
            cross_heads = config.pgwt_cross_heads, # Default: 1
            latent_heads = config.pgwt_latent_heads, # Default: 8
            # cross_dim_head = config.pgwt_cross_dim_head, # Default: 64
            # latent_dim_head = config.pgwt_latent_dim_head, # Default: 64
            # self_per_cross_attn = 1,
            # Modality embedding related
            mod_embed = config.pgwt_mod_embed, # Using / dimension of mdality embeddings
            hidden_size = config.hidden_size,
            use_sa = config.pgwt_use_sa,
            ca_prev_latents = config.pgwt_ca_prev_latents
        )

        self.action_distribution = CategoricalNet(config.hidden_size, action_space.n) # Policy fn
        self.critic = CriticHead(config.hidden_size) # Value fn

        self.train()
        
        # Layers to record for neuroscience based analysis
        self.analysis_layers = analysis_layers

        # Hooks for intermediate features storage
        if len(analysis_layers):
            self._features = {layer: th.empty(0) for layer in self.analysis_layers}
            for layer_id in analysis_layers:
                layer = dict([*self.named_modules()])[layer_id]
                layer.register_forward_hook(self.save_outputs_hook(layer_id))

        # SSL tasks
        if config.ssl_tasks is not None:
            self.ssl_modules = nn.ModuleDict()
            for ssl_task in config.ssl_tasks:
                if ssl_task == "rec-rgb":
                    # TODO: check that the Vis. Encoder is actually using RGB, not RGBD
                    self.ssl_modules["rec-rgb"] = VisualCNNDecoder(self.visual_encoder)
                else:
                    raise NotImplementedError(f"Unsupported ssl-task: {ssl_task}")
                # elif ssl_task == "rec-spectr":
                #     pass
        
    def forward(self, observations, prev_latents, masks):
        x1 = []
        
        # Extracts audio featues
        # TODO: consider having waveform data too ?
        audio_features = self.audio_encoder(observations)
        x1.append(audio_features[:, None, :]) # [B, H] -> [B, 1, H]

        # Extracts vision features
        ## TODO: consider having
        ## - rgb and depth simulatenous input as 4 channel dim input
        ## - deparate encoders for rgb and depth, give one more modality to PGWT
        if not self.visual_encoder.is_blind:
            video_features = self.visual_encoder(observations)
            x1.append(video_features[:, None, :]) # [B, H] -> [B, 1, H]
        
        # For the transformer, prepaer the input data in the [B, N_MODALITY, H] shape
        obs_feat = th.cat(x1, dim=1)

        state_feat, latents = self.state_encoder(obs_feat, prev_latents, masks) # [B, num_latents, latent_dim]

        return state_feat, latents
    
    def act(self, observations, rnn_hidden_states, masks, deterministic=False, actions=None,
                  value_feat_detach=False, actor_feat_detach=False, ssl_tasks=None):
        features, rnn_hidden_states = self(observations, rnn_hidden_states, masks)

        # Estimate the value function
        values = self.critic(features.detach() if value_feat_detach else features)

        # Estimate the policy as distribution to sample actions from
        distribution, action_logits = self.action_distribution(features.detach() if actor_feat_detach else features)

        if actions is None:
            if deterministic:
                actions = distribution.mode()
            else:
                actions = distribution.sample()
        # TODO: maybe some assert on the 
        action_log_probs = distribution.log_probs(actions)

        distribution_entropy = distribution.entropy()

        # Additonal SSL tasks outputs
        ssl_outputs = {}
        if ssl_tasks is not None:
            for ssl_task in ssl_tasks:
                if ssl_task == "rec-rgb":
                    ssl_outputs[ssl_task] = self.ssl_modules[ssl_task](features)
                else:
                    raise NotImplementedError(f"Unsupported SSL task: {ssl_task}")

        return actions, distribution.probs, action_log_probs, action_logits, \
               distribution_entropy, values, rnn_hidden_states, ssl_outputs
    
    def get_value(self, observations, rnn_hidden_states, masks):
        features, _ = self(observations, rnn_hidden_states, masks)

        return self.critic(features)

    def get_grad_norms(self):
        modules = ["visual_encoder", "audio_encoder", "state_encoder", "action_distribution", "critic"]
        grad_norm_dict = {mod_name: compute_grad_norm(self.__getattr__(mod_name)) for mod_name in modules}
        # More specific grad norms for the Perceiver GWT GWWM
        if self.state_encoder.mod_embed:
            grad_norm_dict["mod_embed"] = compute_grad_norm(self.state_encoder.modality_embeddings)
        if self.state_encoder.latent_learned:
            grad_norm_dict["init_rnn_states"] = compute_grad_norm(self.state_encoder.latents)
        
        return grad_norm_dict
    
    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

class Perceiver_GWT_AttGRU_ActorCritic(Perceiver_GWT_ActorCritic):
    def __init__(self, observation_space, action_space, config, extra_rgb=False):
        super().__init__(observation_space, action_space, config, extra_rgb)
        self.config = config

        # Override the state encoder with a custom PerceiverIO
        self.state_encoder = Perceiver_GWT_AttGRU(
            input_dim = config.hidden_size,
            latent_type = config.pgwt_latent_type,
            latent_learned = config.pgwt_latent_learned,
            num_latents = config.pgwt_num_latents, # Our default: 32, Perceiver: 512
            latent_dim = config.pgwt_latent_dim, # Our default: 32, Perceiver default: 512
            cross_heads = config.pgwt_cross_heads, # Default: 1
            latent_heads = config.pgwt_latent_heads, # Default: 8
            # cross_dim_head = config.pgwt_cross_dim_head, # Default: 64
            # latent_dim_head = config.pgwt_latent_dim_head, # Default: 64
            # self_per_cross_attn = 1,
            # Modality embedding related
            mod_embed = config.pgwt_mod_embed, # Using / dimension of mdality embeddings
            hidden_size = config.hidden_size,
            use_sa = config.pgwt_use_sa
        )

    def get_grad_norms(self):
        modules = ["visual_encoder", "audio_encoder", "state_encoder", "action_distribution", "critic"]
        grad_norm_dict = {mod_name: compute_grad_norm(self.__getattr__(mod_name)) for mod_name in modules}
        # More specific grad norms for the Perceiver GWT GWWM
        if self.state_encoder.mod_embed:
            grad_norm_dict["mod_embed"] = compute_grad_norm(self.state_encoder.modality_embeddings)
        if self.state_encoder.latent_learned:
            grad_norm_dict["init_rnn_states"] = compute_grad_norm(self.state_encoder.latents)
        
        return grad_norm_dict

## ActorCritic based on the Deep Ethorlogy Virtual Rodent paper
class ActorCritic_DeepEthologyVirtualRodent(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # TODO: later, we might want to augment the RGB info with the depth.
        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb=False)
        self.audio_encoder = AudioCNN(observation_space, hidden_size, "spectrogram")
        
        self.core_state_encoder = RNNStateEncoder(hidden_size * 2, hidden_size)
        self.policy_state_encoder = RNNStateEncoder(hidden_size * 3, hidden_size)

        self.action_distribution = CategoricalNet(hidden_size, action_space.n) # Policy fn
        self.critic = CriticHead(hidden_size) # Value fn

        self.train()
    
    def forward(self, observations, rnn_hidden_states, masks):
        video_features = self.visual_encoder(observations)
        audio_features = self.audio_encoder(observations)

        x1 = th.cat([audio_features, video_features], dim=1)
        # Current state, current rnn hidden states, respectively
        # Core module RNN step
        core_rnn_hidden_states = rnn_hidden_states[:, :, :self.hidden_size]
        core_x2, core_rnn_hidden_states2 = self.core_state_encoder(x1, core_rnn_hidden_states, masks)
        # Policy module RNN step
        policy_rnn_hidden_states = rnn_hidden_states[:, :, self.hidden_size:]
        policy_x1 = th.cat([x1, core_rnn_hidden_states2[0].detach()], dim=1) # They mention this in Fig 2.3 in the paper
        policy_x2, policy_rnn_hidden_states2 = self.policy_state_encoder(policy_x1, policy_rnn_hidden_states, masks)
        
        return core_x2, policy_x2, core_rnn_hidden_states2, policy_rnn_hidden_states2
    
    def act(self, observations, rnn_hidden_states, masks, deterministic=False, actions=None):
        core_features, policy_features, core_rnn_hidden_states, policy_rnn_hidden_states = \
            self(observations, rnn_hidden_states, masks)

        # Estimate the value function
        values = self.critic(core_features)

        # Estimate the policy as distribution to sample actions from
        distribution, action_logits = self.action_distribution(policy_features)

        if actions is None:
            if deterministic:
                actions = distribution.mode()
            else:
                actions = distribution.sample()
        # TODO: maybe some assert on the 
        action_log_probs = distribution.log_probs(actions)

        distribution_entropy = distribution.entropy().mean()

        return actions, action_log_probs, action_logits, distribution_entropy, values, \
            th.cat([core_rnn_hidden_states, policy_rnn_hidden_states], dim=2) # Very workaround-y method
    
    def get_value(self, observations, rnn_hidden_states, masks):
        core_features, _, _, _ = \
            self(observations, rnn_hidden_states, masks)

        return self.critic(core_features)

# endregion: Actor Critic modules       #
#########################################
