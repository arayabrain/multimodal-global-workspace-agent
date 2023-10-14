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
from ss_baselines.common.utils import CategoricalNet, CategoricalNet2, Flatten

def conv_output_dim(dimension, padding, dilation, kernel_size, stride):
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

class RecurrentVisualEncoder(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, extra_rgb, use_gw=False, obs_center=False):
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
        self.use_gw = use_gw

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
                nn.ReLU(True),
                Flatten(),
            )

            rnn_input_dim = 2304 # TODO: dynamic compute of flattened output
            if self.use_gw:
                rnn_input_dim += 512 # TODO: make this dynamic / based on config

            self.rnn = nn.GRUCell(
                input_size=rnn_input_dim,
                hidden_size=output_size,
            )

        layer_init(self.cnn)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations, prev_states, masks, prev_gw=None):
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

        rnn_input = self.cnn(cnn_input)

        if self.use_gw:
            assert prev_gw is not None, f"RecurVisEnc requires 'gw' tensor when in GW usage mode"
            rnn_input = th.cat([rnn_input, prev_gw], dim=1)

        # TODO check prev_states and masks dimensions
        return self.rnn(rnn_input, prev_states * masks)

# endregion: Vision modules       #
###################################

###################################
# region: Audio modules           #

# From ss_baselines/av_nav/models/audio_cnn.py
class RecurrentAudioEncoder(nn.Module):
    r"""A Simple 3-Conv CNN for processing audio spectrogram features

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, audiogoal_sensor, use_gw=False):
        super().__init__()
        self._n_input_audio = observation_space.spaces[audiogoal_sensor].shape[2]
        self._audiogoal_sensor = audiogoal_sensor
        self.use_gw = use_gw

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
            self._cnn_layers_kernel_size, self._cnn_layers_stride):
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
            nn.ReLU(True),
            Flatten(),
        )

        rnn_input_dim = 2496 # TODO: dynamic compute of flattened output
        if self.use_gw:
            rnn_input_dim += 512 # TODO: make this dynamic / based on config
        
        self.rnn = nn.GRUCell(
            input_size=rnn_input_dim,
            hidden_size=512, # TODO: base this on 
        )

        layer_init(self.cnn)

    def forward(self, observations, prev_states, masks, prev_gw=None):
        cnn_input = []

        audio_observations = observations[self._audiogoal_sensor]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        audio_observations = audio_observations.permute(0, 3, 1, 2)
        cnn_input.append(audio_observations)

        cnn_input = th.cat(cnn_input, dim=1)

        rnn_input = self.cnn(cnn_input)

        if self.use_gw:
            assert prev_gw is not None, f"RecurAudEnc requires 'gw' tensor when in GW usage mode"
            rnn_input = th.cat([rnn_input, prev_gw], dim=1)

        return self.rnn(rnn_input, prev_states * masks)

# endregion: Audio modules        #
###################################

######################################
# region: GWT v3 Custom Attention    #

class CrossAttention(nn.Module):
    def __init__(self, q_size, kv_size, n_heads=1, use_null=True):
        super().__init__()
        self.h_size = q_size
        self.kv_size = kv_size
        self.n_heads = n_heads
        self.use_null = use_null

        self.ln_q = nn.LayerNorm([q_size])
        self.ln_k = nn.LayerNorm([kv_size])
        self.ln_v = nn.LayerNorm([kv_size])

        self.mha = nn.MultiheadAttention(
            q_size,
            n_heads,
            dropout=0.0,
            add_zero_attn=False,
            batch_first=True,
            kdim=kv_size,
            vdim=kv_size,
        )

    def forward(self, modality_features, prev_gw):
        """
            Modality features: dict of:
                - "visual": [B, 1, H]
                - "audio": [B, 1, H]
        """
        B = modality_features["audio"].shape[0]
        H = modality_features["audio"].shape[-1] # TODO: recover from the cfg instead ?

        q = th.cat([
                modality_features["audio"], # [B, 1, H]
                modality_features["visual"], # [B, 1, H]
            ], dim=1)
        kv = th.cat([
            modality_features["audio"], # [B, 1, H]
            modality_features["visual"], # [B, 1, H]
            prev_gw[:, None, :] # [B, 1, H]
            ], dim=1)
        if self.use_null:
            kv = th.cat([
                kv, # [B, 3, H] by this point
                kv.new_zeros([B, 1, H]) # Nulls: [B, 1, H]
            ], dim=1)
        
        q = self.ln_q(q) # [B, 2, H]
        k = self.ln_k(kv) # [B, X, H], X = 3 or 4
        v = self.ln_v(kv) # [B, X, H], X = 3 or 4
        attn_values, attn_weights = self.mha(q, k, v)

        return attn_values, attn_weights # [B, 2, H], [B, 2, X], X = 3 or 4

class GWTv3StateEncoder(nn.Module):
    def __init__(self, 
                 input_size,
                 hidden_size,
                 ca_q_size,
                 ca_kv_size,
                 ca_n_heads=1,
                 ca_use_null=True):
        super().__init__()

        # CrossAttention module
        self.ca = CrossAttention(
            q_size=ca_q_size,
            kv_size=ca_kv_size,
            n_heads=ca_n_heads,
            use_null=ca_use_null
        )

        # RNN module
        self.rnn = nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )

        # Custom RNN params. init
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        
    def forward(self, modality_features, prev_gw, masks):
        """
        - modality_features: dict of:
            - "visual": [B, 1, H]
            - "audio": [B, 1, H]
        - hidden_states: [B, H] or [1, B, H] for compat. with SS1.0
            This is the previous global workspace
        - masks: [B, 1], marks episode end / start
            Used to init prev_gw when a new episode starts
        """
        B = modality_features["audio"].shape[0]

        attn_values, attn_weights = self.ca(
            modality_features, # {"visual": Tnsr, "audio": Tnsr}
            prev_gw=prev_gw * masks # # [B, H]
        )

        rnn_output = self.rnn(
            attn_values.reshape(B, -1), # [B, 2 * H]
            prev_gw * masks, # [B, H]
        )

        return rnn_output
    

# endregion: GWT v3 Custom Attention #
######################################

###################################
# region: GWT v3 Agent            #

class GWTv3ActorCritic(nn.Module):
    def __init__(self,
                    observation_space,
                    action_space,
                    config,
                    extra_rgb=False,
                    analysis_layers=[]):
        super().__init__()
        self.config = config

        self.visual_encoder = RecurrentVisualEncoder(
            observation_space, 
            config.hidden_size,
            extra_rgb=extra_rgb,
            use_gw=config.gwtv3_use_gw
        )
        self.audio_encoder = RecurrentAudioEncoder(
            observation_space,
            config.hidden_size,
            "spectrogram",
            use_gw=config.gwtv3_use_gw
        )
        
        self.state_encoder = GWTv3StateEncoder(
            input_size=config.hidden_size * 2,
            hidden_size=config.hidden_size,
            ca_q_size=config.hidden_size,
            ca_kv_size=config.hidden_size,
            ca_n_heads=config.gwtv3_cross_heads,
            ca_use_null=config.gwtv3_use_null
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

    def single_forward(self, observations, prev_gw, masks, prev_modality_features):
        pass

    def seq_forward(self, observations, prev_gw, masks, prev_modality_features):
        pass

    def forward(self, observations, prev_gw, masks, prev_modality_features):
        # TODO: impelemtn sequential mode
        modality_features = {}

        # Extracts audio featues
        # TODO: consider having waveform data too ?
        audio_features = self.audio_encoder(observations,
                                prev_modality_features["audio"],
                                masks,
                                prev_gw=prev_gw)
        modality_features["audio"] = audio_features

        # Extracts vision features
        ## TODO: consider having
        ## - rgb and depth simulatenous input as 4 channel dim input
        ## - deparate encoders for rgb and depth, give one more modality to PGWT
        if not self.visual_encoder.is_blind:
            visual_features = self.visual_encoder(observations, 
                                    prev_modality_features["visual"], 
                                    masks,
                                    prev_gw=prev_gw)
            modality_features["visual"] = visual_features

        gw = self.state_encoder(
            modality_features = {
                k: v[:, None, :] for k, v in modality_features.items() # Reshape each mod feat to [B, 1, H]
            },
            prev_gw=prev_gw,
            masks=masks
        ) # [B, num_latents, latent_dim]

        # NOTE: modality_features is re-used in the next step
        # for recurrence at the encoder level
        return gw, modality_features # [B, H], {k: [B, H]} for k in "visual", "audio"

    def act(self, observations, rnn_hidden_states, masks, modality_features, deterministic=False, actions=None,
                  value_feat_detach=False, actor_feat_detach=False, ssl_tasks=None):
        gw, modality_features = self(observations, rnn_hidden_states, masks, modality_features)

        features = gw # alias for compat with previous version

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
                if ssl_task in ["rec-rgb-ae", "rec-rgb-ae-2", "rec-rgb-ae-3"]:
                    ssl_outputs[ssl_task] = self.ssl_modules[ssl_task](features)
                elif ssl_task in ["rec-rgb-vis-ae", "rec-rgb-vis-ae-3", "rec-rgb-vis-ae-mse"]:
                    ssl_outputs[ssl_task] = self.ssl_modules[ssl_task](modality_features["vision"])
                else:
                    raise NotImplementedError(f"Unsupported SSL task: {ssl_task}")

        return actions, distribution.probs, action_log_probs, action_logits, \
               distribution_entropy, values, rnn_hidden_states, modality_features, ssl_outputs
    
    def get_value(self, observations, rnn_hidden_states, masks, modality_features):
        features, _ = self(observations, rnn_hidden_states, masks, modality_features)

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

# endregion: GWT v3 Agent         #
###################################