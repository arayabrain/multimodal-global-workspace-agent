from importlib_metadata import distribution
import numpy as np
from ss_baselines.av_nav.ppo.policy import CriticHead

import torch as th
import torch.nn as nn


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

    def __init__(self, observation_space, output_size, extra_rgb):
        super().__init__()
        if "rgb" in observation_space.spaces and not extra_rgb:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

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
                cnn_dims = conv_output_dim(
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
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = th.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)

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
        return self._num_recurrent_layers * (
            2 if "LSTM" in self._rnn_type else 1
        )

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
class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super().__init__()

        # TODO: later, we might want to augment the RGB info with the depth.
        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb=False)
        self.audio_encoder = AudioCNN(observation_space, hidden_size, "spectrogram")
        
        self.state_encoder = RNNStateEncoder(hidden_size * 2, hidden_size)

        self.action_distribution = CategoricalNet(hidden_size, action_space.n) # Policy fn
        self.critic = CriticHead(hidden_size) # Value fn

        self.train()
    
    def forward(self, observations, rnn_hidden_states, masks):
        video_features = self.visual_encoder(observations)
        audio_features = self.audio_encoder(observations)

        x1 = th.cat([audio_features, video_features], dim=1)
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
    
    def act(self, observations, rnn_hidden_states, masks, deterministic=False, actions=None):
        features, rnn_hidden_states = self(observations, rnn_hidden_states, masks)

        # Estimate the value function
        values = self.critic(features)

        # Estimate the policy as distribution to sample actions from
        distribution = self.action_distribution(features)

        if actions is None:
            if deterministic:
                actions = distribution.mode()
            else:
                actions = distribution.sample()
        # TODO: maybe some assert on the 
        action_log_probs = distribution.log_probs(actions)

        distribution_entropy = distribution.entropy().mean()

        return actions, action_log_probs, distribution_entropy, values, rnn_hidden_states
    
    def get_value(self, observations, rnn_hidden_states, masks):
        features, _ = self(observations, rnn_hidden_states, masks)

        return self.critic(features)

# endregion: Actor Critic modules       #
#########################################
