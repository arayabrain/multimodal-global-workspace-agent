from typing import Callable

import numpy as np

import torch as th
import torch.nn as nn

# From ss_baselines/av_nav/models/visual_cnn.py
from ss_baselines.common.utils import CategoricalNet, Flatten

GWTAGENT_DEFAULT_ANALYSIS_LAYER_NAMES = [
  "visual_encoder",
  "audio_encoder",
  "state_encoder"
]

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

# region: Custom recurrent Layer
class GRUCell(nn.Module):

  def __init__(self, inp_size,
               size, norm=False, act=th.tanh, update_bias=-1):
    super(GRUCell, self).__init__()
    self._inp_size = inp_size
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = nn.Linear(inp_size+size, 3*size,
                            bias=norm is not None)
    if norm:
      self._norm = nn.LayerNorm(3*size)

  @property
  def state_size(self):
    return self._size

  def forward(self, inputs, state):
    # NOTE: Removing the line below, we get closer structure to PyTorch instead.
    # state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(th.cat([inputs, state], -1))
    if self._norm:
      parts = self._norm(parts)
    reset, cand, update = th.split(parts, [self._size]*3, -1)
    reset = th.sigmoid(reset)
    cand = self._act(reset * cand)
    update = th.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output
# endregion: Custom recurrent Layer

###################################
# region: Vision modules          #

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

    def __init__(self, observation_space, gw_size, output_size, extra_rgb, use_gw=False, gw_detach=False, gru_type="default", obs_center=False):
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
        self.gw_detach = gw_detach

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
                rnn_input_dim += gw_size

            if gru_type == "default":
                self.rnn = nn.GRUCell(
                    input_size=rnn_input_dim,
                    hidden_size=output_size,
                )
            elif gru_type == "layernorm":
                self.rnn = GRUCell(
                    rnn_input_dim,
                    output_size,
                    norm=True
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
            assert prev_gw is not None, "RecurVisEnc requires 'gw' tensor when in GW usage mode"
            rnn_input = th.cat([
                rnn_input, 
                (prev_gw.detach() if self.gw_detach else prev_gw) * masks
            ], dim=1)

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

    def __init__(self, observation_space, gw_size, output_size, audiogoal_sensor, use_gw=False, gw_detach=False, gru_type="default",):
        super().__init__()
        self._n_input_audio = observation_space.spaces[audiogoal_sensor].shape[2]
        self._audiogoal_sensor = audiogoal_sensor
        self.use_gw = use_gw
        self.gw_detach = gw_detach

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
            rnn_input_dim += gw_size
        
        if gru_type == "default":
            self.rnn = nn.GRUCell(
                input_size=rnn_input_dim,
                hidden_size=output_size,
            )
        elif gru_type == "layernorm":
            self.rnn = GRUCell(
                rnn_input_dim,
                output_size,
                norm=True
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
            assert prev_gw is not None, "RecurAudEnc requires 'gw' tensor when in GW usage mode"
            rnn_input = th.cat([
                rnn_input, 
                (prev_gw.detach() if self.gw_detach else prev_gw) * masks
            ], dim=1)

        return self.rnn(rnn_input, prev_states * masks)

# endregion: Audio modules        #
###################################

######################################
# region: GWT GW modules          #

class CrossAttention(nn.Module):
    def __init__(self, gw_size, feat_size, n_heads=1, use_null=True):
        super().__init__()
        self.n_heads = n_heads
        self.use_null = use_null
        self.gw_size = gw_size
        self.feat_size = feat_size

        # Linear projection to downscale input modalities
        self.proj_vis = nn.Linear(feat_size, gw_size, bias=False)
        self.proj_aud = nn.Linear(feat_size, gw_size, bias=False)

        self.ln_q = nn.LayerNorm([gw_size])
        self.ln_k = nn.LayerNorm([gw_size])
        self.ln_v = nn.LayerNorm([gw_size])

        self.mha = nn.MultiheadAttention(
            gw_size,
            n_heads,
            dropout=0.0,
            add_zero_attn=False,
            batch_first=True,
            kdim=gw_size,
            vdim=gw_size,
        )

    def forward(self, modality_features, prev_gw):
        """
            Modality features: dict of:
                - "visual": [B, 1, H]
                - "audio": [B, 1, H]
        """
        B = modality_features["audio"].shape[0]

        aud_feats = self.proj_vis(modality_features["audio"]) # From H -> GW_H
        vis_feats = self.proj_vis(modality_features["visual"])

        q = th.cat([
                aud_feats, # [B, 1, GW_H]
                vis_feats, # [B, 1, GW_H]
                prev_gw[:, None, :] # [B, 1, GW_H]
            ], dim=1) # [B, 3, GW_H]
        kv = th.cat([
            aud_feats, # [B, 1, GW_H]
            vis_feats, # [B, 1, GW_H]
            prev_gw[:, None, :] # [B, 1, GW_H]
            ], dim=1) # [B, 3, GW_H] so far
        if self.use_null:
            kv = th.cat([
                kv, # [B, 3, GW_H] by this point
                kv.new_zeros([B, 1, self.gw_size]) # Nulls: [B, 1, GW_H]
            ], dim=1) # [B, 4, GW_H]
        
        q = self.ln_q(q) # [B, 3, H]
        k = self.ln_k(kv) # [B, X, H], X = 3 or 4
        v = self.ln_v(kv) # [B, X, H], X = 3 or 4

        attn_values, attn_weights = self.mha(q, k, v)

        return attn_values, attn_weights # [B, 3, H], [B, 3, X], X = 3 or 4

class GWStateEncoder(nn.Module):
    def __init__(self,
                 gw_size,
                 feat_size,
                 ca_n_heads=1,
                 ca_use_null=True,
                 gru_type="layernorm"):
        super().__init__()

        # CrossAttention module
        self.ca = CrossAttention(
            gw_size=gw_size,
            feat_size=feat_size,
            n_heads=ca_n_heads,
            use_null=ca_use_null
        )

        # RNN module
        if gru_type == "default":
            self.rnn = nn.GRUCell(
                input_size=gw_size * 2,
                hidden_size=gw_size,
            )

            # Custom RNN params. init
            for name, param in self.rnn.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

        elif gru_type == "layernorm":
            self.rnn = GRUCell(
                gw_size * 2,
                gw_size,
                norm=True
            )
    
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

        aud_vis_modulated = attn_values[:, :2, :].reshape(B, -1) # [B, 2 * H]
        prev_gw_modulated = attn_values[:, 2, :].reshape(B, -1) # [B, H]

        rnn_output = self.rnn(
            aud_vis_modulated,
            prev_gw_modulated
        )

        return rnn_output

class GRUStateEncoder(nn.Module):
    def __init__(self, 
                gw_size,
                feat_size,
                gru_type="default"):
        super().__init__()

        # RNN module
        if gru_type == "default":
            self.rnn = nn.GRUCell(
                input_size=gw_size * 2,
                hidden_size=gw_size,
            )

            # Custom RNN params. init
            for name, param in self.rnn.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

        elif gru_type == "layernorm":
            self.rnn = GRUCell(
                gw_size * 2,
                gw_size,
                norm=True
            )

        # Linear projection to downscale input modalities
        self.proj_vis = nn.Linear(feat_size, gw_size, bias=False)
        self.proj_aud = nn.Linear(feat_size, gw_size, bias=False)

    def forward(self, modality_features, prev_gw, masks):
        """
        - modality_features: dict of:
            - "visual": [B, 1, H]
            - "audio": [B, 1, H]
        - prev_gw: [B, H] or [1, B, H] for compat. with SS1.0
            This is the previous global workspace
        - masks: [B, 1], marks episode end / start
            Used to init prev_gw when a new episode starts
        """

        aud_feats = self.proj_vis(modality_features["audio"]) # From H -> GW_H
        vis_feats = self.proj_vis(modality_features["visual"])
        
        x = th.cat([
            aud_feats[:, 0, :],    # [B, GW_H]
            vis_feats[:, 0, :],   # [B, GW_H]
            ], dim=1)
        rnn_output = self.rnn(
            x,
            prev_gw * masks, # [B, H]
        )

        return rnn_output

# endregion: GWT GW modules       #
######################################

###################################
# region: GWT Agent            #

class GW_Actor(nn.Module):
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
            config.gw_size,
            config.hidden_size,
            extra_rgb=extra_rgb,
            use_gw=config.recenc_use_gw,
            gw_detach=config.recenc_gw_detach,
            gru_type=config.gru_type,
        )
        self.audio_encoder = RecurrentAudioEncoder(
            observation_space,
            config.gw_size,
            config.hidden_size,
            "spectrogram",
            use_gw=config.recenc_use_gw,
            gw_detach=config.recenc_gw_detach,
            gru_type=config.gru_type,
        )
        
        self.state_encoder = GWStateEncoder(
            gw_size=config.gw_size,
            feat_size=config.hidden_size,
            ca_n_heads=config.gw_cross_heads,
            ca_use_null=config.gw_use_null,
            gru_type=config.gru_type,
        )

        self.action_distribution = CategoricalNet(config.gw_size, action_space.n) # Policy fn

        self.train()
        
        # Layers to record for neuroscience based analysis
        self.analysis_layers = analysis_layers

        # Hooks for intermediate features storage
        if len(analysis_layers):
            self._features = {layer: th.empty(0) for layer in self.analysis_layers}
            for layer_id in analysis_layers:
                layer = dict([*self.named_modules()])[layer_id]
                layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def forward(self, observations, prev_gw, masks, prev_modality_features):
        if masks.dim() == 2: # single step, during eval
            observations = {
                k: v[:, None, :] for k, v in observations.items()
                if k in ["spectrogram", "rgb"]
            }
            masks = masks[:, None, :]

        B, T = observations["spectrogram"].shape[:2]
        modality_features = {}

        # List of state features for actor-critic
        gw_list = []
        modality_features_list = {"audio": [], "visual": []}

        for t in range(T):
            # Extracts audio featues
            obs_dict_t = {
                k: v[:, t] for k, v in observations.items()
                if k in ["spectrogram", "audiogoal", "depth", "rgb"]
            }
            masks_t = masks[:, t]

            audio_features = \
                self.audio_encoder(
                    obs_dict_t,
                    prev_modality_features["audio"],
                    masks_t,
                    prev_gw=prev_gw)
            modality_features["audio"] = audio_features
            modality_features_list["audio"].append(audio_features)

            # Extracts vision features
            if not self.visual_encoder.is_blind:
                visual_features = \
                    self.visual_encoder(
                        obs_dict_t,
                        prev_modality_features["visual"], 
                        masks_t,
                        prev_gw=prev_gw)
                modality_features["visual"] = visual_features
                modality_features_list["visual"].append(visual_features)

            gw = self.state_encoder(
                # Reshape each mod feat to [B, 1, H]
                modality_features = {
                    k: v[:, None, :] for k, v in modality_features.items()
                },
                prev_gw=prev_gw,
                masks=masks_t
            ) # [B, num_latents, latent_dim]
            gw_list.append(gw)

        # TODO: consider dropping legacy encoder name support (YYY.cnn.7)
        if "visual_encoder" in self.analysis_layers:
            self._features["visual_encoder.rnn"] = th.cat(modality_features_list["visual"], dim=0)
        if "audio_encoder" in self.analysis_layers:
            self._features["audio_encoder.rnn"] = th.cat(modality_features_list["audio"], dim=0)

        # Returns gw_list as B * T, GW_H, to mach "action_list" used for CE loss
        return th.stack(gw_list).permute(1, 0, 2).reshape(B * T, -1), \
               gw, modality_features # [B * T, H], [B, H], {k: [B, H]} for k in "visual", "audio"

    def act(self, observations, rnn_hidden_states, masks, modality_features, deterministic=False, actions=None):
        features, gw, modality_features = self(observations, rnn_hidden_states,
                                            masks, modality_features)

        # Estimate the policy as distribution to sample actions from
        distribution, action_logits = self.action_distribution(features)

        if actions is None:
            if deterministic:
                actions = distribution.mode()
            else:
                actions = distribution.sample()
        action_log_probs = distribution.log_probs(actions)

        distribution_entropy = distribution.entropy()

        modality_features_detached = {
            k: v.detach() for k,v in modality_features.items()
        }

        # Compatibility for probing layers of GWT v2 agents
        # Override the features of shape [B, H] to [B*T, H] instead
        if "state_encoder" in self.analysis_layers:
            self._features["state_encoder"] = features

        return actions, distribution.probs, action_log_probs, action_logits, \
               distribution_entropy, gw.detach(), modality_features_detached

    def get_grad_norms(self):
        modules = ["visual_encoder", "audio_encoder", "state_encoder", "action_distribution"]
        grad_norm_dict = {mod_name: compute_grad_norm(self.__getattr__(mod_name)) for mod_name in modules}
        
        return grad_norm_dict
    
    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

# endregion: GWT Agent         #
###################################

###################################
# region: GRU v3 Baseline         #

class GRU_Actor(nn.Module):
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
            config.gw_size,
            config.hidden_size,
            extra_rgb=extra_rgb,
            use_gw=config.recenc_use_gw,
            gw_detach=config.recenc_gw_detach,
            gru_type=config.gru_type,
        )
        self.audio_encoder = RecurrentAudioEncoder(
            observation_space,
            config.gw_size,
            config.hidden_size,
            "spectrogram",
            use_gw=config.recenc_use_gw,
            gw_detach=config.recenc_gw_detach,
            gru_type=config.gru_type,
        )
        
        self.state_encoder = GRUStateEncoder(
            gw_size=config.gw_size,
            feat_size=config.hidden_size,
            gru_type=config.gru_type
        )

        self.action_distribution = CategoricalNet(config.gw_size, action_space.n) # Policy fn

        self.train()
        
        # Layers to record for neuroscience based analysis
        self.analysis_layers = analysis_layers

        # Hooks for intermediate features storage
        if len(analysis_layers):
            self._features = {layer: th.empty(0) for layer in self.analysis_layers}
            for layer_id in analysis_layers:
                layer = dict([*self.named_modules()])[layer_id]
                layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def forward(self, observations, prev_gw, masks, prev_modality_features):
        if masks.dim() == 2: # single step, during eval
            observations = {
                k: v[:, None, :] for k, v in observations.items()
                if k in ["spectrogram", "rgb"]
            }
            masks = masks[:, None, :]

        B, T = observations["spectrogram"].shape[:2]
        modality_features = {}

        # List of state features for actor-critic
        gw_list = []
        modality_features_list = {"audio": [], "visual": []}

        for t in range(T):
            # Extracts audio featues
            obs_dict_t = {
                k: v[:, t] for k, v in observations.items()
                if k in ["spectrogram", "audiogoal", "depth", "rgb"]
            }
            masks_t = masks[:, t]

            audio_features = \
                self.audio_encoder(
                    obs_dict_t,
                    prev_modality_features["audio"],
                    masks_t,
                    prev_gw=prev_gw)
            modality_features["audio"] = audio_features
            modality_features_list["audio"].append(audio_features)

            # Extracts vision features
            if not self.visual_encoder.is_blind:
                visual_features = \
                    self.visual_encoder(
                        obs_dict_t,
                        prev_modality_features["visual"], 
                        masks_t,
                        prev_gw=prev_gw)
                modality_features["visual"] = visual_features
                modality_features_list["visual"].append(visual_features)

            gw = self.state_encoder(
                # Reshape each mod feat to [B, 1, H]
                modality_features = {
                    k: v[:, None, :] for k, v in modality_features.items()
                },
                prev_gw=prev_gw,
                masks=masks_t
            ) # [B, GW_H]
            gw_list.append(gw)

        # TODO: consider dropping legacy encoder name support (YYY.cnn.7)
        if "visual_encoder" in self.analysis_layers:
            self._features["visual_encoder.rnn"] = th.cat(modality_features_list["visual"], dim=0)
        if "audio_encoder" in self.analysis_layers:
            self._features["audio_encoder.rnn"] = th.cat(modality_features_list["audio"], dim=0)

        # Returns gw_list as B * T, GW_H, to mach "action_list" used for CE loss
        return th.stack(gw_list).permute(1, 0, 2).reshape(B * T, -1), \
               gw, modality_features # [B * T, H], [B, H], {k: [B, H]} for k in "visual", "audio"

    def act(self, observations, rnn_hidden_states, masks, modality_features, deterministic=False, actions=None):
        features, gw, modality_features = self(observations, rnn_hidden_states,
                                            masks, modality_features)

        # Estimate the policy as distribution to sample actions from
        distribution, action_logits = self.action_distribution(features)

        if actions is None:
            if deterministic:
                actions = distribution.mode()
            else:
                actions = distribution.sample()
        action_log_probs = distribution.log_probs(actions)

        distribution_entropy = distribution.entropy()

        modality_features_detached = {
            k: v.detach() for k,v in modality_features.items()
        }
        
        # Compatibility for probing layers of GWT v2 agents
        # Override the features of shape [B, H] to [B*T, H] instead
        if "state_encoder" in self.analysis_layers:
            self._features["state_encoder"] = features
        
        return actions, distribution.probs, action_log_probs, action_logits, \
               distribution_entropy, gw.detach(), modality_features_detached

    def get_grad_norms(self):
        modules = ["visual_encoder", "audio_encoder", "state_encoder", "action_distribution"]
        grad_norm_dict = {mod_name: compute_grad_norm(self.__getattr__(mod_name)) for mod_name in modules}
        
        return grad_norm_dict
    
    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

# endregion: GRU v3 Baseline      #
###################################