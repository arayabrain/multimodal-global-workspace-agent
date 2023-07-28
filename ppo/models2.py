from typing import Callable

import torch
import torch as th
from torch import nn
from torch.nn import functional as F

from models import RNNStateEncoder, compute_grad_norm
from ss_baselines.av_nav.ppo.policy import CriticHead
from ss_baselines.common.utils import CategoricalNet2

GWTAGENT_DEFAULT_ANALYSIS_LAYER_NAMES = [
  # "visual_encoder", "audio_encoder",
  "visual_attention", "audio_attention",
  "memory_vision_att_embedding", "memory_audio_att_embedding",
  "visual_embedding", "audio_embedding",
  "state_encoder",
  "action_distribution",
  "critic"
]
class Attention2d(nn.Module):
  def __init__(self, query_channels, key_value_channels):
    super().__init__()
    self.att_channels = query_channels // 8
    self.Wf = nn.Conv2d(query_channels, self.att_channels, 1)
    self.Wg = nn.Conv2d(key_value_channels, self.att_channels, 1)
    self.Wh = nn.Conv2d(key_value_channels, key_value_channels, 1)
    self.gamma = nn.Parameter(torch.zeros(1))  # Initialise attention at 0

  def forward(self, query_input, key_value_input):
    B, C, H, W = key_value_input.size()
    f = self.Wf(query_input).view(B, self.att_channels, -1).permute(0, 2, 1)  # Query
    g = self.Wg(key_value_input).view(B, self.att_channels, -1)  # Key
    h = self.Wh(key_value_input).view(B, C, -1).permute(0, 2, 1)  # Value
    beta = F.softmax(f @ g, dim=2)  # Attention
    o = (beta @ h).permute(0, 2, 1).view(B, C, H, W)
    y = self.gamma * o + key_value_input
    return y

class VisionEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.channels = channels = config.gwt_channels

    self.conv1 = nn.Conv2d(3, channels, 8, stride=4)
    self.conv2 = nn.Conv2d(channels, 2 * channels, 4, stride=2)
    self.conv3 = nn.Conv2d(2 * channels, 2 * channels, 3, stride=2)

  def forward(self, x):
    x = x.permute(0, 3, 1, 2) / 255.0
    if self.config.obs_center:
        x -= 0.5
    
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    return x


class AudioEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.channels = channels = config.gwt_channels

    self.conv1 = nn.Conv2d(2, channels, 5, stride=2)
    self.conv2 = nn.Conv2d(channels, 2 * channels, 3, stride=2)
    self.conv3 = nn.Conv2d(2 * channels, 2 * channels, 3, stride=1)

  def forward(self, x):
    x = x.permute(0, 3, 1, 2)
    
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    return x

class GWTAgent(nn.Module):
  def __init__(self, action_space, config, analysis_layers=[]):
    super().__init__()
    self.config = config

    # TODO: make this parameterizable from the command line
    hidden_size = config.gwt_hid_size
    channels = config.gwt_channels

    # Basic encoders for multimodal observations
    self.visual_encoder = VisionEncoder(config)
    self.audio_encoder = AudioEncoder(config)
    input_dim = hidden_size * 2

    # Multiple modality encoders with bottom-up attention
    self.visual_attention = Attention2d(4 * channels, 2 * channels)
    self.audio_attention = Attention2d(4 * channels, 2 * channels)
    
    # Top-down attention module: query is based on memory, attend to each modality
    self.memory_vision_att_embedding = nn.Linear(hidden_size, 72 * channels)
    self.memory_audio_att_embedding = nn.Linear(hidden_size, 78 * channels)

    self.visual_embedding =  nn.Sequential(nn.Flatten(), nn.Linear(72 * channels, hidden_size))
    self.audio_embedding = nn.Sequential(nn.Flatten(), nn.Linear(78 * channels, hidden_size))

    # self.state_encoder = RNNStateEncoder(input_dim, hidden_size)
    self.state_encoder = nn.GRUCell(input_dim, hidden_size)

    # input_dim for policy / value is num_latents * latent_dim of the Perceiver
    self.action_distribution = CategoricalNet2(hidden_size * 3, hidden_size, action_space.n) # Policy fn
    self.critic = CriticHead(hidden_size * 3) # Value fn

    self.train()

    # Layers to record for neuroscience based analysis
    self.analysis_layers = analysis_layers

    # Hooks for intermediate features storage
    # TODO: redefine the analysis layer for this type of agent too
    if len(analysis_layers):
        self._features = {layer: th.empty(0) for layer in self.analysis_layers}
        for layer_id in analysis_layers:
            layer = dict([*self.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
    
  def save_outputs_hook(self, layer_id: str) -> Callable:
      def fn(_, __, output):
          self._features[layer_id] = output
      return fn
        
  def forward(self, observations, rnn_hidden_states, masks, prev_actions=None):
    if rnn_hidden_states.dim() == 2: # single step
      B = rnn_hidden_states.shape[0] # Batch size for either train or eval
    elif rnn_hidden_states.dim() == 3: # seq. forward
      B = rnn_hidden_states.shape[1]
      rnn_hidden_states = rnn_hidden_states[0] # Undoe the .unsqueeze() needed by SS baseline for seq_forward
    T = observations["spectrogram"].shape[0] // B

    modality_features = {}
    
    rnn_hidden_states2 = [] # Accumulate internal RNN hidden states

    # Extarct visual and audio features
    # TODO: eventually add suppoort for proprio ?
    vision_features = self.visual_encoder(observations["rgb"])
    audio_features = self.audio_encoder(observations["spectrogram"])

    visual_features_final, audio_features_final = [], []

    for t, (t_vis_feats, t_aud_feats, t_masks) in \
      enumerate(zip(
          vision_features.reshape(T, B, *vision_features.shape[1:]),
          audio_features.reshape(T, B, *audio_features.shape[1:]),
          masks.reshape(T, B, -1)
      )):

      rnn_hidden_states *= t_masks # Apply reset masks that are based on ep. termination
      
      # 
      memory_vision_query = self.memory_vision_att_embedding(rnn_hidden_states).reshape_as(t_vis_feats)
      memory_audio_query = self.memory_audio_att_embedding(rnn_hidden_states).reshape_as(t_aud_feats)

      # 
      att_vision = self.visual_attention(th.cat([memory_vision_query, t_vis_feats], dim=1), t_vis_feats)
      att_audio = self.audio_attention(th.cat([memory_audio_query, t_aud_feats], dim=1), t_aud_feats)

      # 
      vision_emb = self.visual_embedding(att_vision)
      audio_emb = self.audio_embedding(att_audio)
      
      visual_features_final.append(vision_emb)
      audio_features_final.append(audio_emb)

      att_sensor = th.cat([vision_emb, audio_emb], dim=1)

      rnn_hidden_states = self.state_encoder(att_sensor, rnn_hidden_states)
      rnn_hidden_states2.append(rnn_hidden_states)

    rnn_hidden_states2 = th.cat(rnn_hidden_states2, dim=0)
    visual_features_final = th.cat(visual_features_final, dim=0)
    audio_features_final = th.cat(audio_features_final, dim=0)
    
    modality_features["vision"] = visual_features_final
    modality_features["audio"] = audio_features_final

    # Pass rnn_hidden_states2 twice for backward compat. with previous code
    return rnn_hidden_states2, rnn_hidden_states2, modality_features

  def act(self, observations, rnn_hidden_states, masks, deterministic=False, actions=None, prev_actions=None,
                value_feat_detach=False, actor_feat_detach=False, ssl_tasks=None):
    features, rnn_hidden_states, modality_features = self(observations, rnn_hidden_states, masks, prev_actions=prev_actions)
    polval_features = th.cat([
      *[v for v in modality_features.values()],
      features
      ], dim=1)

    # Estimate the value funciton
    values = self.critic(polval_features.detach() if value_feat_detach else polval_features)

    # Estimate the policy as a distribution to sample actions from. Also recover the logits
    distribution, action_logits = self.action_distribution(polval_features.detach() if actor_feat_detach else polval_features)

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

    return actions, distribution.probs, action_log_probs, action_logits, \
            distribution_entropy, values, rnn_hidden_states, ssl_outputs

  def get_value(self, observations, rnn_hidden_states, masks, prev_actions=None):
    features, _, modality_features = self(observations, rnn_hidden_states, masks, prev_actions=prev_actions)
    polval_features = th.cat([
      *[v for v in modality_features.values()],
      features
      ], dim=1)
    
    return self.critic(polval_features)

  def get_grad_norms(self):
    modules = [
      "visual_encoder", "audio_encoder",
      "visual_attention", "audio_attention",
      "memory_vision_att_embedding", "memory_audio_att_embedding",
      "visual_embedding", "audio_embedding",
      "state_encoder",
      "action_distribution", "critic"
    ]

    return {mod_name: compute_grad_norm(self.__getattr__(mod_name)) for mod_name in modules}

class GWTAgent_BU(GWTAgent):
  def __init__(self, action_space, config, analysis_layers=[]):
    super().__init__(action_space, config, analysis_layers)
    # TODO: make this parameterizable from the command line
    hidden_size = config.gwt_hid_size
    channels = config.gwt_channels

    # Overrdie the Attention2D mechanism to only perform bottom-up SA
    # Multiple modality encoders with bottom-up attention
    self.visual_attention = Attention2d(2 * channels, 2 * channels)
    self.audio_attention = Attention2d(2 * channels, 2 * channels)

    self.train()

    # Layers to record for neuroscience based analysis
    self.analysis_layers = analysis_layers

    # Hooks for intermediate features storage
    # TODO: redefine the analysis layer for this type of agent too
    if len(analysis_layers):
        self._features = {layer: th.empty(0) for layer in self.analysis_layers}
        for layer_id in analysis_layers:
            layer = dict([*self.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

  def forward(self, observations, rnn_hidden_states, masks, prev_actions=None):
    if rnn_hidden_states.dim() == 2: # single step
      B = rnn_hidden_states.shape[0] # Batch size for either train or eval
    elif rnn_hidden_states.dim() == 3: # seq. forward
      B = rnn_hidden_states.shape[1]
      rnn_hidden_states = rnn_hidden_states[0] # Undoe the .unsqueeze() needed by SS baseline for seq_forward
    T = observations["spectrogram"].shape[0] // B

    modality_features = {}
    
    rnn_hidden_states2 = [] # Accumulate internal RNN hidden states

    # Extarct visual and audio features
    # TODO: eventually add suppoort for proprio ?
    vision_features = self.visual_encoder(observations["rgb"])
    audio_features = self.audio_encoder(observations["spectrogram"])

    visual_features_final, audio_features_final = [], []

    for t, (t_vis_feats, t_aud_feats, t_masks) in \
      enumerate(zip(
          vision_features.reshape(T, B, *vision_features.shape[1:]),
          audio_features.reshape(T, B, *audio_features.shape[1:]),
          masks.reshape(T, B, -1)
      )):

      rnn_hidden_states *= t_masks # Apply reset masks that are based on ep. termination

      # 
      att_vision = self.visual_attention(t_vis_feats, t_vis_feats)
      att_audio = self.audio_attention(t_aud_feats, t_aud_feats)

      # 
      vision_emb = self.visual_embedding(att_vision)
      audio_emb = self.audio_embedding(att_audio)
      
      visual_features_final.append(vision_emb)
      audio_features_final.append(audio_emb)

      att_sensor = th.cat([vision_emb, audio_emb], dim=1)

      rnn_hidden_states = self.state_encoder(att_sensor, rnn_hidden_states)
      rnn_hidden_states2.append(rnn_hidden_states)

    rnn_hidden_states2 = th.cat(rnn_hidden_states2, dim=0)
    visual_features_final = th.cat(visual_features_final, dim=0)
    audio_features_final = th.cat(audio_features_final, dim=0)
    
    modality_features["vision"] = visual_features_final
    modality_features["audio"] = audio_features_final

    # Pass rnn_hidden_states2 twice for backward compat. with previous code
    return rnn_hidden_states2, rnn_hidden_states2, modality_features

class GWTAgent_TD(GWTAgent_BU):
  def forward(self, observations, rnn_hidden_states, masks, prev_actions=None):
    if rnn_hidden_states.dim() == 2: # single step
      B = rnn_hidden_states.shape[0] # Batch size for either train or eval
    elif rnn_hidden_states.dim() == 3: # seq. forward
      B = rnn_hidden_states.shape[1]
      rnn_hidden_states = rnn_hidden_states[0] # Undoe the .unsqueeze() needed by SS baseline for seq_forward
    T = observations["spectrogram"].shape[0] // B

    modality_features = {}
    
    rnn_hidden_states2 = [] # Accumulate internal RNN hidden states

    # Extarct visual and audio features
    # TODO: eventually add suppoort for proprio ?
    vision_features = self.visual_encoder(observations["rgb"])
    audio_features = self.audio_encoder(observations["spectrogram"])

    visual_features_final, audio_features_final = [], []

    for t, (t_vis_feats, t_aud_feats, t_masks) in \
      enumerate(zip(
          vision_features.reshape(T, B, *vision_features.shape[1:]),
          audio_features.reshape(T, B, *audio_features.shape[1:]),
          masks.reshape(T, B, -1)
      )):

      rnn_hidden_states *= t_masks # Apply reset masks that are based on ep. termination

      raise NotImplementedError(f"Forward pass for GWTAgent_TD not implemented yet.")
      # 
      att_vision = self.visual_attention(t_vis_feats, t_vis_feats)
      att_audio = self.audio_attention(t_aud_feats, t_aud_feats)

      # 
      vision_emb = self.visual_embedding(att_vision)
      audio_emb = self.audio_embedding(att_audio)
      
      visual_features_final.append(vision_emb)
      audio_features_final.append(audio_emb)

      att_sensor = th.cat([vision_emb, audio_emb], dim=1)

      rnn_hidden_states = self.state_encoder(att_sensor, rnn_hidden_states)
      rnn_hidden_states2.append(rnn_hidden_states)

    rnn_hidden_states2 = th.cat(rnn_hidden_states2, dim=0)
    visual_features_final = th.cat(visual_features_final, dim=0)
    audio_features_final = th.cat(audio_features_final, dim=0)
    
    modality_features["vision"] = visual_features_final
    modality_features["audio"] = audio_features_final

    # Pass rnn_hidden_states2 twice for backward compat. with previous code
    return rnn_hidden_states2, rnn_hidden_states2, modality_features