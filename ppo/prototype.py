import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


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
  def __init__(self, channels):
    super().__init__()
    self.conv1 = nn.Conv2d(3, channels, 8, stride=4)
    self.conv2 = nn.Conv2d(channels, 2 * channels, 4, stride=2)
    self.conv3 = nn.Conv2d(2 * channels, 2 * channels, 3, stride=2)

  def forward(self, rgb):
    h = F.relu(self.conv1(rgb))
    h = F.relu(self.conv2(h))
    h = F.relu(self.conv3(h))
    return h


class AudioEncoder(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.conv1 = nn.Conv2d(2, channels, 5, stride=2)
    self.conv2 = nn.Conv2d(channels, 2 * channels, 3, stride=2)
    self.conv3 = nn.Conv2d(2 * channels, 2 * channels, 3, stride=1)

  def forward(self, spectrogram):
    h = F.relu(self.conv1(spectrogram))
    h = F.relu(self.conv2(h))
    h = F.relu(self.conv3(h))
    return h


class Agent(nn.Module):
  def __init__(self, hidden_size, channels):
    super().__init__()
    self.vision_encoder, self.audio_encoder = VisionEncoder(channels), AudioEncoder(channels)
    self.memory_vision_att_embedding, self.memory_audio_att_embedding = nn.Linear(hidden_size, 72 * channels), nn.Linear(hidden_size, 78 * channels)
    self.vision_attention, self.audio_attention = Attention2d(4 * channels, 2 * channels), Attention2d(4 * channels, 2 * channels)
    self.vision_embedding, self.audio_embedding = nn.Sequential(nn.Flatten(), nn.Linear(72 * channels, hidden_size)), nn.Sequential(nn.Flatten(), nn.Linear(78 * channels, hidden_size))
    self.working_memory = nn.GRUCell(2 * hidden_size, hidden_size)
    self.policy = nn.Sequential(nn.Linear(3 * hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 4))

  def forward(self, rgb, spectrogram, memory):
    enc_vision, enc_audio = self.vision_encoder(rgb), self.audio_encoder(spectrogram)
    memory_vision_query, memory_audio_query = self.memory_vision_att_embedding(memory).reshape_as(enc_vision), self.memory_audio_att_embedding(memory).reshape_as(enc_audio)
    att_vision, att_audio = self.vision_attention(torch.cat([memory_vision_query, enc_vision], dim=1), enc_vision), self.audio_attention(torch.cat([memory_audio_query, enc_audio], dim=1), enc_audio)
    att_sensor = torch.cat([self.vision_embedding(att_vision), self.audio_embedding(att_audio)], dim=1)
    memory = self.working_memory(att_sensor, memory)
    logits = self.policy(torch.cat([att_sensor, memory], dim=1))
    return Categorical(logits=logits), memory


H, C = 512, 32
rgb, spectrogram = torch.zeros(1, 3, 128, 128), torch.zeros(1, 2, 65, 26)
memory = torch.zeros(1, H)
agent = Agent(H, C)
policy, memory = agent(rgb, spectrogram, memory)
