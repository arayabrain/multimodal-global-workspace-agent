import torch
from torch import nn
from torch.distributions import Categorical


class Agent(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.vision_encoder = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(), nn.Flatten(), nn.Linear(2304, hidden_size))
    self.audio_encoder = nn.Sequential(nn.Conv2d(2, 32, 5, stride=2), nn.ReLU(), nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(), nn.Linear(2496, hidden_size))
    self.proprio_encoder = nn.Linear(4, hidden_size)
    self.working_memory = nn.GRUCell(3 * hidden_size, hidden_size)
    self.policy = nn.Sequential(nn.Linear(4 * hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 4))

  def forward(self, rgb, spectrogram, pose, memory):
    enc_vision, enc_audio, enc_proprio = self.vision_encoder(rgb), self.audio_encoder(spectrogram), self.proprio_encoder(pose)
    # TODO: SA + CA
    enc_sensor = torch.cat([enc_vision, enc_audio, enc_proprio], dim=1)
    memory = self.working_memory(enc_sensor, memory)
    logits = self.policy(torch.cat([enc_sensor, memory], dim=1))
    return Categorical(logits=logits), memory


H = 512
rgb, spectrogram, pose = torch.zeros(1, 3, 128, 128), torch.zeros(1, 2, 65, 26), torch.zeros(1, 4)
memory = torch.zeros(1, H)
agent = Agent(H)
policy, memory = agent(rgb, spectrogram, pose, memory)
