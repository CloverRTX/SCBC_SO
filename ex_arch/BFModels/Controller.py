from torch import nn
import torch.nn.functional as F
import torch

class ColNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # return F.relu(self.fc2(x))
        return torch.tanh(self.fc2(x)) * self.action_bound