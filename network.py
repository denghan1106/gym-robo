import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = self.net(x)
        std = torch.exp(self.log_std)
        return self.mean(x), std

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)
