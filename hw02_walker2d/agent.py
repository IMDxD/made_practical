import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class Agent(nn.Module):

    def __init__(self):
        super().__init__()
        state_dict = torch.load(__file__[:-8] + "/agent.pkl")
        self.model = self.model = nn.Sequential(
            nn.Linear(22, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 6)
        )
        self.log_sigma = nn.Parameter(torch.zeros(1, 6))
        self.load_state_dict(state_dict)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            action_means = self.model(state)
            distr = Normal(action_means, torch.exp(self.log_sigma))
            return torch.tanh(distr.sample()).squeeze(0)

    def reset(self):
        pass
