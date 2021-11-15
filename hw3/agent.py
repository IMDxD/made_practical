import numpy as np
import torch
from torch import nn


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 8),
            nn.Tanh()
        )
        state_dict = torch.load(__file__[:-8] + "/agent.pkl")
        self.load_state_dict(state_dict)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            return self.model(state).cpu().numpy()[0]

    def reset(self):
        pass
