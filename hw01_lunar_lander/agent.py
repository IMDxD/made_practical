import numpy as np
import torch
from torch import nn


class QModel(nn.Module):
    def __init__(self, state_dim, action_dim, fc1=512, fc2=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, fc1),
            nn.BatchNorm1d(fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.BatchNorm1d(fc2),
            nn.ReLU(),
            nn.Linear(fc2, action_dim),
        )

    def forward(self, x):
        return self.fc(x)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)


class Agent:
    def __init__(self):
        self.model = QModel(8, 4)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.eval()

    def act(self, state):
        state = torch.tensor(np.array(state)).view(1, -1)
        action_rewards = self.model(state).squeeze(0).detach().cpu().numpy()
        return np.argmax(action_rewards)
