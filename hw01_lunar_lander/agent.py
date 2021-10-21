import random
import numpy as np
import os
import torch


DEVICE = torch.device("cuda")


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        state = torch.tensor(np.array(state)).view(1, -1).to(DEVICE)
        action_rewards = self.model(state).squeeze(0).detach().cpu().numpy()
        return np.argmax(action_rewards)

