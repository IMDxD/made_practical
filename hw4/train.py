import copy

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

GAMMA = 0.99
TAU = 0.002
ACTOR_CLIP = 1
EPS = 0.02
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
DEVICE = "cuda"
BATCH_SIZE = 128
ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 2_000_000


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class TransitionDataset(Dataset):
    def __getitem__(self, index):
        state, action, next_state, reward, done = self.transitions[index]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.float32),
        )

    def __init__(self, data):
        self.transitions = data

    def __len__(self):
        return len(self.transitions)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim, alpha=1e-1):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=ACTOR_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=ACTOR_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self._alpha = alpha

    def update(self, batch):

        state, action, next_state, reward, done = batch
        state = state.to(DEVICE)
        action = action.to(DEVICE)
        next_state = next_state.to(DEVICE)
        reward = reward.to(DEVICE)
        done = done.to(DEVICE)

        with torch.no_grad():

            noise = torch.randn_like(action).to(DEVICE)
            next_actor_action = self.target_actor(next_state)
            next_actor_action = torch.clamp(
                next_actor_action + EPS * noise, -ACTOR_CLIP, ACTOR_CLIP
            )
            target_q1 = self.target_critic_1(next_state, next_actor_action)
            target_q2 = self.target_critic_2(next_state, next_actor_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward.squeeze(1) + (1 - done.squeeze(1)) * GAMMA * target_q

        self.critic_1_optim.zero_grad()
        self.critic_2_optim.zero_grad()
        self.actor_optim.zero_grad()

        actor_action = self.actor(state)
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        loss_critic_1 = F.mse_loss(current_q1, target_q)
        loss_critic_2 = F.mse_loss(current_q2, target_q)
        loss_critic_1.backward()
        loss_critic_2.backward()

        self.critic_1_optim.step()
        self.critic_2_optim.step()

        loss_actor = -self.critic_1(state, actor_action).mean()
        loss_actor.backward()
        self.actor_optim.step()

        soft_update(self.target_critic_1, self.critic_1)
        soft_update(self.target_critic_2, self.critic_2)
        soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()[0]

    def save(self):
        torch.save(self.actor.state_dict(), "agent.pkl")


def train():
    model = TD3(19, 5)
    suboptimal = np.load("suboptimal.npz", allow_pickle=True)["arr_0"]
    optimal = np.load("optimal.npz", allow_pickle=True)["arr_0"]
    dataset = TransitionDataset(np.vstack((optimal, suboptimal)))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for _ in tqdm.tqdm(range(1000)):
        for batch in dataloader:
            model.update(batch)

    return model


if __name__ == "__main__":
    model = train()
    model.save()
