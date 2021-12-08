import copy
import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

GAMMA = 0.99
TAU = 0.002
ACTOR_CLIP = 1
EPS = 0.02
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
DEVICE = "cuda"
BATCH_SIZE = 128
TRANSITIONS = 1_000_000


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
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
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=ACTOR_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=ACTOR_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.replay_buffer = deque(maxlen=200000)

    def update(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)] for _ in
                           range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)

            # Update actor

            with torch.no_grad():

                noise = torch.randn_like(action).to(DEVICE)
                next_actor_action = self.target_actor(next_state)
                next_actor_action = torch.clamp(next_actor_action + EPS * noise, -ACTOR_CLIP, ACTOR_CLIP)
                target_q1 = self.target_critic_1(next_state, next_actor_action)
                target_q2 = self.target_critic_2(next_state, next_actor_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * GAMMA * target_q

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


class WorldModel(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size):
        super(WorldModel, self).__init__()
        self.rnn = nn.LSTM(state_dim + action_dim, hidden_size, bidirectional=False, batch_first=True)
        self.state_head = nn.Linear(hidden_size, state_dim)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, states, actions, length):
        x = torch.cat((states, actions), dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        out_packed, (_, _) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        next_state = self.state_head(out)
        reward = self.reward_head(out)
        done = self.done_head(out)
        return next_state, reward, done


class WorldModelEnv:

    def __init__(self):
        self.model = WorldModel(19, 5, 256)
        state_dict = torch.load(__file__[:-8] + "/world.pkl", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.h = torch.zeros(1, 256)
        self.c = torch.zeros(1, 256)
        self.initial_steps = np.load(__file__[:-8] + "/initial_state.npy")
        self.current_state = self.choose_state()

    def choose_state(self):
        idx = np.random.randint(self.initial_steps.shape[0])
        return torch.from_numpy(self.initial_steps[idx]).view(1, 1, -1).float()

    def reset(self):
        self.h = torch.zeros(1, 1, 256, dtype=torch.float32)
        self.c = torch.zeros(1, 1, 256, dtype=torch.float32)
        self.current_state = self.choose_state()
        return self.current_state.numpy().ravel()

    def step(self, action):
        with torch.no_grad():
            action = torch.from_numpy(action).view(1, 1, -1).float()
            x = torch.cat((self.current_state, action), dim=-1)
            out, (h, c) = self.model.rnn(x, (self.h, self.c))
            next_state = self.model.state_head(out)
            reward = self.model.reward_head(out)
            done = self.model.done_head(out)
            self.h = h
            self.c = c
            self.current_state = next_state
            next_state = next_state.numpy().ravel()
            reward = reward.item()
            done = done.item() > 0.5
        return next_state, reward, done, None


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = WorldModelEnv()
    test_env = WorldModelEnv()
    td3 = TD3(state_dim=19, action_dim=5)
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0

    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        action = td3.act(state)
        action = np.clip(action + EPS * np.random.randn(*action.shape), -ACTOR_CLIP, ACTOR_CLIP)

        next_state, reward, done, _ = env.step(action)
        td3.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(test_env, td3, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            td3.save()
