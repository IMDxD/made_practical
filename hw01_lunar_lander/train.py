import random
from collections import deque

import numpy as np
import torch
from gym import make
from torch import nn
from torch.optim import Adam, lr_scheduler

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 2_000_000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
BUFFER_SIZE = 128_000
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda")


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


class ExpirienceReplay(deque):
    def sample(self, size):
        batch = random.sample(self, size)
        return list(zip(*batch))


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change
        self.model = QModel(state_dim, action_dim).to(DEVICE)  # Torch model
        self.target_model = QModel(state_dim, action_dim).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = ExpirienceReplay(maxlen=BUFFER_SIZE)
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criteria = nn.MSELoss()
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.8
        )

    def consume_transition(self, transition):
        self.buffer.append(transition)

    def sample_batch(self):
        batch = self.buffer.sample(BATCH_SIZE)
        state, action, next_state, reward, done = batch
        state = torch.tensor(np.array(state, dtype=np.float32))
        action = torch.tensor(np.array(action, dtype=np.int64))
        next_state = torch.tensor(np.array(next_state, dtype=np.float32))
        reward = torch.tensor(np.array(reward, dtype=np.float32))
        done = torch.tensor(np.array(done, dtype=np.int32))
        return state, action, next_state, reward, done

    def train_step(self, batch):
        if not self.model.training:
            self.model.train()
        self.optimizer.zero_grad()
        state, action, next_state, reward, done = batch
        current_q = self.model(state.to(DEVICE))
        next_q = self.model(state.to(DEVICE))
        next_action = torch.argmax(next_q, 1)
        next_target_q = self.target_model(next_state.to(DEVICE))
        action_reward = current_q.gather(1, action.view(-1, 1).to(DEVICE))
        next_actions_reward = next_target_q.gather(1, next_action.view(-1, 1))
        next_actions_reward = next_actions_reward.squeeze(1) * (1 - done.to(DEVICE))
        loss = self.criteria(
            action_reward.squeeze(1), reward.to(DEVICE) + GAMMA * next_actions_reward
        )
        loss.backward()
        self.optimizer.step()
        if self.steps > 1_000_000:
            self.scheduler.step(loss)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, target=False):
        if self.model.training:
            self.model.eval()
        network = self.target_model if target else self.model
        state = torch.tensor(np.array(state)).view(1, -1).to(DEVICE)
        action_rewards = network(state).squeeze(0).detach().cpu().numpy()
        return np.argmax(action_rewards)

    def update(self, transition):
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model.state_dict(), "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    agent.model.eval()
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.0

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    agent.model.train()
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    eps_decay = 2
    eps_min = 0.01

    state = env.reset()

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):

        if i % 25_000 == 0:
            eps = max(eps / eps_decay, eps_min)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(
                f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}"
            )
            dqn.save()
