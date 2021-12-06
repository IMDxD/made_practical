import copy

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.distributions import Normal, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

GAMMA = 0.99
TAU = 0.002
ACTOR_CLIP = 1
EPS = 0.02
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
DEVICE = "cuda"
BATCH_SIZE = 128


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


class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        action_distribution = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean, log_std):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        action_distribution = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )

        action_sample = action_distribution.rsample()

        log_prob = torch.sum(
            action_distribution.log_prob(action_sample), dim=-1
        )

        return action_sample, log_prob


class Scalar(nn.Module):

    def __init__(self, value):
        super(Scalar, self).__init__()
        self.param = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self):
        return self.param


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 2 * action_dim),
        )
        self.action_dim = action_dim
        self.log_std_multiplier = Scalar(1.0)
        self.log_std_offset = Scalar(-1.0)
        self.tanh_gaussian = ReparameterizedTanhGaussian()

    def log_prob(self, state, actions):

        output = self.model(state)
        mean, log_std = torch.split(output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, state, repeat=None):
        if repeat is not None:
            state = state.unsqueeze(1)
            state = torch.repeat_interleave(state, repeat, 1)
        base_network_output = self.model(state)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std)


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
        return self.model(torch.cat([state, action], dim=-1))


class Agent:

    def __init__(self, state_dim, action_dim):

        self.policy = Actor(state_dim, action_dim).to(DEVICE)
        self.qf1 = Critic(state_dim, action_dim).to(DEVICE)
        self.qf2 = Critic(state_dim, action_dim).to(DEVICE)
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=ACTOR_LR)
        self.qf1_optimizer = Adam(self.qf1.parameters(), lr=CRITIC_LR)
        self.qf2_optimizer = Adam(self.qf2.parameters(), lr=CRITIC_LR)

        self.log_alpha = Scalar(0.0).to(DEVICE)

        self.alpha_optimizer = torch.optim.Adam(
            self.log_alpha.parameters(),
            lr=3e-4,
        )

    def update(self, batch):

        state, action, next_state, reward, done = batch
        state = state.to(DEVICE)
        action = action.to(DEVICE)
        next_state = next_state.to(DEVICE)
        reward = reward.to(DEVICE)
        done = done.to(DEVICE)

        new_actions, log_pi = self.policy(state)

        alpha_loss = -(self.log_alpha() * log_pi.detach()).mean()
        alpha = self.log_alpha().exp()

        """ Policy loss """
        q_new_actions = torch.min(
            self.qf1(state, new_actions),
            self.qf2(state, new_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """ Q function loss """
        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)
        with torch.no_grad():
            new_next_actions, next_log_pi = self.policy(next_state)
            target_q_values = torch.min(
                self.target_qf1(next_state, new_next_actions),
                self.target_qf2(next_state, new_next_actions),
            )

            q_target = reward + (1. - done) * GAMMA * target_q_values

        qf1_loss = F.mse_loss(q1_pred, q_target.detach())
        qf2_loss = F.mse_loss(q2_pred, q_target.detach())

        batch_size = action.shape[0]
        action_dim = action.shape[-1]
        cql_random_actions = action.new_empty((batch_size, 10, action_dim), requires_grad=False).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.policy(state, repeat=10)
        cql_next_actions, cql_next_log_pis = self.policy(next_state, repeat=10)
        cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
        cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

        repeated_state = state.unsqueeze(1)
        repeated_state = torch.repeat_interleave(repeated_state, 10, 1)
        cql_q1_rand = self.qf1(repeated_state, cql_random_actions)
        cql_q2_rand = self.qf2(repeated_state, cql_random_actions)
        cql_q1_current_actions = self.qf1(repeated_state, cql_current_actions)
        cql_q2_current_actions = self.qf2(repeated_state, cql_current_actions)
        cql_q1_next_actions = self.qf1(repeated_state, cql_next_actions)
        cql_q2_next_actions = self.qf2(repeated_state, cql_next_actions)

        random_density = np.log(0.5 ** action_dim)
        cql_cat_q1 = torch.cat(
            [cql_q1_rand.squeeze(2) - random_density,
             cql_q1_next_actions.squeeze(2) - cql_next_log_pis.detach(),
             cql_q1_current_actions.squeeze(2) - cql_current_log_pis.detach()],
            dim=1
        )
        cql_cat_q2 = torch.cat(
            [cql_q2_rand.squeeze(2) - random_density,
             cql_q2_next_actions.squeeze(2) - cql_next_log_pis.detach(),
             cql_q2_current_actions.squeeze(2) - cql_current_log_pis.detach()],
            dim=1
        )

        cql_min_qf1_loss = torch.logsumexp(cql_cat_q1, dim=1).mean() * 5.0
        cql_min_qf2_loss = torch.logsumexp(cql_cat_q2, dim=1).mean() * 5.0

        """Subtract the log likelihood of data"""
        cql_min_qf1_loss = cql_min_qf1_loss - q1_pred.mean() * 5.0
        cql_min_qf2_loss = cql_min_qf2_loss - q2_pred.mean() * 5.0

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        soft_update(self.target_qf1, self.qf1)
        soft_update(self.target_qf2, self.qf2)

    def save(self):
        torch.save(self.policy.state_dict(), "agent.pkl")


def train():
    agent = Agent(19, 5)
    optimal = np.load("optimal.npz", allow_pickle=True)["arr_0"]
    dataset = TransitionDataset(optimal)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for _ in tqdm.tqdm(range(5000)):
        for batch in dataloader:
            agent.update(batch)

    return model


if __name__ == "__main__":
    model = train()
    model.save()
