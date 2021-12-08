import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

EPOCHES = 200
LR = 1e-3
HIDDEN_DIM = 256
BATCH_SIZE = 4
DEVICE = "cuda"
GRAD_NORM = 2.0


def collect_data():
    transictions = []
    optimal = np.load("suboptimal.npz", allow_pickle=True)["arr_0"]
    suboptimal = np.load("optimal.npz", allow_pickle=True)["arr_0"]

    for dataset in [optimal, suboptimal]:
        traj = []
        for state, action, next_state, reward, done in dataset:
            if len(traj) > 0 and (state != traj[-1][2]).any():
                transictions.append(traj)
                traj = []
            traj.append((state, action, next_state, reward, done))
            if done:
                transictions.append(traj)
                traj = []
        if len(traj) > 0:
            transictions.append(traj)
    return transictions


class TransitionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, next_state, reward, done = list(zip(*self.data[idx]))
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        reward = np.array(reward)
        done = np.array(done)

        return {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }


class TransitionCollate:
    def __call__(self, batch):
        batch_size = len(batch)
        lengths = [el["state"].shape[0] for el in batch]
        max_len = max(lengths)
        states = torch.zeros(
            batch_size, max_len, batch[0]["state"].shape[1], dtype=torch.float32
        )
        actions = torch.zeros(
            batch_size, max_len, batch[0]["action"].shape[1], dtype=torch.float32
        )
        next_states = torch.zeros(
            batch_size, max_len, batch[0]["next_state"].shape[1], dtype=torch.float32
        )
        rewards = torch.zeros(batch_size, max_len, dtype=torch.float32)
        dones = torch.zeros(batch_size, max_len, dtype=torch.float32)

        for i, element in enumerate(batch):
            state = element["state"]
            action = element["action"]
            next_state = element["next_state"]
            reward = element["reward"]
            done = element["done"]
            states[i, : state.shape[0], :] = torch.from_numpy(state)
            actions[i, : action.shape[0], :] = torch.from_numpy(action)
            next_states[i, : next_state.shape[0], :] = torch.from_numpy(next_state)
            rewards[i, : reward.shape[0]] = torch.from_numpy(reward)
            dones[i, : done.shape[0]] = torch.from_numpy(done)

        lengths = torch.LongTensor(lengths)

        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "dones": dones,
            "lengths": lengths,
        }


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(WorldModel, self).__init__()
        self.rnn = nn.LSTM(
            state_dim + action_dim, hidden_size, bidirectional=False, batch_first=True
        )
        self.state_head = nn.Linear(hidden_size, state_dim)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, states, actions, length):
        x = torch.cat((states, actions), dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, length, batch_first=True, enforce_sorted=False
        )
        out_packed, (_, _) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        next_state = self.state_head(out)
        reward = self.reward_head(out)
        done = self.done_head(out)
        return next_state, reward, done


if __name__ == "__main__":
    transitions = collect_data()
    initial_state = [traj[0][0] for traj in transitions]
    np.save("initial_state.npy", np.array(initial_state))
    dataset = TransitionDataset(transitions)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=TransitionCollate(), shuffle=True
    )
    model = WorldModel(19, 5, HIDDEN_DIM).to(DEVICE)
    optimizer = Adam(model.parameters(), LR)
    done_criteria = nn.BCELoss()
    for e in range(EPOCHES):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            states = batch["states"].to(DEVICE)
            actions = batch["actions"].to(DEVICE)
            next_states = batch["next_states"].to(DEVICE)
            rewards = batch["rewards"].to(DEVICE)
            dones = batch["dones"].to(DEVICE)
            lengths = batch["lengths"]
            next_state_pred, reward_pred, done_pred = model.forward(
                states, actions, lengths
            )
            loss_state = F.mse_loss(
                next_state_pred.view(-1, next_states.shape[-1]),
                next_states.view(-1, next_states.shape[-1]),
            )
            loss_reward = F.mse_loss(reward_pred.squeeze(2).view(-1), rewards.view(-1))
            loss_done = done_criteria(done_pred.squeeze(2).view(-1), dones.view(-1))
            loss = loss_state + loss_reward + loss_done
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step()

            if not (i + 1) % 100:
                print(
                    f"epoch: {e + 1}/{EPOCHES}, "
                    f"iter: {i + 1}/{len(dataloader)}, "
                    f"loss state: {loss_state.item():.3f} "
                    f"loss reward: {loss_reward.item():.3f} "
                    f"loss done: {loss_done.item():.3f} "
                )
    torch.save(model.state_dict(), "world.pkl")
