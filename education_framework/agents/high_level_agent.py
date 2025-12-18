# agents/high_level_agents.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# StateType = Tuple[Hashable, ...]
def _to_f32_batch(x) -> np.ndarray:
    """
    Converts x (list of np arrays / list of lists / np array) to a contiguous float32 ndarray.
    Handles the slow 'list of numpy arrays' case via np.stack.
    """
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.stack(x, axis=0)
    return np.ascontiguousarray(arr, dtype=np.float32)

def _to_i64_batch(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.stack(x, axis=0)
    return np.ascontiguousarray(arr, dtype=np.int64)

@dataclass
class HighLevelAgentConfig:
    num_topics: int
    use_tutee: bool = True

    # DQN hyperparameters
    gamma: float = 0.95
    lr: float = 1e-3
    epsilon: float = 0.1

    buffer_size: int = 50_000
    batch_size: int = 1024
    min_replay_size: int = 1_000

    target_update_steps: int = 1_000
    train_every_steps: int = 100

    # for numerical stability
    max_grad_norm: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # change to "cuda" if you want


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return s, a, r, s2, d


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HighLevelAgent:
    """
    High-level DQN agent.

    Action space:
      - tutor_topic_i for i in [0..num_topics-1]
      - tutee_topic_i for i in [0..num_topics-1] if use_tutee=True
    """

    def __init__(self, config: HighLevelAgentConfig):
        self.cfg = config
        self.actions: List[str] = []
        self._build_action_space()

        self.device = torch.device(self.cfg.device)
        self.replay = ReplayBuffer(self.cfg.buffer_size)

        self.policy_net: Optional[QNetwork] = None
        self.target_net: Optional[QNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None

        self.total_steps = 0

    # ----------------- public API -----------------

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def get_action_meanings(self) -> List[str]:
        return self.actions

    def set_epsilon(self, epsilon: float) -> None:
        self.cfg.epsilon = max(0.0, float(epsilon))

    def decode_action(self, action: int) -> Tuple[str, int]:
        meaning = self.actions[action]
        mode_str, _, topic_str = meaning.partition("_topic_")
        topic_id = int(topic_str)
        mode = "tutor" if mode_str == "tutor" else "tutee"
        return mode, topic_id

    def select_action(self, obs: List[float]) -> int:
        self._ensure_networks(input_dim=len(obs))

        if random.random() < self.cfg.epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            obs_np = np.asarray(obs, dtype=np.float32)
            x = torch.from_numpy(obs_np).unsqueeze(0)  # CPU
            q = self.policy_net(x)
            return int(q.argmax(dim=1).item())

    def update(self, obs: List[float], action: int, reward: float, next_obs: List[float], done: bool) -> None:
        self._ensure_networks(input_dim=len(obs))

        # obs/next_obs are freshly created lists from env; do not copy (major speed win)
        self.replay.push(
            obs,
            int(action),
            float(reward),
            next_obs,
            bool(done),
        )

        self.total_steps += 1

        # only train every N steps
        if self.total_steps % self.cfg.train_every_steps != 0:
            return

        # wait until replay has enough diversity
        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return

        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)


        s_np = _to_f32_batch(s)
        s2_np = _to_f32_batch(s2)
        a_np = _to_i64_batch(a)
        r_np = np.ascontiguousarray(np.asarray(r, dtype=np.float32))
        d_np = np.ascontiguousarray(np.asarray(d, dtype=np.float32))

        s_t = torch.from_numpy(s_np)
        s2_t = torch.from_numpy(s2_np)
        a_t = torch.from_numpy(a_np).unsqueeze(1)
        r_t = torch.from_numpy(r_np)
        d_t = torch.from_numpy(d_np)

        # Q(s,a)
        q_sa = self.policy_net(s_t).gather(1, a_t).squeeze(1)

        # Double DQN:
        # a* = argmax_a Q_policy(s', a)
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(s2_t), dim=1, keepdim=True)
            next_q = self.target_net(s2_t).gather(1, next_actions).squeeze(1)
            target = r_t + self.cfg.gamma * (1.0 - d_t) * next_q

        loss = nn.functional.mse_loss(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        # target network update
        if self.total_steps % self.cfg.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ----------------- internal helpers -----------------

    def _build_action_space(self) -> None:
        self.actions: List[str] = []

        # tutor actions for each topic
        for t in range(self.cfg.num_topics):
            self.actions.append(f"tutor_topic_{t}")

        # tutee actions for each topic (optional)
        # TODO check if it's needed
        if self.cfg.use_tutee:
            for t in range(self.cfg.num_topics):
                self.actions.append(f"tutee_topic_{t}")

    def _ensure_networks(self, input_dim: int) -> None:
        if self.policy_net is not None:
            return

        self.policy_net = QNetwork(input_dim=input_dim, num_actions=self.num_actions).to(self.device)
        self.target_net = QNetwork(input_dim=input_dim, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
