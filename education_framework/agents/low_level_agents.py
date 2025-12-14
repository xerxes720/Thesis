
# agents/low_level_agents.py  (BATCHED VERSION)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch_replay import TorchReplayBuffer


@dataclass
class LowLevelAgentConfig:
    num_topics: int

    gamma: float = 0.95
    lr: float = 1e-3
    epsilon: float = 0.1

    buffer_size: int = 200_000
    batch_size: int = 2048
    min_replay_size: int = 10_000

    target_update_steps: int = 5_000
    train_every_steps: int = 1

    grad_steps_per_update: int = 2

    max_grad_norm: float = 10.0
    hidden_dim: int = 256

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # For maximum GPU throughput, keep experience sharing off in this batched version.
    # (If you need sharing, we can re-add it using a torch-native shared batch builder.)
    experience_sharing: bool = False
    share_mode: str = "off"


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNLowLevelAgent:
    """
    DQN low-level agent with batched select/update and torch replay.
    """

    def __init__(self, config: LowLevelAgentConfig, actions: List[str]):
        self.cfg = config
        self.actions = actions
        self.device = torch.device(self.cfg.device)

        self.total_steps = 0
        self.policy_net: Optional[QNetwork] = None
        self.target_net: Optional[QNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None

        self.obs_dim: Optional[int] = None
        self.replay: Optional[TorchReplayBuffer] = None

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def get_action_meanings(self) -> List[str]:
        return self.actions

    def set_epsilon(self, epsilon: float) -> None:
        self.cfg.epsilon = max(0.0, float(epsilon))

    def _ensure_networks(self, input_dim: int) -> None:
        if self.policy_net is not None:
            return
        self.obs_dim = int(input_dim)
        self.policy_net = QNetwork(input_dim=input_dim, num_actions=self.num_actions, hidden_dim=self.cfg.hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim=input_dim, num_actions=self.num_actions, hidden_dim=self.cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.replay = TorchReplayBuffer(self.cfg.buffer_size, obs_dim=input_dim, pin_memory=True)

        if hasattr(torch, "compile"):
            try:
                self.policy_net = torch.compile(self.policy_net)  # type: ignore
                self.target_net = torch.compile(self.target_net)  # type: ignore
            except Exception:
                pass

    @torch.no_grad()
    def select_action_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        obs_batch: float32 [B, obs_dim] on self.device
        Returns: int64 [B] on self.device
        """
        self._ensure_networks(input_dim=obs_batch.shape[1])
        B = obs_batch.shape[0]

        if self.cfg.epsilon > 0.0:
            rand_mask = (torch.rand((B,), device=obs_batch.device) < self.cfg.epsilon)
        else:
            rand_mask = torch.zeros((B,), dtype=torch.bool, device=obs_batch.device)

        q = self.policy_net(obs_batch)
        greedy = torch.argmax(q, dim=1)

        if rand_mask.any():
            random_actions = torch.randint(0, self.num_actions, (B,), device=obs_batch.device)
            return torch.where(rand_mask, random_actions, greedy)
        return greedy

    def update_batch(
        self,
        obs: torch.Tensor,        # CPU float32 [B, obs_dim]
        actions: torch.Tensor,    # CPU int64 [B]
        rewards: torch.Tensor,    # CPU float32 [B]
        next_obs: torch.Tensor,   # CPU float32 [B, obs_dim]
        dones: torch.Tensor,      # CPU float32 [B]
    ) -> None:
        self._ensure_networks(input_dim=obs.shape[1])
        assert self.replay is not None and self.policy_net is not None and self.target_net is not None and self.optimizer is not None

        self.replay.push_batch(obs, actions, rewards, next_obs, dones)
        self.total_steps += int(obs.shape[0])

        if (self.total_steps % self.cfg.train_every_steps) != 0:
            return
        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return

        for _ in range(int(self.cfg.grad_steps_per_update)):
            s_t, a_t, r_t, s2_t, d_t = self.replay.sample(self.cfg.batch_size, device=self.device)

            q_sa = self.policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions = torch.argmax(self.policy_net(s2_t), dim=1, keepdim=True)
                next_q = self.target_net(s2_t).gather(1, next_actions).squeeze(1)
                target = r_t + self.cfg.gamma * (1.0 - d_t) * next_q

            loss = nn.functional.mse_loss(q_sa, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

        if (self.total_steps % self.cfg.target_update_steps) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# ---------- Actions ----------

def build_tutor_actions() -> List[str]:
    return [
        "hint",
        "worked_example",
        "reflection_question",
        "no_help",
    ]


class TutorLowLevelAgent(DQNLowLevelAgent):
    def __init__(self, config: LowLevelAgentConfig):
        super().__init__(config, actions=build_tutor_actions())


def build_tutee_actions() -> List[str]:
    return [
        "ask_explanation",
        "ask_worked_example",
        "ask_summary",
        "show_mistake_and_ask_fix",
    ]


class TuteeLowLevelAgent(DQNLowLevelAgent):
    def __init__(self, config: LowLevelAgentConfig):
        super().__init__(config, actions=build_tutee_actions())
