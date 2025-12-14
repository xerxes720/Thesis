
# agents/high_level_agent.py  (BATCHED VERSION)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.optim as optim

import torch_replay
from torch_replay import TorchReplayBuffer

@dataclass
class HighLevelAgentConfig:
    num_topics: int
    use_tutee: bool = True

    gamma: float = 0.95
    lr: float = 1e-3
    epsilon: float = 0.1

    buffer_size: int = 200_000
    batch_size: int = 4096
    min_replay_size: int = 10_000

    target_update_steps: int = 5_000
    train_every_steps: int = 1

    # Do more than 1 gradient step per env step to keep GPU busy
    grad_steps_per_update: int = 2

    max_grad_norm: float = 10.0
    hidden_dim: int = 256

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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


class HighLevelAgent:
    """
    High-level DQN agent, optimized for batched training/inference.
    Action encoding:
      - 0..T-1      -> tutor_topic_t
      - T..2T-1     -> tutee_topic_(a-T) (if use_tutee)
    """

    def __init__(self, config: HighLevelAgentConfig):
        self.cfg = config
        self.device = torch.device(self.cfg.device)

        self.total_steps = 0
        self.policy_net: Optional[QNetwork] = None
        self.target_net: Optional[QNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None

        self.obs_dim: Optional[int] = None
        self.replay: Optional[TorchReplayBuffer] = None

    @property
    def num_actions(self) -> int:
        return self.cfg.num_topics * (2 if self.cfg.use_tutee else 1)

    def set_epsilon(self, epsilon: float) -> None:
        self.cfg.epsilon = max(0.0, float(epsilon))

    def decode_actions_batch(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        actions: int64 [B] (device doesn't matter)
        Returns:
          modes: bool [B]  (False=tutor, True=tutee)
          topics: int64 [B]
        """
        T = self.cfg.num_topics
        topics = actions % T
        modes = actions >= T
        return modes, topics

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

        # Optional compiler (works best once shapes are stable)
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
        # epsilon-greedy mask
        if self.cfg.epsilon > 0.0:
            rand_mask = (torch.rand((B,), device=obs_batch.device) < self.cfg.epsilon)
        else:
            rand_mask = torch.zeros((B,), dtype=torch.bool, device=obs_batch.device)

        # greedy
        q = self.policy_net(obs_batch)  # [B, A]
        greedy = torch.argmax(q, dim=1)

        if rand_mask.any():
            random_actions = torch.randint(0, self.num_actions, (B,), device=obs_batch.device)
            actions = torch.where(rand_mask, random_actions, greedy)
            return actions
        return greedy

    def update_batch(
        self,
        obs: torch.Tensor,        # CPU float32 [B, obs_dim]
        actions: torch.Tensor,    # CPU int64   [B]
        rewards: torch.Tensor,    # CPU float32 [B]
        next_obs: torch.Tensor,   # CPU float32 [B, obs_dim]
        dones: torch.Tensor,      # CPU float32 [B] (0/1)
    ) -> None:
        self._ensure_networks(input_dim=obs.shape[1])
        assert self.replay is not None and self.policy_net is not None and self.target_net is not None and self.optimizer is not None

        # store
        self.replay.push_batch(obs, actions, rewards, next_obs, dones)

        self.total_steps += int(obs.shape[0])
        if (self.total_steps % self.cfg.train_every_steps) != 0:
            return
        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return

        for _ in range(int(self.cfg.grad_steps_per_update)):
            s_t, a_t, r_t, s2_t, d_t = self.replay.sample(self.cfg.batch_size, device=self.device)

            # Q(s,a)
            q_sa = self.policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # Double DQN
                next_actions = torch.argmax(self.policy_net(s2_t), dim=1, keepdim=True)
                next_q = self.target_net(s2_t).gather(1, next_actions).squeeze(1)
                target = r_t + self.cfg.gamma * (1.0 - d_t) * next_q

            loss = nn.functional.mse_loss(q_sa, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

        if (self.total_steps % self.cfg.target_update_steps) == 0:
            # For compiled models, load_state_dict still works.
            self.target_net.load_state_dict(self.policy_net.state_dict())
