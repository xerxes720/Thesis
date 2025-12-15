
# agents/low_level_agents.py  (ASYNC + STABLE SETTINGS)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from agents.torch_replay import TorchReplayBuffer


@dataclass
class LowLevelAgentConfig:
    num_topics: int

    gamma: float = 0.95
    lr: float = 3e-4
    epsilon: float = 0.1

    buffer_size: int = 300_000
    batch_size: int = 8192
    min_replay_size: int = 50_000

    target_update_steps: int = 10_000
    train_every_steps: int = 1
    grad_steps_per_update: int = 2

    max_grad_norm: float = 10.0
    hidden_dim: int = 1024

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Keep off here for performance; can re-add later torch-native.
    experience_sharing: bool = False
    share_mode: str = "off"


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int):
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
    def __init__(self, config: LowLevelAgentConfig, actions: List[str]):
        self.cfg = config
        self.actions = actions
        self.device = torch.device(self.cfg.device)

        self.total_steps = 0
        self.policy_net: Optional[QNetwork] = None
        self.target_net: Optional[QNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
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
        self.policy_net = QNetwork(input_dim, self.num_actions, self.cfg.hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim, self.num_actions, self.cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.replay = TorchReplayBuffer(self.cfg.buffer_size, obs_dim=input_dim, pin_memory=True)

        # IMPORTANT: do NOT call torch.compile on Windows (Triton missing)

    @torch.no_grad()
    def select_action_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        self._ensure_networks(int(obs_batch.shape[1]))
        B = obs_batch.shape[0]

        q = self.policy_net(obs_batch)
        greedy = torch.argmax(q, dim=1)

        if self.cfg.epsilon <= 0.0:
            return greedy

        rand_mask = (torch.rand((B,), device=obs_batch.device) < self.cfg.epsilon)
        if rand_mask.any():
            random_actions = torch.randint(0, self.num_actions, (B,), device=obs_batch.device)
            return torch.where(rand_mask, random_actions, greedy)
        return greedy

    def update_batch(
        self,
        obs_cpu: torch.Tensor,
        actions_cpu: torch.Tensor,
        rewards_cpu: torch.Tensor,
        next_obs_cpu: torch.Tensor,
        dones_cpu: torch.Tensor,
    ) -> None:
        self._ensure_networks(int(obs_cpu.shape[1]))
        assert self.replay is not None and self.policy_net is not None and self.target_net is not None and self.optimizer is not None

        self.replay.push_batch(obs_cpu, actions_cpu, rewards_cpu, next_obs_cpu, dones_cpu)
        self.total_steps += int(obs_cpu.shape[0])

        if (self.total_steps % self.cfg.train_every_steps) != 0:
            return
        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return

        for _ in range(int(self.cfg.grad_steps_per_update)):
            s, a, r, s2, d = self.replay.sample(self.cfg.batch_size, device=self.device)

            q_sa = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_a = torch.argmax(self.policy_net(s2), dim=1, keepdim=True)
                next_q = self.target_net(s2).gather(1, next_a).squeeze(1)
                target = r + self.cfg.gamma * (1.0 - d) * next_q

            loss = nn.functional.smooth_l1_loss(q_sa, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

        if (self.total_steps % self.cfg.target_update_steps) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# Action sets (must match learner_model expectations)
def build_tutor_actions() -> List[str]:
    return ["hint", "worked_example", "reflection_question", "no_help"]


def build_tutee_actions() -> List[str]:
    return ["ask_explanation", "ask_worked_example", "ask_summary", "show_mistake_and_ask_fix"]


class TutorLowLevelAgent(DQNLowLevelAgent):
    def __init__(self, config: LowLevelAgentConfig):
        super().__init__(config, actions=build_tutor_actions())


class TuteeLowLevelAgent(DQNLowLevelAgent):
    def __init__(self, config: LowLevelAgentConfig):
        super().__init__(config, actions=build_tutee_actions())
