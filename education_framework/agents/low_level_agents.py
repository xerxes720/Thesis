# agents/low_level_agents.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


def _to_f32_batch(x) -> np.ndarray:
    """
    Converts x (list of np arrays / list of lists / np array) to a contiguous float32 ndarray.
    Handles the slow 'list of numpy arrays' case via np.stack.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.dtype == object:
        arr = np.stack(x, axis=0)
    return np.ascontiguousarray(arr)

def _to_i64_batch(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64)
    if arr.dtype == object:
        arr = np.stack(x, axis=0)
    return np.ascontiguousarray(arr)
@dataclass
class LowLevelAgentConfig:
    num_topics: int = 7

    # DQN hyperparameters
    gamma: float = 0.95
    lr: float = 1e-3
    epsilon: float = 0.2

    buffer_size: int = 50_000
    batch_size: int = 256
    min_replay_size: int = 150

    train_every_steps: int = 20
    # target_update_steps: int = 1_000
    #To be faithful to the original paper: k=5
    target_update_steps: int = 5 * train_every_steps
    experience_sharing: bool = True
    # experience_sharing: bool = False
    share_mode: str = "weighted_cka"  # options: "off", "mutual", "weighted_cka"

    max_grad_norm: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- tutee readiness thresholds (copied from KDDLearnerConfig when building tutee agent) ---
    tutee_ready_quiz: float = 0.50
    tutee_ready_explain: float = 0.60
    tutee_ready_fix: float = 0.65

    # optional: penalty magnitude for "not ready"
    tutee_not_ready_penalty: float = 0.5

    # share_frac: float = 0.2          # 0.3–0.7 works; start 0.5
    share_warmup_updates: int = 300   # same idea as you already use :contentReference[oaicite:2]{index=2}
    # shared_buffer_size: int = 200_000 # independent from per-agent buffer

    cka_layers: Tuple[str, ...] = ("h1", "h2")
    # cka_probe_n: int = 64
    cka_power: float = 1.0 # keep simple; optional
    # shared_loss_weight: float = 1.0  # base weight multiplier for shared samples


# ---------------- Replay Buffer ----------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.np_rng = np.random.default_rng(0)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        n = len(self)
        idx = self.np_rng.integers(0, n, size=batch_size)  # with replacement (DQN standard)
        batch = [self.buffer[i] for i in idx]  # still Python list, but avoids random.sample overhead
        s, a, r, s2, d = zip(*batch)
        return list(s), list(a), list(r), list(s2), list(d)


class QNetwork(nn.Module):
    """
    MLP 64-64, with optional access to intermediate hidden representations for CKA.
    """

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.out(h2)

    def forward_with_reps(self, x: torch.Tensor) -> dict:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        q = self.out(h2)
        return {"h1": h1, "h2": h2, "q": q}


# ---------------- Similarity (Linear CKA) ----------------

def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Linear CKA similarity between representations X and Y.
    X: [n, d1], Y: [n, d2]
    Uses centered features (mean-subtracted) which is common in practice.
    Returns scalar tensor in [0, 1] (approximately).
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # numerator = || X^T Y ||_F^2
    XT_Y = X.T @ Y
    num = (XT_Y ** 2).sum()

    # denom = sqrt(||X^T X||_F^2 * ||Y^T Y||_F^2)
    XT_X = X.T @ X
    YT_Y = Y.T @ Y
    denom = torch.sqrt(((XT_X ** 2).sum() * (YT_Y ** 2).sum()).clamp_min(eps))

    return (num / denom).clamp(0.0, 1.0)


def avg_layer_cka(
        net_a: QNetwork,
        net_b: QNetwork,
        states: torch.Tensor,
        layers: Sequence[str],
) -> float:
    """
    Compute average Linear CKA across specified layers using the same input states.
    """
    with torch.no_grad():
        reps_a = net_a.forward_with_reps(states)
        reps_b = net_b.forward_with_reps(states)

        vals = []
        for layer in layers:
            if layer not in reps_a or layer not in reps_b:
                continue
            cka = linear_cka(reps_a[layer], reps_b[layer]).item()
            vals.append(cka)

        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))


class DQNLowLevelAgent:
    """
    Generic DQN low-level agent with optional experience sharing.

    - Each agent has its own replay buffer.
    - During training, it can also sample from peer buffers.
    - Peer samples are weighted (mutual or similarity-weighted via CKA).
    """

    def __init__(self, config: LowLevelAgentConfig, actions: List[str]):
        self.cfg = config
        self.actions = actions
        self.device = torch.device(self.cfg.device)
        self.num_updates = 0

        self.replay = ReplayBuffer(self.cfg.buffer_size)
        self.policy_net: Optional[QNetwork] = None
        self.target_net: Optional[QNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None

        self.total_steps = 0
        self._peers: List["DQNLowLevelAgent"] = []
        # ---- sharing diagnostics (per-episode counters; reset by main) ----
        self._share_attempts = 0  # number of times _build_shared_batch() ran with sharing active
        self._share_peer_samples = 0  # total peer transitions appended
        self._share_peer_weight_sum = 0.0  # sum(weight * num_peer_samples) across all peers/updates
        self._share_eligible_peers_sum = 0  # sum(#eligible_peers) across sharing attempts
        self._share_selected_peers_sum = 0  # sum(#selected_peers) across sharing attempts
        # self.shared_replay: Optional[ReplayBuffer] = None
        # self.share_ref_net: Optional[QNetwork] = None

    # def set_share_ref_net(self, net: Optional[QNetwork]) -> None:
    #     self.share_ref_net = net
    # def set_shared_replay(self, shared: Optional[ReplayBuffer]) -> None:
    #     self.shared_replay = shared
    def reset_share_stats(self) -> None:
        """Reset per-episode sharing counters."""
        self._share_attempts = 0
        self._share_peer_samples = 0
        self._share_peer_weight_sum = 0.0
        self._share_eligible_peers_sum = 0
        self._share_selected_peers_sum = 0

    def pop_share_stats(self) -> dict:
        """Return current counters and reset them."""
        stats = {
            "share_attempts": int(self._share_attempts),
            "peer_samples": int(self._share_peer_samples),
            "peer_weight_sum": float(self._share_peer_weight_sum),
            "eligible_peers_sum": int(self._share_eligible_peers_sum),
            "selected_peers_sum": int(self._share_selected_peers_sum),
        }
        self.reset_share_stats()
        return stats
    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def get_action_meanings(self) -> List[str]:
        return self.actions

    def set_epsilon(self, epsilon: float) -> None:
        self.cfg.epsilon = max(0.0, float(epsilon))

    def set_peers(self, peers: List["DQNLowLevelAgent"]) -> None:
        """
        Set peer agents used for experience sharing. Typically called once after creation.
        """
        # avoid self references
        self._peers = [p for p in peers if p is not self]

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

        # store own transition
        # obs/next_obs are freshly created lists from env; do not copy (major speed win)
        # self.replay.push(obs, int(action), float(reward), next_obs, bool(done))
        self.replay.push(obs, action, reward, next_obs, done)

        # if self.shared_replay is not None:
        #     self.shared_replay.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

        if self.total_steps % self.cfg.train_every_steps != 0:
            return

        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return

        # -------- Build training batch: own + shared --------
        batch_s, batch_a, batch_r, batch_s2, batch_d, batch_w = self._build_shared_batch(self.cfg.batch_size)

        s_np = _to_f32_batch(batch_s)
        s2_np = _to_f32_batch(batch_s2)
        a_np = _to_i64_batch(batch_a)
        r_np = np.ascontiguousarray(np.asarray(batch_r, dtype=np.float32))
        d_np = np.ascontiguousarray(np.asarray(batch_d, dtype=np.float32))
        w_np = np.ascontiguousarray(np.asarray(batch_w, dtype=np.float32))

        s_t = torch.from_numpy(s_np)
        s2_t = torch.from_numpy(s2_np)
        a_t = torch.from_numpy(a_np).unsqueeze(1)
        r_t = torch.from_numpy(r_np)
        d_t = torch.from_numpy(d_np)
        w_t = torch.from_numpy(w_np)

        s_t = s_t.to(self.device)
        s2_t = s2_t.to(self.device)
        a_t = a_t.to(self.device)
        r_t = r_t.to(self.device)
        d_t = d_t.to(self.device)
        w_t = w_t.to(self.device)

        # Q(s,a)
        q_sa = self.policy_net(s_t).gather(1, a_t).squeeze(1)

        # Double DQN target:
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(s2_t), dim=1, keepdim=True)
            next_q = self.target_net(s2_t).gather(1, next_actions).squeeze(1)
            target = r_t + self.cfg.gamma * (1.0 - d_t) * next_q

        # w_t = w_t.clamp(0.0, 2.0)  # or 1.5 if you want strict
        # w_t = w_t / w_t.mean().clamp_min(1e-8)

        # Weighted TD loss
        td = (q_sa - target)
        loss = (w_t * (td ** 2)).sum() / (w_t.sum().clamp_min(1e-8))
        # loss_per = torch.nn.functional.smooth_l1_loss(q_sa, target, reduction="none")  # Huber
        # loss = (w_t * loss_per).sum() / w_t.sum().clamp_min(1e-8)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.num_updates += 1

        # target net update
        if self.num_updates % max(1, int(self.cfg.target_update_steps)) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # -------- internals --------

    def _build_shared_batch(self, B: int):
        mode = str(getattr(self.cfg, "share_mode", "off"))

        # off => only own samples
        if (mode == "off") or (not getattr(self.cfg, "experience_sharing", False)):
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * B
            return s, a, r, s2, d, w

        # warmup => only own samples
        if int(self.num_updates) < int(getattr(self.cfg, "share_warmup_updates", 0)):
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * B
            return s, a, r, s2, d, w

        peers = list(getattr(self, "_peers", []) or [])
        # no peers => fall back to own (do NOT count as a sharing attempt)
        if len(peers) == 0:
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * B
            return s, a, r, s2, d, w

        # Own batch
        s, a, r, s2, d = self.replay.sample(B)
        w = [1.0] * B

        # If nets not ready, do not share (do NOT count attempt)
        if (self.policy_net is None) or (mode not in ("mutual", "weighted_cka")):
            return s, a, r, s2, d, w

        # Now sharing is truly attempted
        self._share_attempts += 1
        self._share_eligible_peers_sum += len(peers)

        # CKA settings (paper: average across layers)
        cka_layers = tuple(getattr(self.cfg, "cka_layers", ("h1", "h2")))
        power = float(getattr(self.cfg, "cka_power", 1.0))  # paper doesn't emphasize exponent; keep 1.0 for faithful

        for p in peers:
            if (p.policy_net is None) or (len(p.replay) < B):
                continue

            # Paper: sample Bj from peer j, and compute similarity using Bj as probe
            ps, pa, pr, ps2, pd = p.replay.sample(B)

            if mode == "mutual":
                sim_w = 1.0
            else:
                # weighted_cka
                q_np = _to_f32_batch(ps)  # peer states are the probe
                q_t = torch.from_numpy(q_np).to(self.device)

                sim = float(avg_layer_cka(self.policy_net, p.policy_net, q_t, cka_layers))
                sim = max(0.0, min(1.0, sim))
                sim_w = sim ** power

            self._share_selected_peers_sum += 1
            self._share_peer_samples += B
            self._share_peer_weight_sum += float(sim_w) * B
            s.extend(ps);
            a.extend(pa);
            r.extend(pr);
            s2.extend(ps2);
            d.extend(pd)
            w.extend([sim_w] * B)

        return s, a, r, s2, d, w

    def _ensure_networks(self, input_dim: int) -> None:
        if self.policy_net is not None:
            return

        self.policy_net = QNetwork(input_dim=input_dim, num_actions=self.num_actions).to(self.device)
        self.target_net = QNetwork(input_dim=input_dim, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)


# ---------- TUTOR low-level agents ----------

def build_tutor_actions() -> List[str]:
    """
    Discrete assistance types given by the Tutor to the learner.
    """
    return [
        "quiz",
        "hint",
        "worked_example",
        "remediation",
        "review",
    ]


class TutorLowLevelAgent(DQNLowLevelAgent):
    def __init__(self, config: LowLevelAgentConfig):
        super().__init__(config, actions=build_tutor_actions())


# ---------- TUTEE low-level agent ----------
class TuteeLowLevelAgent(DQNLowLevelAgent):
    def __init__(self, config: LowLevelAgentConfig):
        cfg = deepcopy(config)
        cfg.experience_sharing = False
        cfg.share_mode = "off"
        super().__init__(cfg, actions=build_tutee_actions())

    def select_action(self, obs: List[float]) -> int:
        self._ensure_networks(input_dim=len(obs))  # should be 9

        if random.random() < self.cfg.epsilon:
            return random.randrange(self.num_actions)

        # obs = [mastery_k, cfa_k, hint_k, time_k, inc_k, opp_k, complete_k, global_mastery, steps_norm]
        m = float(obs[0])

        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(x).squeeze(0).detach().cpu().numpy()

        idx_quiz, idx_explain, idx_fix = 0, 1, 2
        pen = float(getattr(self.cfg, "tutee_not_ready_penalty", 0.5))

        if m < self.cfg.tutee_ready_quiz:
            q[idx_quiz] -= pen
        if m < self.cfg.tutee_ready_explain:
            q[idx_explain] -= pen
        if m < self.cfg.tutee_ready_fix:
            q[idx_fix] -= pen

        return int(q.argmax())




def build_tutee_actions() -> List[str]:
    # align 1:1 with learner_model LowLevelAction semantics
    return [
        "tutee_quiz",
        "tutee_explain",
        "tutee_fix",
    ]

