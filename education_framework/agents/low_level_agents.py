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

    max_grad_norm: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Experience sharing ---
    experience_sharing: bool = True
    # experience_sharing: bool = False
    share_mode: str = "weighted_cka"  # options: "off", "mutual", "weighted_cka"

    # NEW: fixed-size batch mixing
    share_frac: float = 0.10  # fraction of each batch coming from peers
    share_warmup_updates: int = 100  # don't share until this many gradient updates

    max_peers_per_update: int = 2  # sample up to this many peers each train step (for speed)
    peer_batch_size: int = 32  # how many transitions to sample from each peer
    min_peer_replay_size: int = 500  # peers must have at least this many samples to participate

    share_weight_floor: float = 0.05  # clamp similarity weights
    share_weight_ceiling: float = 1.5
    cka_layers: Tuple[str, ...] = ("h1", "h2")  # which layers to use for similarity

    # NEW: normalize peer weights per update (makes CKA differences matter more)
    normalize_peer_weights: bool = True

    # --- tutee readiness thresholds (copied from KDDLearnerConfig when building tutee agent) ---
    tutee_ready_quiz: float = 0.50
    tutee_ready_explain: float = 0.60
    tutee_ready_fix: float = 0.65

    # optional: penalty magnitude for "not ready"
    tutee_not_ready_penalty: float = 0.5



# ---------------- Replay Buffer ----------------

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
        self.replay.push(obs, int(action), float(reward), next_obs, bool(done))
        self.total_steps += 1

        if self.total_steps % self.cfg.train_every_steps != 0:
            return

        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return

        # -------- Build training batch: own + shared --------
        batch_s, batch_a, batch_r, batch_s2, batch_d, batch_w = self._build_shared_batch()


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

        # Q(s,a)
        q_sa = self.policy_net(s_t).gather(1, a_t).squeeze(1)

        # Double DQN target:
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(s2_t), dim=1, keepdim=True)
            next_q = self.target_net(s2_t).gather(1, next_actions).squeeze(1)
            target = r_t + self.cfg.gamma * (1.0 - d_t) * next_q

        # Weighted TD loss
        td = (q_sa - target)
        loss = (w_t * (td ** 2)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.num_updates += 1

        # target net update
        if self.total_steps % self.cfg.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # -------- internals --------

    def _build_shared_batch(self):
        """
        Returns combined batch arrays:
          s, a, r, s2, d, w (weights)
        Always returns exactly cfg.batch_size samples.
        """
        B = int(self.cfg.batch_size)

        # If sharing disabled, just use own batch.
        if (not self.cfg.experience_sharing) or self.cfg.share_mode == "off":
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * B
            return s, a, r, s2, d, w

        # Warmup: don't share until we've done enough gradient updates.
        if self.num_updates < int(self.cfg.share_warmup_updates):
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * B
            return s, a, r, s2, d, w

        # ---- diagnostics: sharing attempt ----
        self._share_attempts += 1

        # Eligible peers: enough replay + initialized nets
        eligible = [
            p for p in self._peers
            if len(p.replay) >= max(self.cfg.min_peer_replay_size, self.cfg.peer_batch_size)
               and p.policy_net is not None
        ]
        eligible_n = len(eligible)
        self._share_eligible_peers_sum += eligible_n

        # If no peers, just own batch.
        if not eligible:
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * B
            return s, a, r, s2, d, w

        # Choose up to k peers
        k = min(int(self.cfg.max_peers_per_update), eligible_n)
        peers = random.sample(eligible, k)
        self._share_selected_peers_sum += k

        # Decide how many samples come from self vs peers
        peer_total = int(round(B * float(self.cfg.share_frac)))
        peer_total = max(0, min(B, peer_total))
        own_n = B - peer_total

        # Start with own batch (weight 1)
        s, a, r, s2, d = self.replay.sample(own_n)
        w = [1.0] * own_n

        # If peer_total == 0, we are done
        if peer_total == 0:
            # pad to B defensively (shouldn't happen)
            if len(s) < B:
                ps, pa, pr, ps2, pd = self.replay.sample(B - len(s))
                s.extend(ps)
                a.extend(pa)
                r.extend(pr)
                s2.extend(ps2)
                d.extend(pd)
                w.extend([1.0] * len(ps))
            return s, a, r, s2, d, w

        # Allocate peer samples across peers without exceeding peer_batch_size per peer
        remaining = peer_total
        per_peer_ns = []
        for i in range(k):
            peers_left = k - i
            # ideal share of remaining among remaining peers
            n_i = int(math.ceil(remaining / peers_left))
            n_i = min(int(self.cfg.peer_batch_size), n_i)
            n_i = max(0, n_i)
            per_peer_ns.append(n_i)
            remaining -= n_i
            if remaining <= 0:
                break

        # If we couldn't allocate enough because of caps, we'll fill later from own replay.
        # Compute weights for each peer (mutual=1, weighted_cka = CKA)
        peer_weights = []
        peer_batches = []

        for p, n_i in zip(peers, per_peer_ns):
            if n_i <= 0:
                continue

            ps, pa, pr, ps2, pd = p.replay.sample(n_i)

            if self.cfg.share_mode == "mutual":
                weight = 1.0
            elif self.cfg.share_mode == "weighted_cka":
                ps_np = _to_f32_batch(ps)
                states_t = torch.from_numpy(ps_np).to(self.device)
                weight = avg_layer_cka(self.policy_net, p.policy_net, states_t, self.cfg.cka_layers)
                weight = float(weight)
            else:
                weight = 0.0

            peer_batches.append((ps, pa, pr, ps2, pd))
            peer_weights.append(weight)


        # Optional: normalize peer weights so "best peer" stands out more
        if peer_weights and getattr(self.cfg, "normalize_peer_weights", False):
            mean_w = float(np.mean(peer_weights))
            if mean_w > 1e-8:
                peer_weights = [w_i / mean_w for w_i in peer_weights]

        # # --- DEBUG (terminal): peer weight stats + chosen peer weights ---
        # try:
        #     import numpy as _np
        #
        #     # init once
        #     if not hasattr(self, "_dbg_share_print_every"):
        #         self._dbg_share_print_every = 10  # print every N sharing attempts
        #         self._dbg_share_print_i = 0
        #
        #     self._dbg_share_print_i += 1
        #     if self._dbg_share_print_i % self._dbg_share_print_every == 0:
        #         w_pre = _np.asarray(peer_weights, dtype=_np.float32)
        #         w_cl = _np.clip(w_pre, float(self.cfg.share_weight_floor), float(self.cfg.share_weight_ceiling))
        #
        #         def _stats(arr):
        #             if arr.size == 0:
        #                 return (float("nan"),) * 4
        #             return (float(arr.min()), float(arr.mean()), float(arr.max()), float(arr.std()))
        #
        #         pmin, pmean, pmax, pstd = _stats(w_pre)
        #         cmin, cmean, cmax, cstd = _stats(w_cl)
        #
        #         print(
        #             "\n"
        #             f"[SHARE DBG] upd={self._dbg_share_print_i} eligible={eligible_n} selected={k} "
        #             f"w_pre(min/mean/max/std)={pmin:.3f}/{pmean:.3f}/{pmax:.3f}/{pstd:.3f} "
        #             f"w_clamp(min/mean/max/std)={cmin:.3f}/{cmean:.3f}/{cmax:.3f}/{cstd:.3f} "
        #             f"chosen_w={[round(float(x), 3) for x in peer_weights]}",
        #             flush=True,
        #         )
        # except Exception as _e:
        #     print(f"\n[SHARE DBG] debug failed: {_e}", flush=True)
        # # --- END DEBUG ---

        # Append peer samples with clamped weights
        peer_samples_added = 0
        peer_weight_sum = 0.0

        for (ps, pa, pr, ps2, pd), weight in zip(peer_batches, peer_weights):
            weight = max(self.cfg.share_weight_floor, min(self.cfg.share_weight_ceiling, float(weight)))

            s.extend(ps)
            a.extend(pa)
            r.extend(pr)
            s2.extend(ps2)
            d.extend(pd)
            w.extend([weight] * len(ps))

            peer_samples_added += len(ps)
            peer_weight_sum += weight * len(ps)

        # If we undershot B (caps or empty peer_batches), fill remainder from own replay
        if len(s) < B:
            fill = B - len(s)
            fs, fa, fr, fs2, fd = self.replay.sample(fill)
            s.extend(fs)
            a.extend(fa)
            r.extend(fr)
            s2.extend(fs2)
            d.extend(fd)
            w.extend([1.0] * fill)

        # If we overshot due to rounding (shouldn't), truncate
        if len(s) > B:
            s = s[:B]
            a = a[:B]
            r = r[:B]
            s2 = s2[:B]
            d = d[:B]
            w = w[:B]

        # ---- diagnostics ----
        self._share_peer_samples += peer_samples_added
        self._share_peer_weight_sum += peer_weight_sum

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
        self._ensure_networks(input_dim=len(obs))

        if random.random() < self.cfg.epsilon:
            return random.randrange(self.num_actions)

        # --- obs layout (confirmed in main.py):
        # mastery is first block, topic_id is appended at the end via add_topic()
        topic_id = int(float(obs[-1]))
        topic_id = max(0, min(self.cfg.num_topics - 1, topic_id))
        m = float(obs[topic_id])  # mastery[topic_id]

        with torch.no_grad():
            obs_np = np.asarray(obs, dtype=np.float32)
            x = torch.from_numpy(obs_np).unsqueeze(0)
            q = self.policy_net(x).squeeze(0).detach().cpu().numpy()

        # action order is fixed by build_tutee_actions(): [quiz, explain, fix]
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


# class TuteeLowLevelAgent(DQNLowLevelAgent):
#     def __init__(self, config: LowLevelAgentConfig):
#         super().__init__(config, actions=build_tutee_actions())
