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
    batch_size: int = 64
    min_replay_size: int = 150

    # --- single-LL fairness option: keep separate replay buffers per topic ---
    per_topic_replay: bool = False
    # How to pick which topic-buffer to train from when per_topic_replay=True
    # - "current": train on the buffer for the topic that generated this transition
    # - "uniform": sample a topic uniformly among buffers with enough data
    per_topic_sample_mode: str = "current"
    # Capacity per topic when per_topic_replay=True. If None, uses buffer_size // num_topics.
    per_topic_buffer_size: Optional[int] = None
    train_every_steps: int = 20
    # target_update_steps: int = 1_000
    # To be faithful to the original paper: k=5
    target_update_steps: int = 5 * train_every_steps
    experience_sharing: bool = True
    # experience_sharing: bool = False
    share_mode: str = "weighted_cka"  # options: "off", "mutual", "weighted_cka"

    max_grad_norm: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- tutee readiness thresholds (copied from KDDLearnerConfig when building tutee agent) ---
    tutee_ready_quiz: float = 0.45
    tutee_ready_explain: float = 0.50
    tutee_ready_fix: float = 0.55
    tutee_not_ready_penalty: float = 0.0  # no longer used

    # share_frac: float = 0.2          # 0.3–0.7 works; start 0.5
    share_warmup_updates: int = 300  # same idea as you already use :contentReference[oaicite:2]{index=2}
    # shared_buffer_size: int = 200_000 # independent from per-agent buffer

    cka_layers: Tuple[str, ...] = ("h1", "h2")
    # cka_probe_n: int = 64
    cka_power: float = 1.0  # keep simple; optional

    # --- Experience sharing stability knobs ---
    share_max_weight: float = 0.60  # cap peer sample weight to prevent over-trust
    share_weight_ema: float = 0.90  # EMA smoothing for peer weights (0 disables)

    # shared_loss_weight: float = 1.0  # base weight multiplier for shared samples

    # --- Experience sharing speed controls ---
    cka_every_updates: int = 50  # recompute similarity only every N updates
    # --- experience sharing knobs ---
    share_frac: float = 0.20
    max_peers_per_update: int = 2
    min_peer_replay_size: int = 200
    share_similarity_threshold: float = 0.25
    cka_probe_n: int = 64
    share_stop_updates: int = 10 ** 9

    # --- Paper-faithful sharing (Algorithm 1) ---
    # If True: build batch as Bi (size=b) + sum_j Bj (size=b/N each peer),
    # weight Bj by wj computed from CKA on peer's sampled states.
    share_paper_batch: bool = True
    # If 0 -> use 1/num_topics as b/N fraction per peer
    share_peer_frac_per_peer: float = 0.0
    # "sumw" (your current) vs "mean" (paper-faithful weighting behavior)
    share_loss_norm: str = "mean"

    # Similarity metric used when share_mode == "weighted_cka"
    # - "cka": linear CKA on hidden reps (your current behavior)
    # - "q_cos": cosine similarity between per-state Q-vectors (policy-aligned)
    # - "q_argmax": argmax agreement rate (strict, noisier)
    share_similarity_metric: str = "cka"

    # Probe distribution for similarity computation:
    # fraction of probe states drawn from the peer replay (0=self-only, 0.5=symmetric).
    share_probe_peer_frac: float = 0.0
    # soft target update (Polyak). If None, use hard update every target_update_steps.
    target_soft_tau: float | None = None
    use_huber_loss: bool = False


# ---------------- Replay Buffer ----------------

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None):
        self.buffer = deque(maxlen=capacity)
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        self.np_rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done):
        # store as float32 numpy arrays ONCE to avoid repeated conversion later
        s = np.asarray(s, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)
        self.buffer.append((s, int(a), float(r), s2, bool(done)))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        n = len(self)
        idx = self.np_rng.integers(0, n, size=batch_size)  # with replacement
        batch = [self.buffer[i] for i in idx]
        s, a, r, s2, d = zip(*batch)

        # stack -> contiguous arrays (fast path)
        s_np = np.ascontiguousarray(np.stack(s, axis=0), dtype=np.float32)
        s2_np = np.ascontiguousarray(np.stack(s2, axis=0), dtype=np.float32)
        a_np = np.ascontiguousarray(np.fromiter(a, dtype=np.int64, count=batch_size))
        r_np = np.ascontiguousarray(np.fromiter(r, dtype=np.float32, count=batch_size))
        d_np = np.ascontiguousarray(np.fromiter(d, dtype=np.float32, count=batch_size))
        return s_np, a_np, r_np, s2_np, d_np


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


def q_cosine_similarity(
    net_a: QNetwork,
    net_b: QNetwork,
    states: torch.Tensor,
    center: bool = True,
    eps: float = 1e-8,
) -> float:
    """
    Cosine similarity between per-state Q vectors, mapped to [0, 1].
    This is more policy-aligned than CKA because it compares action preferences directly.
    """
    with torch.no_grad():
        qa = net_a(states)
        qb = net_b(states)

        if center:
            qa = qa - qa.mean(dim=1, keepdim=True)
            qb = qb - qb.mean(dim=1, keepdim=True)

        qa = qa / qa.norm(dim=1, keepdim=True).clamp_min(eps)
        qb = qb / qb.norm(dim=1, keepdim=True).clamp_min(eps)

        cos = (qa * qb).sum(dim=1).mean()  # in [-1, 1]
        sim01 = ((cos + 1.0) * 0.5).clamp(0.0, 1.0)
        return float(sim01.item())


def q_argmax_agreement(
    net_a: QNetwork,
    net_b: QNetwork,
    states: torch.Tensor,
) -> float:
    """
    Fraction of states where argmax_a(Q) == argmax_b(Q), in [0, 1].
    """
    with torch.no_grad():
        qa = net_a(states)
        qb = net_b(states)
        aa = torch.argmax(qa, dim=1)
        ab = torch.argmax(qb, dim=1)
        return float((aa == ab).float().mean().item())

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
        # Optional: in single-LL mode we can keep replay separated per topic to remove
        # the implicit cross-topic mini-batch advantage of a shared LL network.
        self._topic_replays: Optional[List[ReplayBuffer]] = None
        if bool(getattr(self.cfg, "per_topic_replay", False)):
            cap = getattr(self.cfg, "per_topic_buffer_size", None)
            if cap is None:
                cap = max(1000, int(self.cfg.buffer_size) // max(1, int(self.cfg.num_topics)))
            self._topic_replays = [ReplayBuffer(int(cap)) for _ in range(int(self.cfg.num_topics))]

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
        self._cka_cache = {}  # peer_id -> (last_update, sim_w)
        self._last_cka_update = -10 ** 9

        # optional peer-gating scores (e.g., action-effect similarity computed in main)
        # maps id(peer) -> similarity in [-1, 1]
        self._peer_gate_sims: dict[int, float] = {}
        self._peer_gate_thr: float = 0.2  # if > -1, peers below this sim are ignored
        self._peer_gate_power: float = 2.0

        # self.shared_replay: Optional[ReplayBuffer] = None
        # self.share_ref_net: Optional[QNetwork] = None

    def _get_peer_weight(self, peer: "DQNLowLevelAgent", probe_states_np: np.ndarray) -> float:
        """
        Returns a stable peer weight in [0, share_max_weight].

        - mutual: always 1.0 (binary sharing baseline)
        - weighted_cka: CKA-based similarity with:
            * recompute throttling (cka_every_updates)
            * thresholding (share_similarity_threshold)
            * power mapping (cka_power)
            * EMA smoothing (share_weight_ema)
            * hard cap (share_max_weight)
        """
        mode = str(getattr(self.cfg, "share_mode", "off"))
        if mode == "mutual":
            return 1.0

        peer_id = id(peer)

        every = int(getattr(self.cfg, "cka_every_updates", 50))

        tau = float(getattr(self.cfg, "share_similarity_threshold", 0.75))
        tau = max(0.0, min(0.999, tau))

        power = float(getattr(self.cfg, "cka_power", 1.0))

        w_cap = float(getattr(self.cfg, "share_max_weight", 0.60))
        w_cap = max(0.0, min(1.0, w_cap))

        ema = float(getattr(self.cfg, "share_weight_ema", 0.90))
        ema = max(0.0, min(0.999, ema))

        layers = tuple(getattr(self.cfg, "cka_layers", ("h1", "h2")))

        cached = self._cka_cache.get(peer_id, None)
        if cached is not None:
            last_u, w_cached = cached
            if (self.num_updates - last_u) < every:
                return float(w_cached)
        else:
            w_cached = 0.0

        metric = str(getattr(self.cfg, "share_similarity_metric", "cka")).lower().strip()
        peer_frac = float(getattr(self.cfg, "share_probe_peer_frac", 0.0))
        peer_frac = max(0.0, min(1.0, peer_frac))

        # Optional symmetric probe: mix in some states from the peer replay
        probe = probe_states_np
        if peer_frac > 0.0 and isinstance(probe_states_np, np.ndarray) and probe_states_np.ndim == 2:
            total_n = int(probe_states_np.shape[0])
            n_peer = max(0, min(total_n, int(round(total_n * peer_frac))))
            n_self = total_n - n_peer

            parts = []

            if n_self > 0:
                idxs = np.random.randint(0, probe_states_np.shape[0], size=n_self)
                parts.append(np.ascontiguousarray(probe_states_np[idxs], dtype=np.float32))

            if n_peer > 0 and len(peer.replay) >= n_peer:
                peer_states, _, _, _, _ = peer.replay.sample(n_peer)
                parts.append(np.ascontiguousarray(peer_states, dtype=np.float32))

            if parts:
                probe = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)

        with torch.no_grad():
            q_t = torch.from_numpy(probe).to(self.device)

            if metric in ("q_cos", "qcos", "q_cosine"):
                sim = float(q_cosine_similarity(self.policy_net, peer.policy_net, q_t, center=True))
            elif metric in ("q_argmax", "argmax", "qargmax"):
                sim = float(q_argmax_agreement(self.policy_net, peer.policy_net, q_t))
            else:
                sim = float(avg_layer_cka(self.policy_net, peer.policy_net, q_t, layers))

        sim = max(0.0, min(1.0, sim))

        # threshold + normalize [tau, 1] -> [0, 1]
        # threshold + normalize [tau, 1] -> [0, 1]
        if sim < tau:
            w_target = 0.0
        else:
            sim01 = (sim - tau) / max(1e-6, (1.0 - tau))
            w_target = float(sim01 ** power)

        # cap to prevent "over-trusting" any single peer
        w_target = min(float(w_target), float(w_cap))

        # IMPORTANT: if similarity drops below tau, hard-stop sharing (no EMA tail).
        # if w_target <= 0.0:
        #     w_new = 0.0
        # elif ema > 0.0:
        #     w_new = float(ema * float(w_cached) + (1.0 - ema) * float(w_target))
        # else:
        #     w_new = float(w_target)
        # smoother behavior (no hard-stop flip):
        if ema > 0.0:
            w_new = float(ema * float(w_cached) + (1.0 - ema) * float(w_target))
        else:
            w_new = float(w_target)

        w_new = max(0.0, float(w_new))

        self._cka_cache[peer_id] = (int(self.num_updates), float(w_new))
        return float(w_new)

    def _apply_peer_gate(self, peer: "DQNLowLevelAgent", w: float) -> float:
        # Optionally modulate a peer weight by an externally provided similarity score.
        if not getattr(self, "_peer_gate_sims", None):
            return float(w)
        s = float(self._peer_gate_sims.get(id(peer), 1.0))
        thr = float(getattr(self, "_peer_gate_thr", -1.0))

        if thr > -0.5 and s < thr:
            return 0.0
        s = max(0.0, s)  # negative similarity => disable
        p = float(getattr(self, "_peer_gate_power", 1.0))
        if p != 1.0:
            s = float(s ** p)

        return float(w) * float(s)

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

    # def set_peers(self, peers: List["DQNLowLevelAgent"]) -> None:
    # ---------- per-topic replay helpers (single-LL fairness) ----------

    def _has_topic_replay(self) -> bool:
        return (self._topic_replays is not None) and (len(self._topic_replays) == int(self.cfg.num_topics))

    def _topic_replay(self, topic_id: int) -> ReplayBuffer:
        assert self._topic_replays is not None
        tid = int(topic_id)
        if tid < 0 or tid >= len(self._topic_replays):
            raise ValueError(f"topic_id out of range: {topic_id}")
        return self._topic_replays[tid]

    def _choose_train_topic(self, prefer_topic_id: Optional[int]) -> Optional[int]:
        """Choose which topic-buffer to sample from when per_topic_replay=True."""
        if not self._has_topic_replay():
            return None

        mode = str(getattr(self.cfg, "per_topic_sample_mode", "current"))

        # default: current topic

        if mode == "current" and prefer_topic_id is not None:
            return int(prefer_topic_id)

        # uniform among topic buffers with enough data
        need = int(max(self.cfg.min_replay_size, self.cfg.batch_size))
        ok = [t for t in range(int(self.cfg.num_topics)) if len(self._topic_replay(t)) >= need]
        if not ok:
            return int(prefer_topic_id) if prefer_topic_id is not None else None
        return int(np.random.choice(ok))

    def _sample_own(
            self, B: int, topic_id: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch from own replay (global or per-topic)."""
        if self._has_topic_replay():
            tid = self._choose_train_topic(topic_id)
            if tid is None:
                return self.replay.sample(B)

            return self._topic_replay(tid).sample(B)

        return self.replay.sample(B)

    def set_peers(
            self,
            peers: List["DQNLowLevelAgent"],
            sims: Optional[List[float]] = None,
            sim_threshold: Optional[float] = None,
            sim_power: Optional[float] = None,
    ) -> None:
        """
        Set peer agents used for experience sharing.
        Optionally pass per-peer similarity scores (e.g., action-effect cosine similarity)
        which will be combined with CKA to reduce negative transfer.
                """
        # avoid self references
        self._peers = [p for p in peers if p is not self]
        self._peer_gate_sims = {}

        if sims is not None:
            for p, s in zip(peers, sims):
                if p is self:
                    continue
                self._peer_gate_sims[id(p)] = float(s)

        if sim_threshold is not None:
            self._peer_gate_thr = float(sim_threshold)

        if sim_power is not None:
            self._peer_gate_power = float(sim_power)
        # prune cached peer weights for peers that are no longer allowed (prevents stale negative transfer)
        allowed = {id(p) for p in self._peers}
        if self._cka_cache:
            self._cka_cache = {pid: v for pid, v in self._cka_cache.items() if pid in allowed}

    def select_action(self, obs: List[float]) -> int:
        self._ensure_networks(input_dim=len(obs))

        if random.random() < self.cfg.epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            obs_np = np.asarray(obs, dtype=np.float32)
            x = torch.from_numpy(obs_np).unsqueeze(0)  # CPU
            q = self.policy_net(x)
            return int(q.argmax(dim=1).item())

    def update(self, obs: List[float], action: int, reward: float, next_obs: List[float], done: bool,
               topic_id: Optional[int] = None) -> None:
        self._ensure_networks(input_dim=len(obs))

        # store own transition
        # obs/next_obs are freshly created lists from env; do not copy (major speed win)
        # self.replay.push(obs, int(action), float(reward), next_obs, bool(done))
        # self.replay.push(obs, action, reward, next_obs, done)

        if self._has_topic_replay():
            tid = int(topic_id) if topic_id is not None else 0
            self._topic_replay(tid).push(obs, action, reward, next_obs, done)
        else:
            self.replay.push(obs, action, reward, next_obs, done)

        # if self.shared_replay is not None:
        #     self.shared_replay.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

        if self.total_steps % self.cfg.train_every_steps != 0:
            return

        if self._has_topic_replay():
            tid = self._choose_train_topic(topic_id)
            if len(self._topic_replay(tid)) < max(self.cfg.min_replay_size, self.cfg.batch_size):
                return
        else:

            if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
                return

        # if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
        #     return

        # -------- Build training batch: own + shared --------
        batch_s, batch_a, batch_r, batch_s2, batch_d, batch_w = self._build_shared_batch(self.cfg.batch_size,
                                                                                                  topic_id=topic_id)

        s_np, a_np, r_np, s2_np, d_np, w_np = batch_s, batch_a, batch_r, batch_s2, batch_d, batch_w

        s_t = torch.from_numpy(s_np)
        s2_t = torch.from_numpy(s2_np)
        a_t = torch.from_numpy(a_np).unsqueeze(1)
        r_t = torch.from_numpy(r_np)
        d_t = torch.from_numpy(d_np)
        if isinstance(batch_w, np.ndarray):
            w_np = np.ascontiguousarray(batch_w.astype(np.float32, copy=False))
        else:
            w_np = np.ascontiguousarray(np.asarray(batch_w, dtype=np.float32))

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
        # loss = (w_t * (td ** 2)).sum() / (w_t.sum().clamp_min(1e-8))

        use_huber = bool(getattr(self.cfg, "use_huber_loss", False))
        if use_huber:
            loss_per = torch.nn.functional.smooth_l1_loss(q_sa, target, reduction="none")
        else:
            loss_per = (q_sa - target) ** 2

        norm_mode = str(getattr(self.cfg, "share_loss_norm", "mean"))
        if norm_mode == "sumw":
            loss = (w_t * loss_per).sum() / (w_t.sum().clamp_min(1e-8))
        else:
            loss = (w_t * loss_per).mean()
        # norm_mode = str(getattr(self.cfg, "share_loss_norm", "mean"))
        #
        # if norm_mode == "sumw":
        #     # your previous behavior
        #     loss = (w_t * (td ** 2)).sum() / (w_t.sum().clamp_min(1e-8))
        # else:
        #     # paper-faithful: weights scale sample contribution but do not renormalize by sum(w)
        #     loss = (w_t * (td ** 2)).mean()

        # loss_per = torch.nn.functional.smooth_l1_loss(q_sa, target, reduction="none")  # Huber
        # loss = (w_t * loss_per).sum() / w_t.sum().clamp_min(1e-8)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.num_updates += 1

        # target net update
        tau = getattr(self.cfg, "target_soft_tau", None)
        if tau is not None and float(tau) > 0.0:
            # Polyak / soft update every gradient step
            with torch.no_grad():
                for p, pt in zip(self.policy_net.parameters(), self.target_net.parameters()):
                    pt.data.mul_(1.0 - float(tau)).add_(p.data, alpha=float(tau))
        else:
            # fallback: hard copy
            if self.num_updates % max(1, int(self.cfg.target_update_steps)) == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    # -------- internals --------

    def _build_shared_batch(self, B: int, topic_id: Optional[int] = None):
        mode = str(getattr(self.cfg, "share_mode", "off"))

        # --- helpers ---
        def _cat2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            # concatenate along batch dimension
            if x.size == 0:
                return y
            if y.size == 0:
                return x
            return np.concatenate([x, y], axis=0)

        # --- no sharing / warmup ---
        if (mode == "off") or (not getattr(self.cfg, "experience_sharing", False)):
            s, a, r, s2, d = self._sample_own(B, topic_id)
            w = np.ones((B,), dtype=np.float32)
            return s, a, r, s2, d, w

        if int(self.num_updates) < int(getattr(self.cfg, "share_warmup_updates", 0)):
            s, a, r, s2, d = self._sample_own(B, topic_id)
            w = np.ones((B,), dtype=np.float32)
            return s, a, r, s2, d, w

        # stop sharing after a given update (avoids harming late-stage specialists)
        if int(self.num_updates) >= int(getattr(self.cfg, "share_stop_updates", 10 ** 9)):
            s, a, r, s2, d = self._sample_own(B, topic_id)
            w = np.ones((B,), dtype=np.float32)
            return s, a, r, s2, d, w

        peers = list(getattr(self, "_peers", []) or [])
        if len(peers) == 0:
            s, a, r, s2, d = self._sample_own(B, topic_id)
            w = np.ones((B,), dtype=np.float32)
            return s, a, r, s2, d, w

        # ---- paper-faithful batch (Algorithm 1) ----

        if bool(getattr(self.cfg, "share_paper_batch", False)):
            return self._build_shared_batch_paper(B, mode, topic_id=topic_id)
        # ----- fixed batch budget -----
        share_frac = float(getattr(self.cfg, "share_frac", 0.25))
        share_B = int(round(B * share_frac))
        share_B = max(0, min(B, share_B))
        own_B = B - share_B

        # sample own portion
        s, a, r, s2, d = self._sample_own(own_B, topic_id)
        w = np.ones((own_B,), dtype=np.float32)

        # if can't share for some reason, pad with own
        if (self.policy_net is None) or (mode not in ("mutual", "weighted_cka")) or (share_B <= 0):
            if own_B < B:
                ps, pa, pr, ps2, pd = self.replay.sample(B - own_B)
                s = _cat2(s, ps)
                a = _cat2(a, pa)
                r = _cat2(r, pr)
                s2 = _cat2(s2, ps2)
                d = _cat2(d, pd)
                w = _cat2(w, np.ones((B - own_B,), dtype=np.float32))
            return s, a, r, s2, d, w

        # ----- compute sims, pick top-k -----
        cka_layers = tuple(getattr(self.cfg, "cka_layers", ("h1", "h2")))
        raw_tau = float(getattr(self.cfg, "share_similarity_threshold", 0.30))
        raw_tau = max(0.0, min(0.999, raw_tau))
        power = float(getattr(self.cfg, "cka_power", 2.0))

        kmax = int(getattr(self.cfg, "max_peers_per_update", 1))
        min_peer = int(getattr(self.cfg, "min_peer_replay_size", B))

        eligible = [p for p in peers if (p.policy_net is not None) and (len(p.replay) >= min_peer)]
        if not eligible:
            # pad with own
            if own_B < B:
                ps, pa, pr, ps2, pd = self.replay.sample(B - own_B)
                s = _cat2(s, ps)
                a = _cat2(a, pa)
                r = _cat2(r, pr)
                s2 = _cat2(s2, ps2)
                d = _cat2(d, pd)
                w = _cat2(w, np.ones((B - own_B,), dtype=np.float32))
            return s, a, r, s2, d, w

        scored = []
        if mode == "mutual":
            random.shuffle(eligible)
            chosen = eligible[:kmax]
            # scored = [(1.0, p) for p in chosen]
            scored = [(float(self._apply_peer_gate(p, 1.0)), p) for p in chosen]
            scored = [(w, p) for (w, p) in scored if w > 0.0]
        else:
            # Build ONE probe batch per update and reuse across peers (reduces noise/spikes)
            probe_n = int(getattr(self.cfg, "cka_probe_n", 64))
            probe_n = max(8, min(probe_n, max(8, s.shape[0])))

            # probe on SELF distribution (random subset avoids bias to early rows)
            if isinstance(s, np.ndarray) and s.shape[0] >= probe_n:
                idxs = np.random.randint(0, s.shape[0], size=probe_n)
                probe_states = s[idxs]
            else:
                probe_states, *_ = self.replay.sample(probe_n)

            probe_states_np = np.ascontiguousarray(probe_states, dtype=np.float32)

            for p in eligible:
                sim_w = float(self._get_peer_weight(p, probe_states_np))
                sim_w = float(self._apply_peer_gate(p, sim_w))
                if sim_w <= 0.0:
                    continue
                scored.append((sim_w, p))

            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:kmax]

        # No eligible peers -> pad with own
        if not scored:
            if own_B < B:
                ps, pa, pr, ps2, pd = self.replay.sample(B - own_B)
                s = _cat2(s, ps)
                a = _cat2(a, pa)
                r = _cat2(r, pr)
                s2 = _cat2(s2, ps2)
                d = _cat2(d, pd)
                w = _cat2(w, np.ones((B - own_B,), dtype=np.float32))
            return s, a, r, s2, d, w

        # ----- allocate share_B across selected peers -----
        self._share_attempts += 1
        self._share_eligible_peers_sum += len(eligible)

        per_peer = max(1, share_B // len(scored))
        remaining = share_B

        for sim_w, p in scored:
            take = min(per_peer, remaining)
            if take <= 0:
                break

            ps, pa, pr, ps2, pd = p.replay.sample(take)

            self._share_selected_peers_sum += 1
            self._share_peer_samples += take
            self._share_peer_weight_sum += float(sim_w) * take

            s = _cat2(s, ps)
            a = _cat2(a, pa)
            r = _cat2(r, pr)
            s2 = _cat2(s2, ps2)
            d = _cat2(d, pd)
            w = _cat2(w, np.full((take,), float(sim_w), dtype=np.float32))

            remaining -= take

        # pad if rounding left us short
        cur = s.shape[0]
        if cur < B:
            ps, pa, pr, ps2, pd = self.replay.sample(B - cur)
            s = _cat2(s, ps)
            a = _cat2(a, pa)
            r = _cat2(r, pr)
            s2 = _cat2(s2, ps2)
            d = _cat2(d, pd)
            w = _cat2(w, np.ones((B - cur,), dtype=np.float32))

        # final safety
        assert s.shape[0] == B and a.shape[0] == B and r.shape[0] == B and s2.shape[0] == B and d.shape[0] == B, \
            f"Batch size mismatch: s={s.shape}, a={a.shape}, r={r.shape}, s2={s2.shape}, d={d.shape}"
        assert w.shape[0] == B, f"Weight size mismatch: w={w.shape}, expected {B}"

        return s, a, r, s2, d, w

    def _build_shared_batch_paper(self, B: int, mode: str, topic_id: Optional[int] = None):

        def _cat2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            if x.size == 0: return y
            if y.size == 0: return x
            return np.concatenate([x, y], axis=0)

        # Self batch
        s, a, r, s2, d = self._sample_own(B, topic_id)
        w = np.ones((B,), dtype=np.float32)

        # Eligible peers
        min_peer = int(getattr(self.cfg, "min_peer_replay_size", B))
        peers = [p for p in self._peers if (p.policy_net is not None) and (len(p.replay) >= min_peer)]
        if not peers or self.policy_net is None:
            return s, a, r, s2, d, w

        self._share_attempts += 1
        self._share_eligible_peers_sum += len(peers)

        # ---- score peers (stable weights) ----
        # ---- score peers (stable weights) ----
        # IMPORTANT: probe similarity on *self* state distribution (or a shared probe batch).
        # Using peer states here estimates "how similar peer j is to itself", which is not what
        # we need for transfer into *this* agent.
        scored = []
        probe_n = int(getattr(self.cfg, "cka_probe_n", B))
        probe_n = max(8, min(int(probe_n), int(s.shape[0]) if isinstance(s, np.ndarray) else int(B)))

        if mode != "mutual":
            # Reuse the same probe batch across all peers (less noise, fewer CKA calls).
            if isinstance(s, np.ndarray) and s.shape[0] >= probe_n:
                idxs = np.random.randint(0, s.shape[0], size=probe_n)
                probe_states = s[idxs]
            else:
                probe_states, *_ = self.replay.sample(probe_n)
            probe_states_np = np.ascontiguousarray(probe_states, dtype=np.float32)
        else:
            probe_states_np = None  # unused

        for p in peers:
            if mode == "mutual":
                sim_w = 1.0
                sim_w = float(self._apply_peer_gate(p, 1.0))
            else:
                sim_w = float(self._get_peer_weight(p, probe_states_np))  # <-- tau/cap/EMA inside
                sim_w = float(self._apply_peer_gate(p, sim_w))

            if sim_w > 0.0:
                scored.append((sim_w, p))

        if not scored:
            return s, a, r, s2, d, w

        # ---- pick top-k peers ----
        kmax = int(getattr(self.cfg, "max_peers_per_update", 2))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:max(1, kmax)]
        self._share_selected_peers_sum += len(scored)

        # ---- fixed peer budget ----
        share_frac = float(getattr(self.cfg, "share_frac", 0.20))
        peer_budget = max(1, int(round(B * share_frac)))
        per_peer = max(1, peer_budget // max(1, len(scored)))


        # ---- append peer samples ----
        min_w = float(getattr(self.cfg, "share_min_effective_weight", 0.05))

        for base_w, p in scored:
            # base_w is whatever your scored list computed (cka/qcos/etc.)
            # Apply peer-gate on top of that (action-effect similarity gate)
            eff_w = float(self._apply_peer_gate(p, float(base_w)))

            # Drop peers with tiny effective weight (prevents noisy sharing)
            if eff_w < min_w:
                continue

            ps, pa, pr, ps2, pd = p.replay.sample(per_peer)

            s = _cat2(s, ps)
            a = _cat2(a, pa)
            r = _cat2(r, pr)
            s2 = _cat2(s2, ps2)
            d = _cat2(d, pd)
            w = _cat2(w, np.full((ps.shape[0],), eff_w, dtype=np.float32))

            self._share_peer_samples += int(ps.shape[0])
            self._share_peer_weight_sum += float(eff_w) * float(ps.shape[0])

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

        m = float(obs[0])

        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(x).squeeze(0).detach().cpu().numpy()

        idx_quiz, idx_explain, idx_fix = 0, 1, 2

        # hard mask not-ready actions
        if m < self.cfg.tutee_ready_quiz:
            q[idx_quiz] = -1e9
        if m < self.cfg.tutee_ready_explain:
            q[idx_explain] = -1e9
        if m < self.cfg.tutee_ready_fix:
            q[idx_fix] = -1e9

        return int(q.argmax())


def build_tutee_actions() -> List[str]:
    # align 1:1 with learner_model LowLevelAction semantics
    return [
        "tutee_quiz",
        "tutee_explain",
        "tutee_fix",
    ]
