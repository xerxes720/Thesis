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
    share_frac: float = 0.70  # fraction of each batch coming from peers
    share_warmup_updates: int = 300  # don't share until this many gradient updates

    max_peers_per_update: int = 6  # sample up to this many peers each train step (for speed)
    peer_batch_size: int = 256  # how many transitions to sample from each peer
    min_peer_replay_size: int = 150  # peers must have at least this many samples to participate

    # share_weight_ema_alpha: float = 0.90  # 0.85–0.95
    # share_cka_every_updates: int = 20  # recompute CKA every N updates
    # share_min_peer_per_update: int = 1  # keep at least 1 peer if any passes gating

    share_weight_floor: float = 0.25  # clamp similarity weights
    share_weight_ceiling: float = 1.5
    share_similarity_threshold: float = 0.0  # CRITICAL: if below this, do not use peer samples

    # NEW: make high-sim peers matter more
    share_weight_power: float = 4.0  # sharpen similarity weights (>=1.0)

    cka_layers: Tuple[str, ...] = ("h1", "h2")  # which layers to use for similarity

    # NEW: normalize peer weights per update (makes CKA differences matter more)
    normalize_peer_weights: bool = True

    # --- tutee readiness thresholds (copied from KDDLearnerConfig when building tutee agent) ---
    tutee_ready_quiz: float = 0.50
    tutee_ready_explain: float = 0.60
    tutee_ready_fix: float = 0.65

    # optional: penalty magnitude for "not ready"
    tutee_not_ready_penalty: float = 0.5

    # --- Paper-style experience sharing (Algorithm 1) ---
    paper_style_sharing: bool = False  # if True, ignore share_frac / max_peers_per_update caps
    paper_peer_batch_size: int = 256  # usually == batch_size (b in paper)
    paper_use_all_peers: bool = True  # loop over all j != i

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

        w_t = w_t.clamp(0.0, 2.0)  # or 1.5 if you want strict
        w_t = w_t / w_t.mean().clamp_min(1e-8)

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
        """
        Build a minibatch with (optional) experience sharing.

        Returns:
            s, a, r, s2, d, w
            where w is a per-sample weight (float) applied to TD error / loss.
        """
        # -----------------------------
        # Fast path: no sharing
        # -----------------------------
        if (not getattr(self.cfg, "experience_sharing", False)) or getattr(self.cfg, "share_mode", "off") == "off":
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * len(s)
            return s, a, r, s2, d, w

        # -----------------------------
        # Warmup gate (in update-count space)
        # -----------------------------
        if int(self.num_updates) < int(getattr(self.cfg, "share_warmup_updates", 0)):
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * len(s)
            return s, a, r, s2, d, w

        # -----------------------------
        # Find eligible peers
        # -----------------------------
        self._share_attempts += 1

        peers_all = list(self._peers)
        if not peers_all:
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * len(s)
            return s, a, r, s2, d, w

        min_peer_size = int(getattr(self.cfg, "min_peer_replay_size", 0))
        eligible = []
        for p in peers_all:
            # must have a replay large enough and a network initialized
            if getattr(p, "replay", None) is None:
                continue
            if len(p.replay) < max(min_peer_size, 1):
                continue
            if getattr(p, "policy_net", None) is None:
                continue
            eligible.append(p)

        self._share_eligible_peers_sum += len(eligible)

        if not eligible:
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * len(s)
            return s, a, r, s2, d, w

        # -----------------------------
        # Config knobs
        # -----------------------------
        share_mode = str(getattr(self.cfg, "share_mode", "off"))
        thr = float(getattr(self.cfg, "share_similarity_threshold", 0.0))
        power = float(getattr(self.cfg, "share_weight_power", 1.0))
        w_floor = float(getattr(self.cfg, "share_weight_floor", 0.0))
        w_ceil = float(getattr(self.cfg, "share_weight_ceiling", 1.0))
        cka_layers = tuple(getattr(self.cfg, "cka_layers", ("h1", "h2")))

        # paper-style mode (optional)
        paper_style = bool(getattr(self.cfg, "paper_style_sharing", False))
        paper_peer_batch = int(getattr(self.cfg, "paper_peer_batch_size", B))
        paper_use_all = bool(getattr(self.cfg, "paper_use_all_peers", True))

        # -----------------------------
        # Choose candidate peers
        # -----------------------------
        # A: for teacher selection, we want to score all eligible peers (topics are few).
        candidates = eligible

        # -----------------------------
        # Helper: compute a peer weight
        # -----------------------------
        def _peer_weight(peer) -> float:
            if share_mode == "mutual":
                return 1.0
            if share_mode != "weighted_cka":
                return 0.0

            w = float(avg_layer_cka(self.policy_net, peer.policy_net, anchor_states, cka_layers))
            if power != 1.0:
                w = float(max(0.0, w) ** power)
            w = max(w_floor, min(w_ceil, w))
            return w

        probe_n = min(64, len(self.replay))
        probe_n = max(1, probe_n)
        ps_anchor, _, _, _, _ = self.replay.sample(probe_n)
        anchor_np = _to_f32_batch(ps_anchor)
        anchor_states = torch.from_numpy(anchor_np).to(self.device)

        # -----------------------------
        # A: Teacher selection (always pick best peer)
        # -----------------------------
        scored = []
        for p in candidates:
            wi = _peer_weight(p)
            scored.append((p, float(wi)))

        # if something went wrong / all zero, fall back to uniform teacher choice
        best_peer, best_w = max(scored, key=lambda t: t[1])
        if best_w <= 1e-12:
            best_peer = random.choice(candidates)
            best_w = 1.0

        peer_infos = [(best_peer, float(best_w))]

        # diagnostics
        self._share_selected_peers_sum += 1


        # If nobody survived gating, fall back to own batch
        if not peer_infos:
            s, a, r, s2, d = self.replay.sample(B)
            w = [1.0] * len(s)
            return s, a, r, s2, d, w

        # =========================================================
        # Mode A: "paper_style_sharing"
        #   - sample b from self
        #   - for each gated peer, sample b from peer, weight by similarity
        #   - then SUBSAMPLE down to B if we exceeded, or FILL if short
        # =========================================================
        if paper_style:
            b = max(1, min(B, int(paper_peer_batch)))

            # own chunk
            s, a, r, s2, d = self.replay.sample(b)
            w = [1.0] * len(s)

            added = 0
            peer_weight_sum = 0.0

            for p, wi in peer_infos:
                ps, pa, pr, ps2, pd = p.replay.sample(b)
                s.extend(ps);
                a.extend(pa);
                r.extend(pr);
                s2.extend(ps2);
                d.extend(pd)
                w.extend([wi] * len(ps))

                added += len(ps)
                peer_weight_sum += wi * len(ps)

            self._share_peer_samples += added
            self._share_peer_weight_sum += peer_weight_sum

            # If we overshot B, subsample uniformly back to B (keeps mixture unbiased)
            if len(s) > B:
                idx = np.random.choice(len(s), size=B, replace=False)
                s = [s[i] for i in idx]
                a = [a[i] for i in idx]
                r = [r[i] for i in idx]
                s2 = [s2[i] for i in idx]
                d = [d[i] for i in idx]
                w = [w[i] for i in idx]

            # If short, fill remainder with own
            if len(s) < B:
                missing = B - len(s)
                ps, pa, pr, ps2, pd = self.replay.sample(missing)
                s.extend(ps);
                a.extend(pa);
                r.extend(pr);
                s2.extend(ps2);
                d.extend(pd)
                w.extend([1.0] * len(ps))

            return s, a, r, s2, d, w

        # =========================================================
        # Mode B: True weighted transfer (recommended)
        #   - keep batch size fixed at B
        #   - allocate peer_total = round(B*share_frac) across peers by weight
        #   - sample that many from each peer
        # =========================================================
        peer_total = int(round(B * float(getattr(self.cfg, "share_frac", 0.15))))
        peer_total = max(0, min(B, peer_total))

        min_w = 0.35  # tune 0.25–0.45
        if best_w < min_w:
            peer_total = int(round(B * 0.20))  # only 20% peer this update

        own_n = B - peer_total

        # own samples
        s, a, r, s2, d = self.replay.sample(own_n)
        w = [1.0] * len(s)

        if peer_total <= 0:
            # fill to B if needed
            missing = B - len(s)
            if missing > 0:
                ps, pa, pr, ps2, pd = self.replay.sample(missing)
                s.extend(ps);
                a.extend(pa);
                r.extend(pr);
                s2.extend(ps2);
                d.extend(pd)
                w.extend([1.0] * len(ps))
            return s, a, r, s2, d, w

        weights = np.asarray([wi for _, wi in peer_infos], dtype=np.float64)
        wsum = float(weights.sum())
        if wsum <= 1e-12:
            probs = np.ones_like(weights) / max(1, len(weights))
        else:
            probs = weights / wsum

        # alloc = np.random.multinomial(peer_total, probs).tolist()
        # NEW: deterministic allocation (much smoother)
        expected = probs * peer_total
        alloc = np.floor(expected).astype(int)

        rem = peer_total - int(alloc.sum())
        if rem > 0:
            # distribute remainder to highest fractional parts
            frac = expected - alloc
            idx = np.argsort(-frac)[:rem]
            alloc[idx] += 1

        alloc = alloc.tolist()

        added = 0
        peer_weight_sum = 0.0

        for (p, wi), n_i in zip(peer_infos, alloc):
            if n_i <= 0:
                continue
            ps, pa, pr, ps2, pd = p.replay.sample(int(n_i))
            s.extend(ps);
            a.extend(pa);
            r.extend(pr);
            s2.extend(ps2);
            d.extend(pd)
            w.extend([wi] * len(ps))

            added += len(ps)
            peer_weight_sum += wi * len(ps)

        # fill remainder with own if short
        missing = B - len(s)
        if missing > 0:
            ps, pa, pr, ps2, pd = self.replay.sample(missing)
            s.extend(ps);
            a.extend(pa);
            r.extend(pr);
            s2.extend(ps2);
            d.extend(pd)
            w.extend([1.0] * len(ps))

        self._share_peer_samples += added
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
