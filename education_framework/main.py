# education_framework/main.py

from __future__ import annotations
from education_framework.agents.high_level_agent import HighLevelAgent, HighLevelAgentConfig
from education_framework.agents.low_level_agents import (
    DQNLowLevelAgent,
    TutorLowLevelAgent,
    TuteeLowLevelAgent,
    LowLevelAgentConfig,
    build_tutor_actions,
    build_tutee_actions,
)

from education_framework.environment.learner_model import (
    KDDLearnerModel,
    KDDLearnerConfig,
    KDDModelBundle,
    ActionMeta,
    LowLevelAction,
)
import itertools

import argparse
import csv
import math
import random
import sys
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# --- Ensure imports work whether you run:
#   python -m education_framework.main
# or:
#   python education_framework/main.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# RUN WITH python -m education_framework.main --bundle education_framework/data/kdd_bundle.joblib --episodes 2000 --max_steps 300

import numpy as np
import torch


def _ll_pairwise_action_agreement(
        tutor_agents: List["DQNLowLevelAgent"],
        batch: int = 256,
) -> float:
    """
    Measures how similar tutor LL agents are by comparing greedy actions on states
    sampled from their replay buffers. Returns mean agreement over all i<j pairs.
    - If only 1 agent exists or not enough replay, returns NaN.
    """
    if len(tutor_agents) <= 1:
        return float("nan")

    # Ensure networks exist
    for ag in tutor_agents:
        if getattr(ag, "policy_net", None) is None:
            return float("nan")

    # Collect pairwise agreements
    agreements: List[float] = []
    with torch.no_grad():
        for i in range(len(tutor_agents)):
            ag_i = tutor_agents[i]
            if len(getattr(ag_i.replay, "buffer", [])) < max(50, batch):
                continue
            # sample states from agent i replay; ignore the rest of tuple
            s_i, _, _, _, _ = ag_i.replay.sample(batch)
            s_i_t = torch.tensor(s_i, dtype=torch.float32, device=ag_i.device)

            q_i = ag_i.policy_net(s_i_t)
            a_i = torch.argmax(q_i, dim=1)

            for j in range(i + 1, len(tutor_agents)):
                ag_j = tutor_agents[j]
                q_j = ag_j.policy_net(s_i_t.to(ag_j.device))
                a_j = torch.argmax(q_j, dim=1).to(a_i.device)
                agreements.append(float((a_i == a_j).float().mean().item()))

    if len(agreements) == 0:
        return float("nan")
    return float(np.mean(agreements))


def _safe_mean(x):
    return float(sum(x) / max(1, len(x)))


def topic_entropy(topic_ids):
    """Normalized entropy in [0,1] for topic selection concentration."""
    if not topic_ids:
        return 0.0
    c = Counter(topic_ids)
    n = sum(c.values())
    probs = [v / n for v in c.values()]
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    ent_max = math.log(len(c) + 1e-12)
    return float(ent / (ent_max + 1e-12))


# ----------------------------
# Environment wrapper (KDD)
# ----------------------------

@dataclass
class KDDEnvConfig:
    num_topics: int = 7
    max_steps: int = 200
    initial_mastery: float = 0.2
    lambda_step: float = 0.003  # NEW: reward penalty per step-cost unit


class KDDHierEnv:
    """
    Thin wrapper so your existing main/run_episode can stay mostly unchanged.

    API compatibility:
      - reset() -> obs (list[float])
      - step_tutor(topic_id, action_str) -> (obs, reward, done, info)
      - step_tutee(topic_id, action_str) -> (obs, reward, done, info)
      - num_topics attribute
    """

    def __init__(
            self,
            *,
            bundle: KDDModelBundle,
            cfg: Optional[KDDEnvConfig] = None,
            learner_cfg: Optional[KDDLearnerConfig] = None,
            seed: int = 0,
    ) -> None:
        self.cfg = cfg or KDDEnvConfig(num_topics=bundle.n_topics)
        self.num_topics = int(self.cfg.num_topics)
        self.max_steps = int(self.cfg.max_steps)
        self.lambda_step = float(self.cfg.lambda_step)

        self.model = KDDLearnerModel(
            cfg=learner_cfg or KDDLearnerConfig(n_topics=self.num_topics),
            bundle=bundle,
            seed=seed,
        )
        self.step_count = 0

        # String -> KDD action mapping (keep simple and stable)
        self._tutor_action_map: Dict[str, ActionMeta] = {
            # tutor actions from build_tutor_actions()
            'quiz': ActionMeta(action=LowLevelAction.TUTOR_QUIZ, is_tutee=False, force_generation=False),
            'hint': ActionMeta(action=LowLevelAction.TUTOR_HINT, is_tutee=False, force_generation=False),
            'worked_example': ActionMeta(action=LowLevelAction.TUTOR_WORKED_EXAMPLE, is_tutee=False,
                                         force_generation=False),
            'remediation': ActionMeta(action=LowLevelAction.TUTOR_REMEDIATION, is_tutee=False, force_generation=False),
            'review': ActionMeta(action=LowLevelAction.TUTOR_REVIEW, is_tutee=False, force_generation=False),

            # safety fallback used by old main
            'no_help': ActionMeta(action=LowLevelAction.TUTOR_QUIZ, is_tutee=False, force_generation=False),
        }

        self._tutee_action_map: Dict[str, ActionMeta] = {
            # tutee actions from build_tutee_actions()
            # Map to *distinct* action ids (5..7) so the QualityTreeBank can assign separate effects.
            'tutee_quiz': ActionMeta(action=LowLevelAction.TUTEE_QUIZ, is_tutee=True, force_generation=False),
            'tutee_explain': ActionMeta(action=LowLevelAction.TUTEE_EXPLAIN, is_tutee=True, force_generation=False),
            'tutee_fix': ActionMeta(action=LowLevelAction.TUTEE_FIX, is_tutee=True,
                                    force_generation=False),
        }

    def reset(self):
        self.step_count = 0
        self.model.reset(initial_mastery=self.cfg.initial_mastery)
        return self.get_observation()

    def get_observation(self):
        """
        Observation for the RL agents.

        Keep it compact but informative:
          [mastery(num_topics), cfa_ema(num_topics), hint_ema(num_topics), time_ema(num_topics), inc_ema(num_topics), global_mastery, total_steps_norm]
        """
        s = self.model.state
        global_mastery = float(np.mean(s.mastery))
        total_steps_norm = float(s.total_steps) / max(1.0, float(self.max_steps))

        opp_min = max(1, int(self.model.cfg.opp_min))
        opp_norm = np.clip(s.opp.astype(np.float32) / float(opp_min), 0.0, 1.0)

        topic_complete = np.array(
            [1.0 if self.model.is_topic_complete(i) else 0.0 for i in range(self.num_topics)],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                s.mastery.astype(np.float32),
                s.cfa_ema.astype(np.float32),
                s.hint_ema.astype(np.float32),
                s.time_ema.astype(np.float32),
                s.inc_ema.astype(np.float32),

                # NEW blocks:
                opp_norm,  # length = num_topics
                topic_complete,  # length = num_topics

                np.array([global_mastery, total_steps_norm], dtype=np.float32),
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def get_ll_observation(self, topic_id: int) -> List[float]:
        """
           Paper-style *semantics*: LL agents observe learner performance variables for the
           CURRENT topic context only (like Gridlock's current area), not all topics.

           We return one scalar per per-topic block at index=topic_id, plus the global tail.
           This prevents LL from seeing other topics while still being Markov for the current
           tutoring interaction.
        """
        x = np.asarray(self.get_observation(), dtype=np.float32)

        n_blocks = 7  # mastery,cfa,hint,time,inc,opp_norm,topic_complete
        tail = 2  # global_mastery,total_steps_norm
        T = int(self.num_topics)
        expected = n_blocks * T + tail

        if x.shape[0] != expected:
            # fallback if obs layout changes
            return x.tolist()

        t = int(topic_id)
        t = max(0, min(T - 1, t))

        feats = []

        for b in range(n_blocks):
            start = b * T
            feats.append(float(x[start + t]))  # pick the topic-specific scalar

        # append global tail unchanged
        feats.extend([float(x[-2]), float(x[-1])])
        return np.asarray(feats, dtype=np.float32).tolist()

    def _done(self) -> bool:
        # Episode ends if learner achieved the simulator's completion criterion OR we hit a hard cap.
        if self.model.is_done():
            return True
        if self.step_count >= self.max_steps:
            return True
        return False

    def step_tutor(self, topic_id: int, action: str):
        meta = self._tutor_action_map.get(action)
        if meta is None:
            meta = self._tutor_action_map["quiz"]

        _, info = self.model.step(topic_id=topic_id, action_meta=meta)
        self.step_count += float(info.get("step_cost", 1))

        done = self._done()

        base_reward_global = float(info.get("reward_global", info.get("reward", 0.0)))
        base_reward_local = float(info.get("reward_local", base_reward_global))
        step_cost = float(info.get("step_cost", 1))

        step_penalty = self.lambda_step * step_cost
        reward_hl = base_reward_global - step_penalty
        reward_ll = base_reward_local - step_penalty

        info = dict(info)
        info["base_reward_global"] = base_reward_global
        info["base_reward_local"] = base_reward_local
        info["step_penalty"] = float(step_penalty)
        info["reward_hl"] = float(reward_hl)
        info["reward_ll"] = float(reward_ll)

        return self.get_observation(), reward_hl, done, {"mode": "tutor", **info}

        # base_reward = float(info.get("reward", 0.0))
        # step_cost = float(info.get("step_cost", 1))
        #
        # # NEW: penalize step-cost in the reward signal (what RL learns)
        # reward = base_reward - self.lambda_step * step_cost
        # # reward = base_reward
        #
        # # Optional: keep diagnostics in info
        # info = dict(info)
        # info["base_reward"] = base_reward
        # info["step_penalty"] = float(self.lambda_step * step_cost)
        # info["reward_after_penalty"] = float(reward)
        #
        # return self.get_observation(), reward, done, {"mode": "tutor", **info}

    def step_tutee(self, topic_id: int, action: str):
        meta = self._tutee_action_map.get(action)
        if meta is None:
            meta = self._tutee_action_map["tutee_explain"]

        _, info = self.model.step(topic_id=topic_id, action_meta=meta)
        self.step_count += float(info.get("step_cost", 1))

        done = self._done()

        base_reward_global = float(info.get("reward_global", info.get("reward", 0.0)))
        base_reward_local = float(info.get("reward_local", base_reward_global))
        step_cost = float(info.get("step_cost", 1))

        # reward = base_reward - self.lambda_step * step_cost
        # reward = base_reward - self.lambda_step * step_cost
        # reward = base_reward
        reward_hl = base_reward_global - self.lambda_step * step_cost
        reward_ll = base_reward_local - self.lambda_step * step_cost

        info = dict(info)
        info["base_reward_global"] = base_reward_global
        info["base_reward_local"] = base_reward_local
        info["step_penalty"] = float(self.lambda_step * step_cost)
        info["reward_hl"] = float(reward_hl)
        info["reward_ll"] = float(reward_ll)

        return self.get_observation(), reward_hl, done, {"mode": "tutee", **info}


# ----------------------------
# Agent creation (unchanged)
# ----------------------------

def create_agents(
        num_topics: int,
        use_tutee: bool,
        ll_mode: str = "multi",
        experience_sharing: bool = False,
        share_mode: str = "weighted_cka",
        topic_cluster_ids=None,  # NEW
):
    hl_cfg = HighLevelAgentConfig(num_topics=num_topics, use_tutee=use_tutee)
    hl_cfg.device = "cpu"
    high_level_agent = HighLevelAgent(hl_cfg)

    ll_cfg = LowLevelAgentConfig(num_topics=num_topics)
    ll_cfg.device = "cpu"

    # --- Fairness: equalize LL update budget across modes ---
    # Base config assumed tuned for "multi" specialists. For a single shared LL, scale down update frequency
    # so it doesn't get an implicit sample-efficiency advantage.
    ll_cfg.train_every_steps = 20
    ll_cfg.target_update_steps = 5 * ll_cfg.train_every_steps  # paper-ish
    if ll_mode == "single":
        n = max(1, num_topics)
        ll_cfg.train_every_steps = int(ll_cfg.train_every_steps) * n
        ll_cfg.target_update_steps = 5 * ll_cfg.train_every_steps  # keep k=5 rule
        ll_cfg.min_replay_size = int(max(ll_cfg.min_replay_size, ll_cfg.batch_size) * n)
        # ll_cfg.target_update_steps = 200  # keep k=5 rule
        # ll_cfg.buffer_size = max(5_000, int(ll_cfg.buffer_size) // max(1, num_topics))
        # ll_cfg.min_replay_size = int(ll_cfg.min_replay_size) * max(1, num_topics)

    # --- compensate multi-agent data starvation ---
    # In multi-agent, each topic policy sees fewer transitions; increase update frequency.
    if ll_mode == "multi":
        # base_te = int(ll_cfg.train_every_steps)
        # ll_cfg.train_every_steps = max(1, base_te // max(1, num_topics))
        # ll_cfg.target_update_steps = 5 * ll_cfg.train_every_steps  # keep your k=5 rule

        # Sharing warmup expressed in gradient updates; if we update more often, warmup should shrink.
        ll_cfg.share_warmup_updates = max(200, int(ll_cfg.share_warmup_updates) // max(1, num_topics))

    # --- make min_replay_size smaller ONLY for multi-agent (data-starved per-topic buffers) ---
    BASE_MIN_REPLAY = 1000

    if ll_mode == "multi":
        # each topic agent gets fewer transitions; start learning earlier
        ll_cfg.min_replay_size = 200
    else:
        # single shared LL agent sees all topics; can afford larger warmup
        ll_cfg.min_replay_size = BASE_MIN_REPLAY

    # --- sharing config (applies to tutor agents only) ---
    ll_cfg.experience_sharing = bool(experience_sharing) and (ll_mode == "multi")
    ll_cfg.share_mode = share_mode if ll_cfg.experience_sharing else "off"

    if ll_cfg.experience_sharing:
        if ll_cfg.share_mode == "mutual":
            # Conservative, early-only, cluster-local bootstrap
            ll_cfg.share_frac = 0.05
            ll_cfg.max_peers_per_update = 1
            ll_cfg.share_warmup_updates = 200
            ll_cfg.share_stop_updates = 700  # STOP mutual later to avoid harming specialists
            ll_cfg.min_peer_replay_size = 300


        elif ll_cfg.share_mode == "weighted_cka":
            ll_cfg.share_frac = 0.10
            ll_cfg.max_peers_per_update = 2
            ll_cfg.cka_probe_n = 64
            ll_cfg.share_similarity_threshold = 0.75
            ll_cfg.cka_power = 4.0
            ll_cfg.cka_every_updates = 100  # reduce noise + compute cost
            ll_cfg.share_max_weight = 0.60  # cap peer influence (prevents over-trust spikes)
            ll_cfg.share_weight_ema = 0.90  # smooth weights over time (stability)
            ll_cfg.share_warmup_updates = 300
            ll_cfg.share_stop_updates = 1500  # stop late to avoid harming specialists
            ll_cfg.min_peer_replay_size = 1000  # only share from mature peers

    # --- build tutor agents: single vs multi ---
    if ll_mode == "single":
        tutor_agents = [TutorLowLevelAgent(ll_cfg)]  # one shared tutor DQN
    else:
        tutor_agents = [TutorLowLevelAgent(ll_cfg) for _ in range(num_topics)]  # per-topic

    import copy

    tutee_agent = None
    if use_tutee:
        tutee_cfg = copy.copy(ll_cfg)
        tutee_cfg.experience_sharing = False
        tutee_cfg.share_mode = "off"
        tutee_cfg.tutee_ready_quiz = 0.50
        tutee_cfg.tutee_ready_explain = 0.60
        tutee_cfg.tutee_ready_fix = 0.65
        tutee_cfg.tutee_not_ready_penalty = 0.5
        tutee_agent = TuteeLowLevelAgent(tutee_cfg)

    # --- peers only when multi + sharing enabled ---
    if ll_mode == "multi" and ll_cfg.experience_sharing and ll_cfg.share_mode != "off":
        clusters = topic_cluster_ids
        for i, agent in enumerate(tutor_agents):
            if clusters is not None and len(clusters) == len(tutor_agents):
                ci = clusters[i]
                peers = [p for j, p in enumerate(tutor_agents) if j != i and clusters[j] == ci]
            else:
                peers = [p for j, p in enumerate(tutor_agents) if j != i]
            agent.set_peers(peers)
    else:
        for a in tutor_agents:
            a.set_peers([])

    # use_share = ll_cfg.experience_sharing and (ll_cfg.share_mode != "off") and (ll_mode == "multi")
    # shared = ReplayBuffer(ll_cfg.shared_buffer_size) if use_share else None
    # for a in tutor_agents:
    #     a.set_shared_replay(shared)

    return high_level_agent, tutor_agents, tutee_agent


def add_topic(obs, topic_id: int) -> np.ndarray:
    return np.concatenate([obs, np.array([topic_id], dtype=np.float32)])


def summarize_obs(obs, num_topics: int) -> np.ndarray:
    """
    Topic-agnostic compressed view of the learner state.
    Removes topic identity and prevents access to all per-topic raw values.
    """
    x = np.asarray(obs, dtype=np.float32)

    n_blocks = 7
    tail = 2
    expected = n_blocks * num_topics + tail
    if x.shape[0] != expected:
        return x  # fallback

    feats = []
    # For each per-topic block, keep only aggregate stats (mean/min/max)
    for b in range(n_blocks):
        start = b * num_topics
        end = start + num_topics
        v = x[start:end]
        feats.extend([float(v.mean()), float(v.min()), float(v.max())])

    # keep global tail (global_mastery, total_steps_norm)
    feats.extend([float(x[-2]), float(x[-1])])

    return np.asarray(feats, dtype=np.float32)


# def canonicalize_obs(obs, topic_id: int, num_topics: int) -> np.ndarray:
#     """
#     Reorders per-topic blocks so that the active topic_id is always at index 0.
#     This makes experiences from different topics share the same semantics, enabling safe sharing.
#
#     Expected env obs layout (from get_observation()):
#       [mastery, cfa_ema, hint_ema, time_ema, inc_ema, opp_norm, topic_complete] each length=num_topics
#       + [global_mastery, total_steps_norm] length=2
#     """
#     x = np.asarray(obs, dtype=np.float32)
#
#     n_blocks = 7
#     tail = 2
#     expected = n_blocks * num_topics + tail
#     if x.shape[0] != expected:
#         # Fallback: do nothing if obs layout changed
#         return x
#
#     perm = np.array([topic_id] + [i for i in range(num_topics) if i != topic_id], dtype=np.int64)
#
#     blocks = []
#     for b in range(n_blocks):
#         start = b * num_topics
#         end = start + num_topics
#         blocks.append(x[start:end][perm])
#
#     tail_vec = x[n_blocks * num_topics:]
#     return np.concatenate(blocks + [tail_vec], axis=0)


# --- in main.py ---

class FlatAgent:
    """
    True flat baseline:
      One DQN chooses (mode, topic_id, ll_action_str) each step.
    """

    def __init__(self, cfg: LowLevelAgentConfig, num_topics: int):
        self.num_topics = int(num_topics)

        # NOTE: The flat baseline has a large discrete action space (topic x tutor_action).
        # To reduce the "structural handicap" vs HRL (where completed topics are masked at HL),
        # we (1) mask completed topics for action selection, and (2) expand the flat observation
        # with a per-topic grouped representation (feature engineering) to make "choose topic +
        # choose action" easier for a vanilla MLP.

        tutor_actions = build_tutor_actions()
        self._n_tutor_actions = int(len(tutor_actions))
        self.actions = []
        for t in range(self.num_topics):
            for a in tutor_actions:
                self.actions.append(("tutor", t, a))

        self.agent = DQNLowLevelAgent(cfg, actions=[self._encode(x) for x in self.actions])

    # -------- observation shaping (flat-only) --------

    def _expand_obs(self, obs) -> np.ndarray:
        """Expand the raw env observation with a per-topic grouped view."""
        x = np.asarray(obs, dtype=np.float32)

        T = int(self.num_topics)
        n_blocks = 7
        tail = 2
        expected = n_blocks * T + tail
        if x.shape[0] != expected:
            return x

        mastery = x[0:T]
        cfa = x[1*T:2*T]
        hint = x[2*T:3*T]
        t_ema = x[3*T:4*T]
        inc = x[4*T:5*T]
        opp = x[5*T:6*T]
        done = x[6*T:7*T]

        per_topic = np.stack([mastery, cfa, hint, t_ema, inc, opp, done], axis=1).reshape(-1)
        return np.concatenate([x, per_topic.astype(np.float32, copy=False)], axis=0)

    def _valid_action_indices(self, obs) -> List[int]:
        """Return the indices of actions whose topic is NOT completed (masking)."""
        x = np.asarray(obs, dtype=np.float32)
        T = int(self.num_topics)
        n_blocks = 7
        tail = 2
        expected = n_blocks * T + tail

        # Fallback: if obs layout differs, do not mask.
        if x.shape[0] != expected:
            return list(range(len(self.actions)))

        topic_complete = x[6*T:7*T]
        valid_topics = [t for t in range(T) if float(topic_complete[t]) < 0.5]
        if len(valid_topics) == 0:
            valid_topics = list(range(T))

        A = int(self._n_tutor_actions)
        valid = []
        for t in valid_topics:
            base = int(t) * A
            valid.extend(list(range(base, base + A)))
        return valid

    def _encode(self, tpl):
        mode, topic_id, a = tpl
        return f"{mode}|topic={topic_id}|{a}"

    def decode_action(self, idx: int):
        return self.actions[int(idx)]

    def set_epsilon(self, eps: float):
        self.agent.set_epsilon(eps)

    def select_action(self, obs):
        # Completed-topic masking (flat-only)
        valid = self._valid_action_indices(obs)
        if len(valid) == 0:
            valid = list(range(len(self.actions)))

        # Expanded observation (flat-only)
        obs_x = self._expand_obs(obs)
        self.agent._ensure_networks(input_dim=int(len(obs_x)))

        # Epsilon-greedy but over VALID actions only.
        if random.random() < float(self.agent.cfg.epsilon):
            return int(random.choice(valid))

        with torch.no_grad():
            x = torch.as_tensor(obs_x, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
            q = self.agent.policy_net(x).squeeze(0)

            q_np = q.detach().cpu().numpy()
            # hard-mask invalid actions
            mask = np.full_like(q_np, -1e9, dtype=np.float32)
            mask[np.asarray(valid, dtype=np.int64)] = 0.0
            q_np = q_np + mask
            return int(q_np.argmax())

    def update(self, obs, a_idx, r, next_obs, done):
        # Keep training in the same expanded observation space.
        obs_x = self._expand_obs(obs)
        next_obs_x = self._expand_obs(next_obs)
        self.agent.update(obs_x, a_idx, r, next_obs_x, done)



def run_episode_flat(env, flat_agent: FlatAgent, train: bool = True):
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        a_idx = flat_agent.select_action(obs)
        mode, topic_id, ll_action_str = flat_agent.decode_action(a_idx)

        if mode == "tutee":
            next_obs, reward, done, info = env.step_tutee(topic_id, ll_action_str)
        else:
            next_obs, reward, done, info = env.step_tutor(topic_id, ll_action_str)

        if train:
            flat_agent.update(obs, a_idx, reward, next_obs, done)

        total_reward += float(reward)
        steps += float(info.get("step_cost", 1.0))
        obs = next_obs

    return total_reward, steps, info


# ----------------------------
# Episode loop (minimal changes)
# ----------------------------

def run_episode(env, high_level_agent, tutor_agents, tutee_agent, train: bool = True):
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    # Track HL topic streaks (diagnose "stuck on one topic")
    hl_topic_seq: List[int] = []
    step_topic_count = {}
    tutor_hl_count = 0
    tutee_hl_count = 0
    hl_trace = []

    num_topics = env.num_topics
    topic_counts = [0 for _ in range(num_topics)]
    if len(tutor_agents) == 1:
        ll_rewards = [0.0]  # one shared LL agent
    else:
        ll_rewards = [0.0 for _ in range(
            num_topics)]  # one per topic
    tutee_reward_total = 0.0  # optional: total reward earned in tutee mode

    tutor_action_counts = defaultdict(int)
    tutee_action_counts = defaultdict(int)

    tutor_action_names = tutor_agents[0].get_action_meanings()
    for a in tutor_action_names:
        tutor_action_counts[a] = 0

    A = len(tutor_action_names)
    tutor_dm_sum = np.zeros((num_topics, A), dtype=np.float64)
    tutor_dm_count = np.zeros((num_topics, A), dtype=np.int64)

    # NEW: per-topic tutor action counts (even in single-LL; topic_id still exists at HL)
    topic_tutor_action_counts: List[Dict[str, int]] = [
        {a: 0 for a in tutor_action_names} for _ in range(num_topics)]

    if tutee_agent is not None:
        tutee_action_names = tutee_agent.get_action_meanings()
        for a in tutee_action_names:
            tutee_action_counts[a] = 0

    while not done:
        hl_action_idx = high_level_agent.select_action(obs)
        mode, topic_id = high_level_agent.decode_action(hl_action_idx)
        step_topic_count[topic_id] = step_topic_count.get(topic_id, 0) + 1

        topic_counts[topic_id] += 1
        hl_trace.append(f"{mode}_topic_{topic_id}")

        if mode == "tutor":
            tutor_hl_count += 1
            if len(tutor_agents) == 1:
                tutor_agent = tutor_agents[0]  # single shared agent
            else:
                tutor_agent = tutor_agents[topic_id]  # per-topic agent

            # Record topic choice for streak stats
            hl_topic_seq.append(int(topic_id))
            # Always provide the active-topic-first view.
            # - For multi-LL: keeps “in-topic” semantics stable
            # - For single-LL: avoids giving a raw topic_id while keeping it Markov

            tutor_obs = env.get_ll_observation(topic_id)

            ll_action_idx = tutor_agent.select_action(tutor_obs)
            # if len(tutor_agents) > 1:
            #     ref = tutor_agents[0].policy_net
            #     if ref is not None:
            #         for ag in tutor_agents:
            #             ag.set_share_ref_net(ref)
            ll_action_str = tutor_agent.get_action_meanings()[ll_action_idx]
            tutor_action_counts[ll_action_str] += 1
            topic_tutor_action_counts[int(topic_id)][ll_action_str] += 1

            # next_obs, reward, done, info = env.step_tutor(topic_id, ll_action_str)

            m_prev_topic = float(env.model.state.mastery[int(topic_id)])
            next_obs, reward_hl, done, info = env.step_tutor(topic_id, ll_action_str)
            m_new_topic = float(env.model.state.mastery[int(topic_id)])
            dm = m_new_topic - m_prev_topic
            if 0 <= int(topic_id) < num_topics and 0 <= int(ll_action_idx) < A:
                tutor_dm_sum[int(topic_id), int(ll_action_idx)] += float(dm)
                tutor_dm_count[int(topic_id), int(ll_action_idx)] += 1
            reward_ll = float(info.get("reward_ll", reward_hl))



            # if len(tutor_agents) == 1:
            #     # single shared LL tutor needs topic_id to disambiguate
            #     next_tutor_obs = add_topic(next_obs, topic_id)
            # else:
            #     # per-topic LL tutor must NOT include topic_id (keeps sharing “in-topic”)
            #     next_tutor_obs = canonicalize_obs(next_obs, topic_id, num_topics)
            next_tutor_obs = env.get_ll_observation(topic_id)

            # per-agent reward attribution:
            # - multi-agent: reward belongs to that topic’s LL agent
            # - single-agent: reward belongs to the single shared LL agent (index 0)
            ll_idx = 0 if len(tutor_agents) == 1 else int(topic_id)
            ll_rewards[ll_idx] += reward_ll

            if train:
                high_level_agent.update(obs, hl_action_idx, reward_hl, next_obs, done)
                tutor_agent.update(tutor_obs, ll_action_idx, reward_ll, next_tutor_obs, done)

        elif mode == "tutee" and tutee_agent is not None:
            tutee_hl_count += 1

            # tutee_obs = add_topic(obs, topic_id)
            # tutee_obs = np.concatenate([np.asarray(env.get_ll_observation(topic_id), dtype=np.float32),
            #                             np.array([topic_id], dtype=np.float32)]).tolist()

            tutee_obs = env.get_ll_observation(topic_id)

            ll_action_idx = tutee_agent.select_action(tutee_obs)
            ll_action_str = tutee_agent.get_action_meanings()[ll_action_idx]
            tutee_action_counts[ll_action_str] += 1

            next_obs, reward_hl, done, info = env.step_tutee(topic_id, ll_action_str)
            reward_ll = float(info.get("reward_ll", reward_hl))
            # next_tutee_obs = add_topic(next_obs, topic_id)
            # next_tutee_obs = np.concatenate([np.asarray(env.get_ll_observation(topic_id), dtype=np.float32),
            #                             np.array([topic_id], dtype=np.float32)]).tolist()

            next_tutee_obs = env.get_ll_observation(topic_id)

            tutee_reward_total += reward_ll

            if train:
                high_level_agent.update(obs, hl_action_idx, reward_hl, next_obs, done)
                tutee_agent.update(tutee_obs, ll_action_idx, reward_ll, next_tutee_obs, done)

            # next_obs, reward, done, info = env.step_tutee(topic_id, ll_action_str)
            # next_tutee_obs = add_topic(next_obs, topic_id)
            # tutee_reward_total += float(reward)
            #
            # if train:
            #     high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
            #     tutee_agent.update(tutee_obs, ll_action_idx, reward, next_tutee_obs, done)

        else:
            # Safety fallback
            next_obs, reward_hl, done, info = env.step_tutor(topic_id, "no_help")
            if train:
                high_level_agent.update(obs, hl_action_idx, reward_hl, next_obs, done)

        total_reward += float(reward_hl)
        # Steps should reflect the simulator's effective step cost (tutee consumes more budget).
        steps += float(info.get("step_cost", 1))
        obs = next_obs
        # if step_topic_count[topic_id] > 20:
        #     print(f"WARNING: Topic {topic_id} selected {step_topic_count[topic_id]} times in one episode!")

    # env.model.state.teach_boost *= float(env.model.cfg.teach_boost_decay)

    # --- episode-end diagnostics: mastery vector, bottleneck topic, longest HL streak ---
    # Try common locations; adapt if your env stores mastery elsewhere.
    mastery_vec = None

    if hasattr(env, "model") and hasattr(env.model, "state") and hasattr(env.model.state, "mastery"):
        mastery_vec = np.asarray(env.model.state.mastery, dtype=np.float32).copy()
    else:
        mastery_vec = np.zeros(int(getattr(env, "num_topics", 0)), dtype=np.float32)

    if mastery_vec.size > 0:
        bottleneck_topic = int(np.argmin(mastery_vec))
        bottleneck_mastery = float(mastery_vec[bottleneck_topic])
    else:
        bottleneck_topic = -1
        bottleneck_mastery = float("nan")

    # longest consecutive same-topic streak in hl_topic_seq
    longest_streak = 0

    if len(hl_topic_seq) > 0:
        cur = 1
        longest_streak = 1

        for k in range(1, len(hl_topic_seq)):
            if hl_topic_seq[k] == hl_topic_seq[k - 1]:
                cur += 1
            else:
                longest_streak = max(longest_streak, cur)
                cur = 1
        longest_streak = max(longest_streak, cur)

    return (
        total_reward,
        steps,
        topic_counts,
        tutor_action_counts,
        topic_tutor_action_counts,
        tutee_action_counts,
        tutor_hl_count,
        tutee_hl_count,
        hl_trace,
        ll_rewards,  # NEW
        tutee_reward_total,  # NEW
        mastery_vec,  # NEW
        bottleneck_topic,  # NEW
        bottleneck_mastery,  # NEW
        longest_streak,
        tutor_dm_sum,  # NEW
        tutor_dm_count,  # NEW
    )


def _sample_topic_for_case2(env) -> int:
    # simplest: pick among incomplete topics; fallback uniform
    m = env.model.state.mastery
    thresh = getattr(env.model.cfg, "mastery_done_threshold", 0.95)
    candidates = [i for i, mi in enumerate(m) if float(mi) < float(thresh)]
    if not candidates:
        candidates = list(range(env.num_topics))
    return int(np.random.choice(candidates))


# def run_episode_flat(env, flat_agent: FlatAgent, train: bool = True):
#     obs = env.reset()
#     done = False
#     total_reward = 0.0
#     steps = 0
#
#     topic_id = _sample_topic_for_case2(env)
#
#     while not done:
#         obs_t = add_topic(obs, topic_id)  # topic is part of state (OK)
#         a_idx = flat_agent.select_action(obs_t)
#         mode, ll_action_str = flat_agent.decode_action(a_idx)
#
#         if mode == "tutee":
#             next_obs, reward, done, info = env.step_tutee(topic_id, ll_action_str)
#         else:
#             next_obs, reward, done, info = env.step_tutor(topic_id, ll_action_str)
#
#         # keep next state on the same topic for a valid transition
#         next_topic_id = topic_id
#
#         # optionally switch topics only sometimes (reduces handicap, still realistic)
#         # switch if topic complete OR with small probability
#         if env.model.is_topic_complete(topic_id) or random.random() < 0.10:
#             next_topic_id = _sample_topic_for_case2(env)
#
#         next_obs_t = add_topic(next_obs, next_topic_id)
#
#         if train:
#             flat_agent.update(obs_t, a_idx, reward, next_obs_t, done)
#
#         total_reward += float(reward)
#         steps += float(info.get("step_cost", 1))
#         obs = next_obs
#         topic_id = next_topic_id
#
#     return total_reward, steps


# ----------------------------
# Policy-collapse diagnostics
# ----------------------------

def _entropy_from_counts(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ps = [c / total for c in counts.values() if c > 0]
    return float(-sum(p * math.log(p + 1e-12) for p in ps))


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    # p, q are probability vectors summing to 1
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
    kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
    return 0.5 * (kl_pm + kl_qm)


def _collect_state_bank(
        env: "KDDHierEnv",
        high_level_agent: "HighLevelAgent",
        tutor_agents: List["TutorLowLevelAgent"],
        tutee_agent: Optional["TuteeLowLevelAgent"],
        *,
        per_topic_target: int = 2000,
        max_episodes: int = 200,
) -> Dict[int, List[List[float]]]:
    """
    Collect LL observations actually encountered during eval rollouts.
    bank[topic] = list of tutor_obs vectors.
    """
    T = env.num_topics
    bank: Dict[int, List[List[float]]] = {t: [] for t in range(T)}

    # make evaluation deterministic-ish
    old_eps = []
    for ag in tutor_agents:
        old_eps.append(ag.cfg.epsilon)
        ag.set_epsilon(0.0)
    if tutee_agent is not None:
        old_t_eps = tutee_agent.cfg.epsilon
        tutee_agent.set_epsilon(0.0)
    else:
        old_t_eps = None

    try:
        for _ep in range(max_episodes):
            obs = env.reset()
            done = False
            while not done:
                hl_a = high_level_agent.select_action(obs)
                mode, topic_id = high_level_agent.decode_action(hl_a)

                if mode == "tutor":
                    tutor_obs = env.get_ll_observation(topic_id)
                    bank[int(topic_id)].append(tutor_obs)

                    # take an action just to advance env (eval; no learning)
                    if len(tutor_agents) == 1:
                        ag = tutor_agents[0]
                    else:
                        ag = tutor_agents[int(topic_id)]
                    a_idx = ag.select_action(tutor_obs)
                    a_str = ag.get_action_meanings()[a_idx]
                    next_obs, r_hl, done, info = env.step_tutor(topic_id, a_str)

                elif mode == "tutee" and tutee_agent is not None:
                    t_obs = env.get_ll_observation(topic_id)
                    a_idx = tutee_agent.select_action(t_obs)
                    a_str = tutee_agent.get_action_meanings()[a_idx]
                    next_obs, r_hl, done, info = env.step_tutee(topic_id, a_str)

                else:
                    next_obs, r_hl, done, info = env.step_tutor(topic_id, "no_help")

                obs = next_obs

                # stop early if full
                if all(len(bank[t]) >= per_topic_target for t in range(T)):
                    return bank
        return bank
    finally:
        # restore epsilons
        for ag, e in zip(tutor_agents, old_eps):
            ag.set_epsilon(e)
        if tutee_agent is not None and old_t_eps is not None:
            tutee_agent.set_epsilon(old_t_eps)


def _eval_swap_episode(
        env: "KDDHierEnv",
        high_level_agent: "HighLevelAgent",
        tutor_agents: List["TutorLowLevelAgent"],
        tutee_agent: Optional["TuteeLowLevelAgent"],
        topic_to_agent: Optional[List[int]],
) -> (float, float, int):
    """
    Evaluate 1 episode with optional topic->agent permutation (swap test).
    Returns: (total_reward, steps_cost, completed_flag)
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0.0

    while not done:
        hl_a = high_level_agent.select_action(obs)
        mode, topic_id = high_level_agent.decode_action(hl_a)

        if mode == "tutor":
            tutor_obs = env.get_ll_observation(topic_id)

            if len(tutor_agents) == 1:
                ag = tutor_agents[0]
            else:
                if topic_to_agent is None:
                    ag = tutor_agents[int(topic_id)]
                else:
                    ag = tutor_agents[int(topic_to_agent[int(topic_id)])]

            a_idx = ag.select_action(tutor_obs)
            a_str = ag.get_action_meanings()[a_idx]
            next_obs, r_hl, done, info = env.step_tutor(topic_id, a_str)

        elif mode == "tutee" and tutee_agent is not None:
            t_obs = env.get_ll_observation(topic_id)
            a_idx = tutee_agent.select_action(t_obs)
            a_str = tutee_agent.get_action_meanings()[a_idx]
            next_obs, r_hl, done, info = env.step_tutee(topic_id, a_str)

        else:
            next_obs, r_hl, done, info = env.step_tutor(topic_id, "no_help")

        total_reward += float(r_hl)
        steps += float(info.get("step_cost", 1))
        obs = next_obs

    completed = 1 if env.model.is_done() else 0
    return float(total_reward), float(steps), int(completed)


def run_policy_collapse_diagnostics(
        env: "KDDHierEnv",
        high_level_agent: "HighLevelAgent",
        tutor_agents: List["TutorLowLevelAgent"],
        tutee_agent: Optional["TuteeLowLevelAgent"],
        *,
        state_bank_per_topic: int = 2000,
        eval_episodes: int = 200,
        cka_states_per_topic: int = 512,
        seed: int = 0,
) -> None:
    """
    Prints:
      - swap test results
      - action entropy + pairwise JS divergence between action distributions
      - greedy action agreement on fixed state bank
      - pairwise CKA(h1/h2) between agents
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- force greedy eval for LL ----
    old_eps = []
    for ag in tutor_agents:
        old_eps.append(ag.cfg.epsilon)
        ag.set_epsilon(0.0)
    if tutee_agent is not None:
        old_t_eps = tutee_agent.cfg.epsilon
        tutee_agent.set_epsilon(0.0)
    else:
        old_t_eps = None

    try:
        T = env.num_topics
        n_ll = len(tutor_agents)

        print("\n=== Policy Collapse Diagnostics ===")
        print(f"- num_topics: {T}")
        print(f"- n_ll_agents: {n_ll} (1 => shared-LL)")
        print(f"- eval_episodes: {eval_episodes}")
        print(f"- state_bank_per_topic: {state_bank_per_topic}")

        # ----------------------------
        # 1) Swap test (only meaningful if per-topic LL agents exist)
        # ----------------------------
        if n_ll == 1:
            print("\n[Swap test] Skipped (shared single LL agent).")
        else:
            # baseline mapping: topic t -> agent t
            base_rewards, base_steps, base_done = [], [], []
            for _ in range(eval_episodes):
                r, s, d = _eval_swap_episode(env, high_level_agent, tutor_agents, tutee_agent, topic_to_agent=None)
                base_rewards.append(r);
                base_steps.append(s);
                base_done.append(d)

            # random permutation mapping
            perm = list(range(T))
            random.shuffle(perm)
            swap_rewards, swap_steps, swap_done = [], [], []
            for _ in range(eval_episodes):
                r, s, d = _eval_swap_episode(env, high_level_agent, tutor_agents, tutee_agent, topic_to_agent=perm)
                swap_rewards.append(r);
                swap_steps.append(s);
                swap_done.append(d)

            print("\n[Swap test]")
            print(f"- perm mapping (topic->agent): {perm}")
            print(
                f"- baseline: mean_reward={_safe_mean(base_rewards):.4f}  mean_steps={_safe_mean(base_steps):.2f}  completion={_safe_mean(base_done):.3f}")
            print(
                f"- swapped : mean_reward={_safe_mean(swap_rewards):.4f}  mean_steps={_safe_mean(swap_steps):.2f}  completion={_safe_mean(swap_done):.3f}")
            print("  Interpretation: if swapped ≈ baseline, agents are interchangeable (little specialization).")

        # ----------------------------
        # 2) Collect state bank (real visited tutor states)
        # ----------------------------
        bank = _collect_state_bank(
            env, high_level_agent, tutor_agents, tutee_agent,
            per_topic_target=state_bank_per_topic,
            max_episodes=max(200, eval_episodes),
        )
        sizes = {t: len(bank[t]) for t in range(T)}
        print("\n[State bank]")
        print(f"- collected per topic: {sizes}")

        # ----------------------------
        # 3) Action distributions, entropy, JS divergence
        # ----------------------------
        if n_ll == 1:
            # single shared agent: just compute entropy over all topics pooled
            ag = tutor_agents[0]
            counts = Counter()
            for t in range(T):
                for s in bank[t]:
                    a = ag.select_action(s)
                    counts[int(a)] += 1
            ent = _entropy_from_counts(dict(counts))
            print("\n[Actions] shared-LL")
            print(f"- greedy action entropy (pooled over topics): {ent:.4f}  (lower => near-deterministic policy)")
        else:
            # per-topic: each agent evaluated on its own topic bank
            action_probs = []
            entropies = []
            for t in range(T):
                ag = tutor_agents[t]
                counts = np.zeros(ag.num_actions, dtype=np.float64)
                for s in bank[t]:
                    a = ag.select_action(s)
                    counts[int(a)] += 1.0
                probs = counts / max(1.0, counts.sum())
                action_probs.append(probs)
                entropies.append(_entropy_from_counts({i: int(counts[i]) for i in range(len(counts))}))

            # pairwise JS divergence
            js_vals = []
            for i, j in itertools.combinations(range(T), 2):
                js_vals.append(_js_divergence(action_probs[i], action_probs[j]))

            print("\n[Actions] per-topic LL")
            print(f"- mean greedy entropy across topics: {float(np.mean(entropies)):.4f}")
            print(
                f"- min/max greedy entropy across topics: {float(np.min(entropies)):.4f} / {float(np.max(entropies)):.4f}")
            print(f"- mean pairwise JS divergence between agents’ action distributions: {float(np.mean(js_vals)):.6f}")
            print("  Interpretation: very low entropy + JS≈0 => policies collapsed to near-identical behavior.")

        # ----------------------------
        # 4) Greedy action agreement + Q-vector similarity on same states
        # ----------------------------
        # Use topic 0 bank as common reference states (or first non-empty topic)
        ref_topic = next((t for t in range(T) if len(bank[t]) > 0), 0)
        ref_states = bank[ref_topic]
        if len(ref_states) == 0:
            print("\n[Agreement] Skipped (empty state bank).")
        else:
            # sample a fixed subset
            M = min(1024, len(ref_states))
            idx = np.random.choice(len(ref_states), size=M, replace=False)
            S = np.asarray([ref_states[i] for i in idx], dtype=np.float32)
            S_t = torch.from_numpy(S)

            def greedy_actions(agent: "TutorLowLevelAgent") -> np.ndarray:
                agent._ensure_networks(input_dim=S.shape[1])  # type: ignore
                with torch.no_grad():
                    q = agent.policy_net(S_t)  # type: ignore
                    return q.argmax(dim=1).cpu().numpy()

            def q_vectors(agent: "TutorLowLevelAgent") -> np.ndarray:
                agent._ensure_networks(input_dim=S.shape[1])  # type: ignore
                with torch.no_grad():
                    q = agent.policy_net(S_t).cpu().numpy()  # type: ignore
                    return q

            if n_ll == 1:
                acts = greedy_actions(tutor_agents[0])
                # agreement with itself is trivial; report determinism via unique actions
                print("\n[Agreement] shared-LL")
                print(
                    f"- unique greedy actions on ref states: {len(set(map(int, acts)))} / {tutor_agents[0].num_actions}")
            else:
                acts_list = [greedy_actions(tutor_agents[i]) for i in range(T)]
                agree_vals = []
                for i, j in itertools.combinations(range(T), 2):
                    agree = float(np.mean(acts_list[i] == acts_list[j]))
                    agree_vals.append(agree)

                # q cosine similarity (mean over states)
                q_cos_vals = []
                q_list = [q_vectors(tutor_agents[i]) for i in range(T)]
                for i, j in itertools.combinations(range(T), 2):
                    qi = q_list[i].reshape(M, -1)
                    qj = q_list[j].reshape(M, -1)
                    num = np.sum(qi * qj, axis=1)
                    den = (np.linalg.norm(qi, axis=1) * np.linalg.norm(qj, axis=1) + 1e-12)
                    q_cos_vals.append(float(np.mean(num / den)))

                print("\n[Agreement] on same ref-state batch")
                print(f"- mean pairwise greedy-action agreement: {float(np.mean(agree_vals)):.4f}")
                print(f"- mean pairwise Q-vector cosine similarity: {float(np.mean(q_cos_vals)):.4f}")
                print("  Interpretation: agreement→1 and cosine→1 => agents are essentially the same policy.")

        # ----------------------------
        # 5) Representation similarity (CKA on h1/h2)
        # ----------------------------
        if n_ll == 1:
            print("\n[CKA] Skipped (shared-LL; only one network).")
        else:
            # build CKA state batch from multiple topics (concat)
            all_states = []
            per = max(1, int(cka_states_per_topic))
            for t in range(T):
                if len(bank[t]) == 0:
                    continue
                take = min(per, len(bank[t]))
                sel = np.random.choice(len(bank[t]), size=take, replace=False)
                all_states.extend([bank[t][k] for k in sel])

            if len(all_states) < 32:
                print("\n[CKA] Skipped (not enough states).")
            else:
                X = torch.as_tensor(np.asarray(all_states, dtype=np.float32))
                # ensure nets exist
                for ag in tutor_agents:
                    ag._ensure_networks(input_dim=X.shape[1])  # type: ignore

                cka_vals = []
                for i, j in itertools.combinations(range(T), 2):
                    net_i = tutor_agents[i].policy_net  # type: ignore
                    net_j = tutor_agents[j].policy_net  # type: ignore
                    cka = avg_layer_cka(net_i, net_j, X, layers=("h1", "h2"))
                    cka_vals.append(cka)

                print("\n[CKA] pairwise avg CKA(h1,h2)")
                print(
                    f"- mean: {float(np.mean(cka_vals)):.4f}  min/max: {float(np.min(cka_vals)):.4f} / {float(np.max(cka_vals)):.4f}")
                print("  Interpretation: CKA close to 1 => representations collapsed / highly similar.")

        print("\n=== End Diagnostics ===\n")

    finally:
        # restore epsilons
        for ag, e in zip(tutor_agents, old_eps):
            ag.set_epsilon(e)
        if tutee_agent is not None and old_t_eps is not None:
            tutee_agent.set_epsilon(old_t_eps)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, default="education_framework/data/kdd_bundle.joblib")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--log_window", type=int, default=100)
    ap.add_argument("--use_tutee", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=200)

    # ---- sensitivity sweep for mechanistic tutee strength ----
    # Example:
    #   python main.py --bundle ... --episodes 2000 --max_steps 300 \
    #     --sweep_bases 0.005,0.01,0.015,0.02,0.03 --sweep_seeds 0,1,2
    # ap.add_argument(
    #     "--sweep_bases",
    #     type=str,
    #     default="",
    #     help="Comma-separated list of tutee_bonus_base values to sweep (enables sweep mode).",
    # )
    # ap.add_argument(
    #     "--sweep_seeds",
    #     type=str,
    #     default="",
    #     help="Comma-separated list of seeds to run for each sweep setting (default: seed,seed+1,seed+2).",
    # )
    # ap.add_argument(
    #     "--sweep_out_dir",
    #     type=str,
    #     default="runs/sweep",
    #     help="Output directory for sweep runs and summary.csv.",
    # )
    ap.add_argument(
        "--eval_window",
        type=int,
        default=100,
        help="Window size (episodes) for reporting mean reward/steps/mastery at the end.",
    )
    ap.add_argument(
        "--ll_mode",
        type=str,
        default="multi",
        choices=["single", "multi"],
        help="Low-level tutor agent mode: single shared agent vs per-topic agents.",
    )
    ap.add_argument(
        "--experience_sharing",
        action="store_true",
        default=False,
        help="Enable experience sharing between low-level tutor agents (only meaningful in ll_mode=multi).",
    )
    ap.add_argument(
        "--share_mode",
        type=str,
        default="weighted_cka",
        choices=["off", "mutual", "weighted_cka"],
        help="Sharing mode used by low-level tutor agents when experience_sharing is enabled.",
    )
    ap.add_argument(
        "--arch",
        type=str,
        default="hrl",
        choices=["hrl", "flat"],
        help="hrl: High-level + low-level (paper main). flat: single-agent RL baseline (paper exp1).",
    )
    ap.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional extra tag appended to metrics filename (e.g., 'ablation1').",
    )
    ap.add_argument("--diag_policy_collapse", action="store_true", default=False)

    # --- diagnostics (separation debugging) ---
    ap.add_argument(
        "--log_ll_per_topic",
        action="store_true",
        default=False,
        help="Print per-topic tutor action distributions (for HRL runs).",
    )
    ap.add_argument(
        "--log_ll_agreement",
        action="store_true",
        default=False,
        help="Print pairwise greedy-action agreement between tutor LL agents (multi-LL only).",
    )
    ap.add_argument(
        "--agreement_batch",
        type=int,
        default=256,
        help="Batch size for LL agreement diagnostic (states sampled from replay).",
    )
    ap.add_argument(
        "--log_ll_action_effects",
        action="store_true",
        default=False,
        help="Print per-topic mean mastery gain (Δmastery) conditioned on tutor action (HRL runs).",
    )
    args = ap.parse_args()

    # def _metrics_filename(*, seed: int, use_tutee: bool) -> str:
    #     # Keep names filesystem-friendly and stable.
    #     arch = str(args.arch)
    #     ll_mode = str(args.ll_mode) if args.arch == "hrl" else "single"
    #     es = int(bool(args.experience_sharing)) if args.arch == "hrl" else 0
    #     share_mode = str(args.share_mode) if args.arch == "hrl" else "off"
    #     tutee = int(bool(use_tutee))
    #
    #     tag = (args.run_tag or "").strip()
    #     tag_part = f"__tag={tag}" if tag else ""
    #
    #     return (
    #         f"metrics__arch={arch}"
    #         f"__ll={ll_mode}"
    #         f"__es={es}"
    #         f"__share={share_mode}"
    #         f"__tutee={tutee}"
    #         f"__seed={int(seed)}"
    #         f"{tag_part}"
    #         f".csv"
    #     )
    def _metrics_filename(*, seed: int) -> str:
        tag = (args.run_tag or "").strip()
        if not tag:
            tag = "run"  # fallback so you never create a blank name
        return f"{tag}__seed={int(seed)}.csv"

    bundle_path = Path(args.bundle).resolve()
    if not bundle_path.exists():
        # also try relative to project root (the folder containing education_framework)
        root = Path(__file__).resolve().parents[1]
        alt = (root / args.bundle).resolve()
        if alt.exists():
            bundle_path = alt
        else:
            raise FileNotFoundError(f"Bundle not found: {bundle_path} (also tried {alt})")

    bundle = joblib.load(bundle_path)

    # bundle_path = Path(args.bundle)
    # if not bundle_path.exists():
    #     raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    # bundle: KDDModelBundle = joblib.load(str(bundle_path))

    def _parse_csv_list(s: str, cast_fn):
        s = (s or "").strip()
        if not s:
            return []
        return [cast_fn(x.strip()) for x in s.split(",") if x.strip()]

    # ---- Single run mode (original behavior) ----

    def _train_one_run(
            *,
            seed: int,
            use_tutee: bool,
            tutee_bonus_base: Optional[float],
            out_dir: Path,
    ) -> Dict[str, float]:

        def set_global_seed(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        """Train once and write a per-episode metrics.csv. Returns end-window summary metrics."""

        out_dir.mkdir(parents=True, exist_ok=True)

        learner_cfg = KDDLearnerConfig(n_topics=bundle.n_topics)
        if tutee_bonus_base is not None:
            learner_cfg.tutee_bonus_base = float(tutee_bonus_base)
            learner_cfg.tutee_reward_lambda = 0.0

        set_global_seed(seed)

        env = KDDHierEnv(
            bundle=bundle,
            cfg=KDDEnvConfig(num_topics=bundle.n_topics, max_steps=args.max_steps, initial_mastery=0.2),
            learner_cfg=learner_cfg,
            seed=seed,
        )

        num_topics = env.num_topics

        if args.arch == "hrl":
            high_level_agent, tutor_agents, tutee_agent = create_agents(
                num_topics=bundle.n_topics,
                use_tutee=use_tutee,
                ll_mode=args.ll_mode,
                experience_sharing=args.experience_sharing,
                share_mode=args.share_mode,
                topic_cluster_ids=getattr(learner_cfg, "topic_cluster_ids", None),
            )
            print("[LL CFG]", {
                "ll_mode": args.ll_mode,
                "train_every_steps": tutor_agents[0].cfg.train_every_steps,
                "target_update_steps": tutor_agents[0].cfg.target_update_steps,
                "min_replay_size": tutor_agents[0].cfg.min_replay_size,
                "buffer_size": tutor_agents[0].cfg.buffer_size,
            })

            flat_agent = None
        else:
            # flat single-agent RL baseline (no HL/LL decomposition)
            ll_cfg = LowLevelAgentConfig(num_topics=bundle.n_topics)
            ll_cfg.device = "cpu"
            ll_cfg.experience_sharing = False
            ll_cfg.share_mode = "off"
            ll_cfg.min_replay_size = 1000

            flat_agent = FlatAgent(ll_cfg, num_topics=bundle.n_topics)

            high_level_agent, tutor_agents, tutee_agent = None, [], None

        print("=== Training (KDD env) ===")
        print(f"- Architecture:    {args.arch}")
        print(f"- Number of topics: {num_topics}")
        print(f"- Tutee enabled:   {use_tutee}")
        if args.arch == "hrl":
            print(f"- LL mode:         {args.ll_mode}")
            print(f"- Experience share:{args.experience_sharing} ({args.share_mode})")
        if tutee_bonus_base is not None:
            print(f"- Tutee bonus base:{tutee_bonus_base}")
        print(f"- Episodes:        {args.episodes}")
        print(f"- Seed:            {seed}")
        print(f"- Max steps:       {args.max_steps}")
        print(f"- Bundle:          {bundle_path}")
        print(f"- Out dir:         {out_dir}")

        window_rewards: List[float] = []
        window_steps: List[int] = []
        window_mastery: List[float] = []
        window_done: List[int] = []

        if args.arch == "hrl":
            tutor_action_names = tutor_agents[0].get_action_meanings()
            window_tutor_action_counts = {a: 0 for a in tutor_action_names}

            if tutee_agent is not None:
                tutee_action_names = tutee_agent.get_action_meanings()
                window_tutee_action_counts = {a: 0 for a in tutee_action_names}
            else:
                tutee_action_names = []
                window_tutee_action_counts = {}
        else:
            # flat agent has its own action meanings (encoded mode|topic|action)
            tutor_action_names = []
            window_tutor_action_counts = {}
            tutee_action_names = []
            window_tutee_action_counts = {}

        window_topic_counts = [0 for _ in range(num_topics)]
        window_tutor_hl = 0
        window_tutee_hl = 0
        window_topic_tutor_action_counts: List[Dict[str, int]] = [
            {a: 0 for a in tutor_action_names} for _ in range(num_topics)
        ]

        # NEW: per-topic per-action mastery gain stats (tutor mode only)
        if args.arch == "hrl":
            A = len(tutor_action_names)
            window_tutor_dm_sum = np.zeros((num_topics, A), dtype=np.float64)
            window_tutor_dm_count = np.zeros((num_topics, A), dtype=np.int64)
        else:
            window_tutor_dm_sum = None
            window_tutor_dm_count = None

        # NEW: mastery/bottleneck/streak window stats
        window_mastery_sum = np.zeros(num_topics, dtype=np.float64)
        window_bottleneck_counts = np.zeros(num_topics, dtype=np.int64)
        window_bottleneck_mastery_sum = np.zeros(num_topics, dtype=np.float64)
        window_longest_streak_sum = 0.0
        window_eps_count = 0
        eps_start = 0.2
        eps_end = 0.005
        # eps_decay_episodes = max(1, args.episodes)
        eps_decay_episodes = 1000

        rows = []

        for episode in tqdm(range(1, args.episodes + 1), desc=f"Training(seed={seed})"):
            # epsilon schedule
            progress = min(1.0, episode / eps_decay_episodes)
            eps = eps_start + (eps_end - eps_start) * progress
            eps = max(0.00, eps)
            # eps = max(0.02, eps_start + (eps_end - eps_start) * progress)
            if args.arch == "hrl":
                high_level_agent.set_epsilon(eps)
                for ag in tutor_agents:
                    ag.set_epsilon(eps)
                if tutee_agent is not None:
                    tutee_agent.set_epsilon(eps)
            else:
                flat_agent.set_epsilon(eps)
            # ---- reset per-episode sharing counters (MUST be before the episode runs) ----
            if args.arch == "hrl":
                for ag in tutor_agents:
                    ag.reset_share_stats()
            else:
                flat_agent.agent.reset_share_stats()
            if args.arch == "hrl":
                (
                    total_reward,
                    steps,
                    topic_counts,
                    tutor_action_counts,
                    topic_tutor_action_counts,
                    tutee_action_counts,
                    tutor_hl_count,
                    tutee_hl_count,
                    hl_trace,
                    ll_rewards,  # NEW
                    tutee_reward_total,  # NEW
                    ep_mastery_vec,  # NEW
                    ep_bottleneck_topic,  # NEW
                    ep_bottleneck_mastery,  # NEW
                    ep_longest_streak,
                    ep_tutor_dm_sum,  # NEW
                    ep_tutor_dm_count,  # NEW
                ) = run_episode(env, high_level_agent, tutor_agents, tutee_agent, train=True)
                flat_agent_reward = 0.0
            else:
                total_reward, steps, info = run_episode_flat(env, flat_agent, train=True)
                # fill HRL-only stats with zeros/empties for logging consistency
                topic_counts = [0 for _ in range(num_topics)]
                tutor_action_counts = {}
                tutee_action_counts = {}
                topic_tutor_action_counts = []
                tutor_hl_count = 0
                tutee_hl_count = 0
                hl_trace = []
                flat_agent_reward = float(total_reward)
                ll_rewards = [0.0 for _ in range(num_topics)]  # NEW (avoid crash)
                tutee_reward_total = 0.0  # NEW (avoid crash)

            mean_mastery = float(np.mean(env.model.state.mastery))
            min_mastery = float(np.min(env.model.state.mastery))
            completed = 1 if env.model.is_done() else 0

            # ---- collect sharing diagnostics for this episode ----
            if args.arch == "hrl":
                n_ll_agents = len(tutor_agents)
                share_enabled_eff = int(
                    tutor_agents and tutor_agents[0].cfg.experience_sharing and tutor_agents[0].cfg.share_mode != "off")
                share_mode_eff = tutor_agents[0].cfg.share_mode if tutor_agents else "off"
                stats_list = [ag.pop_share_stats() for ag in tutor_agents]
            else:
                n_ll_agents = 1
                share_enabled_eff = 0
                share_mode_eff = "off"
                stats_list = [flat_agent.agent.pop_share_stats()]

            share_attempts = sum(s["share_attempts"] for s in stats_list)
            share_peer_samples = sum(s["peer_samples"] for s in stats_list)
            peer_weight_sum = sum(s["peer_weight_sum"] for s in stats_list)
            eligible_peers_sum = sum(s["eligible_peers_sum"] for s in stats_list)
            selected_peers_sum = sum(s["selected_peers_sum"] for s in stats_list)

            share_mean_peer_weight = (peer_weight_sum / share_peer_samples) if share_peer_samples > 0 else 0.0
            share_eligible_peers_mean = (eligible_peers_sum / share_attempts) if share_attempts > 0 else 0.0
            share_selected_peers_mean = (selected_peers_sum / share_attempts) if share_attempts > 0 else 0.0

            # --- Fig.7 signal: average episodic reward per agent (unbiased) ---
            if args.arch != "hrl":
                # flat baseline: there is exactly 1 learning agent
                avg_agent_reward = float(total_reward)
            else:
                n_tutor_agents = len(ll_rewards)  # 1 if shared-LL, else num_topics
                n_agents = n_tutor_agents + (1 if (use_tutee and tutee_agent is not None) else 0)

                # Sum of per-agent episodic rewards:
                # - tutor LL agents: ll_rewards already holds their episodic totals
                # - tutee agent: tutee_reward_total is its episodic total
                # IMPORTANT: this is NOT the same as env total_reward in general (because HL also learns),
                # but Fig.7 is "reward over agents" => per-agent rewards, not team total.
                agent_reward_sum = float(np.sum(ll_rewards))
                if use_tutee and tutee_agent is not None:
                    agent_reward_sum += float(tutee_reward_total)

                avg_agent_reward = agent_reward_sum / max(1, n_agents)

            # --- ensure ll_rewards always matches header length (num_topics) ---
            ll_rewards_full = list(ll_rewards) + [0.0] * (num_topics - len(ll_rewards))
            ll_rewards_full = ll_rewards_full[:num_topics]  # safety

            # --- tutee reward should be 0 if tutee disabled ---
            tutee_r = float(tutee_reward_total) if use_tutee else 0.0

            # --- effective agent count for Fig.7 ---
            n_ll_eff = 1 if (args.arch == "hrl" and args.ll_mode == "single") else (
                num_topics if args.arch == "hrl" else 1)
            n_agents_eff = n_ll_eff + (1 if use_tutee else 0)

            # --- average reward over all agents (paper Fig.7 signal) ---
            # --- Fig.7 signals (make both explicit so we stop oscillating) ---
            if args.arch != "hrl":
                avg_reward_per_learning_agent = float(total_reward)
                avg_reward_per_topic_slot = float(total_reward)  # flat has no topics separation
            else:
                agent_reward_sum = float(np.sum(ll_rewards))
                if use_tutee and tutee_agent is not None:
                    agent_reward_sum += float(tutee_reward_total)

                n_tutor_agents = len(ll_rewards)  # 1 if shared-LL else num_topics
                n_learning_agents = n_tutor_agents + (1 if (use_tutee and tutee_agent is not None) else 0)

                # B) average per *learning* agent (1 vs 7 causes the ×7 effect)
                avg_reward_per_learning_agent = agent_reward_sum / max(1, n_learning_agents)

                # A) average per *topic slot* (paper-style "over N agents", N=num_topics)
                denom = num_topics + (1 if (use_tutee and tutee_agent is not None) else 0)
                avg_reward_per_topic_slot = agent_reward_sum / max(1, denom)

            rows.append([
                episode, float(total_reward), float(steps), mean_mastery, min_mastery, completed,
                float(flat_agent_reward),
                float(avg_agent_reward),  # NEW: correct Fig.7 signal

                # ---- sharing diagnostics ----
                int(n_ll_agents),
                int(share_enabled_eff),
                str(share_mode_eff),
                int(share_attempts),
                int(share_peer_samples),
                float(share_mean_peer_weight),
                float(share_eligible_peers_mean),
                float(share_selected_peers_mean),

                *[float(x) for x in ll_rewards_full],
                float(tutee_reward_total),
                avg_reward_per_learning_agent,
                avg_reward_per_topic_slot,
            ])
            window_rewards.append(float(total_reward))
            window_steps.append(int(steps))
            window_mastery.append(mean_mastery)
            window_done.append(completed)

            if args.arch == "hrl":
                for i in range(num_topics):
                    window_topic_counts[i] += int(topic_counts[i])
                for a in tutor_action_names:
                    window_tutor_action_counts[a] += int(tutor_action_counts.get(a, 0))
                for a in window_tutee_action_counts:
                    window_tutee_action_counts[a] += int(tutee_action_counts.get(a, 0))

                if topic_tutor_action_counts:
                    for t in range(num_topics):
                        for a in tutor_action_names:
                            window_topic_tutor_action_counts[t][a] += int(
                                topic_tutor_action_counts[t].get(a, 0)
                            )
                # NEW: accumulate tutor Δmastery stats per action/topic
                if window_tutor_dm_sum is not None and ep_tutor_dm_sum is not None:
                    window_tutor_dm_sum += np.asarray(ep_tutor_dm_sum, dtype=np.float64)
                if window_tutor_dm_count is not None and ep_tutor_dm_count is not None:
                    window_tutor_dm_count += np.asarray(ep_tutor_dm_count, dtype=np.int64)

                window_tutor_hl += int(tutor_hl_count)
                window_tutee_hl += int(tutee_hl_count)

                # NEW: aggregate mastery/bottleneck/streak
                window_eps_count += 1

                if ep_mastery_vec is not None and len(ep_mastery_vec) == num_topics:
                    window_mastery_sum += np.asarray(ep_mastery_vec, dtype=np.float64)

                if 0 <= int(ep_bottleneck_topic) < num_topics and np.isfinite(ep_bottleneck_mastery):
                    t = int(ep_bottleneck_topic)
                    window_bottleneck_counts[t] += 1
                    window_bottleneck_mastery_sum[t] += float(ep_bottleneck_mastery)
                elif 0 <= int(ep_bottleneck_topic) < num_topics:
                    t = int(ep_bottleneck_topic)
                    window_bottleneck_counts[t] += 1

                if ep_longest_streak is not None:
                    window_longest_streak_sum += float(ep_longest_streak)

            if episode % args.log_window == 0:
                w = args.log_window
                mean_reward_w = sum(window_rewards) / max(1, len(window_rewards))
                mean_steps_w = sum(window_steps) / max(1, len(window_steps))
                mean_mastery_w = sum(window_mastery) / max(1, len(window_mastery))
                done_rate_w = sum(window_done) / max(1, len(window_done))

                total_topic_choices = sum(window_topic_counts) or 1
                topic_freqs = [c / total_topic_choices for c in window_topic_counts]

                total_tutor_actions = sum(window_tutor_action_counts.values()) or 1
                tutor_action_freqs = {a: window_tutor_action_counts[a] / total_tutor_actions for a in
                                      tutor_action_names}

                print(f"[Episode {episode:4d}]")
                print(f"  Mean reward (last {w}):          {mean_reward_w: .4f}")
                print(f"  Mean steps per episode (cost):   {mean_steps_w: .1f}")
                print(f"  Mean learner mastery:            {mean_mastery_w: .3f}")
                print(f"  Completion rate (last {w}):      {done_rate_w: .2f}")

                if args.arch == "hrl":
                    print("  Topic choice frequencies:")

                    for i, f in enumerate(topic_freqs):
                        print(f"    - Topic {i}: {f * 100:5.1f}% of high-level choices")
                    print("  Tutor action frequencies:")
                    for a, f in tutor_action_freqs.items():
                        print(f"    - {a:20s}: {f * 100:5.1f}% of tutor actions")

                    if use_tutee and window_tutee_action_counts:
                        total_tutee_actions = sum(window_tutee_action_counts.values()) or 1
                        print("  Tutee action frequencies:")
                        for a, c in window_tutee_action_counts.items():
                            f = c / total_tutee_actions
                            print(f"    - {a:20s}: {f * 100:5.1f}% of tutee actions")

                        if args.log_ll_per_topic:
                            print("  Tutor action frequencies per topic (window):")
                            for t in range(num_topics):
                                tot = sum(window_topic_tutor_action_counts[t].values()) or 1
                                parts = []
                                for a in tutor_action_names:
                                    f = window_topic_tutor_action_counts[t][a] / tot
                                    parts.append(f"{a}={f * 100:4.1f}%")
                                print(f"    - Topic {t}: " + " | ".join(parts))

                    # NEW: pairwise LL agreement (multi only)
                    if args.log_ll_agreement:
                        agree = _ll_pairwise_action_agreement(
                            tutor_agents, batch=int(args.agreement_batch)
                        )
                        print(f"  LL greedy-action agreement (pairwise mean): {agree: .3f}")
                    # NEW: show bottleneck + mastery per topic (window)
                    if window_eps_count > 0 and num_topics > 0:
                        mean_mastery_by_topic = window_mastery_sum / max(1, window_eps_count)
                        print("  Mean mastery by topic (window):")
                        for t in range(num_topics):
                            print(f"    - Topic {t}: {mean_mastery_by_topic[t]:.3f}")

                        bn = int(np.argmax(window_bottleneck_counts)) if window_bottleneck_counts.sum() > 0 else -1
                        if bn >= 0:
                            bn_ct = int(window_bottleneck_counts[bn])
                            bn_m = window_bottleneck_mastery_sum[bn] / max(1, bn_ct) if bn_ct > 0 else float("nan")
                            print(
                                f"  Bottleneck topic (most frequent): {bn} (count={bn_ct}/{window_eps_count}, mean_bottleneck_mastery={bn_m:.3f})")

                        print(
                            f"  HL longest same-topic streak (mean over window): {window_longest_streak_sum / max(1, window_eps_count):.1f}")

                    total_hl = window_tutor_hl + window_tutee_hl or 1
                    print(f"  High-level mode frequencies (last {w} episodes):")
                    print(f"    - tutor: {window_tutor_hl / total_hl * 100:5.1f}% of high-level decisions")
                    if use_tutee:
                        print(f"    - tutee: {window_tutee_hl / total_hl * 100:5.1f}% of high-level decisions")

                    if hl_trace:
                        print("  High-level decision sequence (last episode):")
                        print(f"    --> {' --> '.join(hl_trace)}")
                if args.log_ll_action_effects and window_tutor_dm_sum is not None and window_tutor_dm_count is not None:
                    print("  Tutor action effects per topic (mean Δmastery | action) (window):")
                    denom = np.maximum(1, window_tutor_dm_count)
                    mean_dm = window_tutor_dm_sum / denom  # [T, A]

                    for t in range(num_topics):
                        row = mean_dm[t]
                        cnt = window_tutor_dm_count[t]

                        best_ai = int(np.argmax(row)) if row.size > 0 else -1
                        parts = []
                        for ai, a in enumerate(tutor_action_names):
                            m = float(row[ai])
                            n = int(cnt[ai])
                            star = "*" if ai == best_ai else ""
                            parts.append(f"{a}={m:+.4f}{star} (n={n})")

                        # optional: total samples for that topic in the window
                        total_n = int(cnt.sum())
                        print(f"    - Topic {t} (total n={total_n}): " + " | ".join(parts))

                    # summarize similarity of these effect-vectors across topics
                    min_n = 20  # threshold per action
                    V = mean_dm.copy()

                    # mask unreliable dims per topic
                    mask = (window_tutor_dm_count >= min_n).astype(np.float64)
                    V = V * mask

                    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
                    Vn = V / norms
                    S = Vn @ Vn.T
                    mask = ~np.eye(num_topics, dtype=bool)
                    vals = S[mask]
                    if vals.size > 0:
                        print("  Action-effect vector cosine similarity across topics (window):")
                        print(
                            f"    min={float(vals.min()):.3f} mean={float(vals.mean()):.3f} max={float(vals.max()):.3f} std={float(vals.std()):.3f}")

                print()

                # reset window
                window_rewards.clear()
                window_steps.clear()
                window_mastery.clear()
                window_done.clear()
                if window_tutor_dm_sum is not None:
                    window_tutor_dm_sum[:] = 0.0
                if window_tutor_dm_count is not None:
                    window_tutor_dm_count[:] = 0

                if args.arch == "hrl":
                    window_topic_counts[:] = [0 for _ in range(num_topics)]
                    window_tutor_action_counts = {a: 0 for a in tutor_action_names}
                    window_topic_tutor_action_counts = [
                        {a: 0 for a in tutor_action_names} for _ in range(num_topics)
                    ]
                    window_tutor_hl = 0
                    window_tutee_hl = 0
                    window_mastery_sum[:] = 0.0
                    window_bottleneck_counts[:] = 0
                    window_bottleneck_mastery_sum[:] = 0.0
                    window_longest_streak_sum = 0.0
                    window_eps_count = 0
                    if use_tutee and tutee_agent is not None:
                        window_tutee_action_counts = {a: 0 for a in tutee_action_names}


        header = [
            "arch", "ll_mode", "experience_sharing", "share_mode", "use_tutee",
            "episode", "reward", "steps", "mastery_mean", "mastery_min", "completed", "flat_agent_reward",
            "avg_agent_reward",

            # ---- sharing diagnostics ----
            "n_ll_agents",
            "share_enabled_eff",
            "share_mode_eff",
            "share_attempts",
            "share_peer_samples",
            "share_mean_peer_weight",
            "share_eligible_peers_mean",
            "share_selected_peers_mean",
        ]
        header += [f"ll_reward_{i}" for i in range(num_topics)]
        header += ["tutee_reward_total"]
        header += ["avg_reward_per_learning_agent"]
        header += ["avg_reward_per_topic_slot"]
        metrics_path = out_dir / _metrics_filename(seed=seed)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in rows:
                w.writerow(
                    [args.arch, args.ll_mode, int(args.experience_sharing), args.share_mode, int(use_tutee), *row])

        print(f"Wrote metrics: {metrics_path}")

        # end-window summary
        ew = int(max(1, min(args.eval_window, len(rows))))
        tail = rows[-ew:]
        mean_reward_tail = float(np.mean([r[1] for r in tail]))
        mean_steps_tail = float(np.mean([r[2] for r in tail]))
        mean_mastery_tail = float(np.mean([r[3] for r in tail]))
        min_mastery_tail = float(np.mean([r[4] for r in tail]))
        completion_rate_tail = float(np.mean([r[5] for r in tail]))
        # ---- optional: run collapse diagnostics after training ----
        if args.arch == "hrl" and args.diag_policy_collapse:
            run_policy_collapse_diagnostics(
                env=env,
                high_level_agent=high_level_agent,
                tutor_agents=tutor_agents,
                tutee_agent=tutee_agent,
                state_bank_per_topic=2000,
                eval_episodes=200,
                cka_states_per_topic=512,
                seed=seed,
            )
        return {
            "mean_reward_last": mean_reward_tail,
            "mean_steps_last": mean_steps_tail,
            "mean_mastery_last": mean_mastery_tail,
            "min_mastery_last": min_mastery_tail,
            "completion_rate_last": completion_rate_tail,
        }

    # ----------------------------
    # Single-run mode (backwards compatible)
    # ----------------------------
    out_dir = Path("education_framework/runs")
    metrics = _train_one_run(seed=int(args.seed), use_tutee=bool(args.use_tutee), tutee_bonus_base=None,
                             out_dir=out_dir)
    print("Training finished.")
    print("End-window metrics:", metrics)


if __name__ == "__main__":
    main()
