# education_framework/main.py

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
from tqdm import tqdm

# --- Ensure imports work whether you run:
#   python -m education_framework.main
# or:
#   python education_framework/main.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from education_framework.agents.high_level_agent import HighLevelAgent, HighLevelAgentConfig
from education_framework.agents.low_level_agents import (
    TutorLowLevelAgent,
    TuteeLowLevelAgent,
    LowLevelAgentConfig,
)

from education_framework.environment.learner_model import (
    KDDLearnerModel,
    KDDLearnerConfig,
    KDDModelBundle,
    ActionMeta,
    LowLevelAction,
)


# ----------------------------
# Environment wrapper (KDD)
# ----------------------------

@dataclass
class KDDEnvConfig:
    num_topics: int = 8
    max_steps: int = 500
    initial_mastery: float = 0.2


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

        self.model = KDDLearnerModel(
            cfg=learner_cfg or KDDLearnerConfig(n_topics=self.num_topics),
            bundle=bundle,
            seed=seed,
        )
        self.step_count = 0

        # String -> KDD action mapping (keep simple and stable)
        self._tutor_action_map: Dict[str, ActionMeta] = {
            # tutor actions from build_tutor_actions() :contentReference[oaicite:2]{index=2}
            "quiz": ActionMeta(action=LowLevelAction.INDEPENDENT_PRACTICE, is_tutee=False, force_generation=False),
            "hint": ActionMeta(action=LowLevelAction.SCAFFOLDED_PRACTICE, is_tutee=False, force_generation=False),
            "worked_example": ActionMeta(action=LowLevelAction.WORKED_EXAMPLE_THEN_PRACTICE, is_tutee=False, force_generation=False),
            # safety fallback used by old main
            "no_help": ActionMeta(action=LowLevelAction.INDEPENDENT_PRACTICE, is_tutee=False, force_generation=False),
        }

        self._tutee_action_map: Dict[str, ActionMeta] = {
            # tutee actions from build_tutee_actions() :contentReference[oaicite:3]{index=3}
            "ask_explanation": ActionMeta(action=LowLevelAction.TEACH_BACK_OR_DIAGNOSE, is_tutee=True, force_generation=True),
            "ask_summary": ActionMeta(action=LowLevelAction.TEACH_BACK_OR_DIAGNOSE, is_tutee=True, force_generation=True),
            "ask_worked_example": ActionMeta(action=LowLevelAction.WORKED_EXAMPLE_THEN_PRACTICE, is_tutee=True, force_generation=True),
            "show_mistake_and_ask_fix": ActionMeta(action=LowLevelAction.ERROR_FOCUSED_REMEDIATION, is_tutee=True, force_generation=True),
        }

    def reset(self) -> List[float]:
        self.step_count = 0
        self.model.reset(initial_mastery=self.cfg.initial_mastery)
        return self.get_observation()

    def get_observation(self) -> List[float]:
        """
        Observation for the RL agents.

        Keep it compact but informative:
          [mastery(8), cfa_ema(8), hint_ema(8), time_ema(8), inc_ema(8), global_mastery, total_steps_norm]
        """
        s = self.model.state
        global_mastery = float(np.mean(s.mastery))
        total_steps_norm = float(s.total_steps) / max(1.0, float(self.max_steps))

        obs = np.concatenate(
            [
                s.mastery.astype(np.float32),
                s.cfa_ema.astype(np.float32),
                s.hint_ema.astype(np.float32),
                s.time_ema.astype(np.float32),
                s.inc_ema.astype(np.float32),
                np.array([global_mastery, total_steps_norm], dtype=np.float32),
            ],
            axis=0,
        )
        return obs.astype(np.float32).tolist()

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
        self.step_count += 1

        done = self._done()
        # If time-out without completion, optionally damp reward (keeps training stable)
        reward = float(info.get("reward", 0.0))
        if (self.step_count >= self.max_steps) and (not self.model.is_done()):
            reward -= 0.25

        return self.get_observation(), reward, done, {"mode": "tutor", **info}

    def step_tutee(self, topic_id: int, action: str):
        meta = self._tutee_action_map.get(action)
        if meta is None:
            meta = self._tutee_action_map["ask_explanation"]

        _, info = self.model.step(topic_id=topic_id, action_meta=meta)
        self.step_count += 1

        done = self._done()
        reward = float(info.get("reward", 0.0))
        if (self.step_count >= self.max_steps) and (not self.model.is_done()):
            reward -= 0.25

        return self.get_observation(), reward, done, {"mode": "tutee", **info}


# ----------------------------
# Agent creation (unchanged)
# ----------------------------

def create_agents(num_topics: int, use_tutee: bool = True):
    """
    Create and return:
      - one high-level agent
      - a list of low-level tutor agents (one per topic)
      - one low-level tutee agent (or None if use_tutee=False)
    """
    hl_cfg = HighLevelAgentConfig(num_topics=num_topics, use_tutee=use_tutee)
    hl_cfg.device = "cpu"
    high_level_agent = HighLevelAgent(hl_cfg)

    ll_cfg = LowLevelAgentConfig(num_topics=num_topics)
    ll_cfg.device = "cpu"
    tutor_agents = [TutorLowLevelAgent(ll_cfg) for _ in range(num_topics)]
    tutee_agent = TuteeLowLevelAgent(ll_cfg) if use_tutee else None

    # experience sharing among tutor agents (as in your current main) :contentReference[oaicite:4]{index=4}
    for i, agent in enumerate(tutor_agents):
        peers = [p for j, p in enumerate(tutor_agents) if j != i]
        agent.set_peers(peers)

    return high_level_agent, tutor_agents, tutee_agent


def add_topic(obs, topic_id: int) -> np.ndarray:
    obs_np = np.asarray(obs, dtype=np.float32)
    return np.concatenate([obs_np, np.array([topic_id], dtype=np.float32)])


# ----------------------------
# Episode loop (minimal changes)
# ----------------------------

def run_episode(env, high_level_agent, tutor_agents, tutee_agent, train: bool = True):
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    tutor_hl_count = 0
    tutee_hl_count = 0
    hl_trace = []

    num_topics = env.num_topics
    topic_counts = [0 for _ in range(num_topics)]

    tutor_action_counts = defaultdict(int)
    tutee_action_counts = defaultdict(int)

    tutor_action_names = tutor_agents[0].get_action_meanings()
    for a in tutor_action_names:
        tutor_action_counts[a] = 0

    if tutee_agent is not None:
        tutee_action_names = tutee_agent.get_action_meanings()
        for a in tutee_action_names:
            tutee_action_counts[a] = 0

    while not done:
        hl_action_idx = high_level_agent.select_action(obs)
        mode, topic_id = high_level_agent.decode_action(hl_action_idx)
        topic_counts[topic_id] += 1
        hl_trace.append(f"{mode}_topic_{topic_id}")

        if mode == "tutor":
            tutor_hl_count += 1
            tutor_agent = tutor_agents[topic_id]

            tutor_obs = add_topic(obs, topic_id)
            ll_action_idx = tutor_agent.select_action(tutor_obs)
            ll_action_str = tutor_agent.get_action_meanings()[ll_action_idx]
            tutor_action_counts[ll_action_str] += 1

            next_obs, reward, done, _ = env.step_tutor(topic_id, ll_action_str)
            next_tutor_obs = add_topic(next_obs, topic_id)

            if train:
                high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
                tutor_agent.update(tutor_obs, ll_action_idx, reward, next_tutor_obs, done)

        elif mode == "tutee" and tutee_agent is not None:
            tutee_hl_count += 1

            tutee_obs = add_topic(obs, topic_id)
            ll_action_idx = tutee_agent.select_action(tutee_obs)
            ll_action_str = tutee_agent.get_action_meanings()[ll_action_idx]
            tutee_action_counts[ll_action_str] += 1

            next_obs, reward, done, _ = env.step_tutee(topic_id, ll_action_str)
            next_tutee_obs = add_topic(next_obs, topic_id)

            if train:
                high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
                tutee_agent.update(tutee_obs, ll_action_idx, reward, next_tutee_obs, done)

        else:
            # Safety fallback
            next_obs, reward, done, _ = env.step_tutor(topic_id, "no_help")
            if train:
                high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)

        total_reward += float(reward)
        steps += 1
        obs = next_obs

    return (
        total_reward,
        steps,
        topic_counts,
        tutor_action_counts,
        tutee_action_counts,
        tutor_hl_count,
        tutee_hl_count,
        hl_trace,
    )


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, default="education_framework/models/kdd_bundle.joblib")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--log_window", type=int, default=100)
    ap.add_argument("--use_tutee", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=500)
    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    bundle: KDDModelBundle = joblib.load(str(bundle_path))

    env = KDDHierEnv(
        bundle=bundle,
        cfg=KDDEnvConfig(num_topics=bundle.n_topics, max_steps=args.max_steps, initial_mastery=0.2),
        learner_cfg=KDDLearnerConfig(n_topics=bundle.n_topics),
        seed=args.seed,
    )

    num_topics = env.num_topics
    high_level_agent, tutor_agents, tutee_agent = create_agents(num_topics=num_topics, use_tutee=args.use_tutee)

    print("=== Training hierarchical RL tutor (KDD env) ===")
    print(f"- Number of topics: {num_topics}")
    print(f"- Tutee enabled:   {args.use_tutee}")
    print(f"- Episodes:        {args.episodes}")
    print(f"- Bundle:          {bundle_path}")

    window_rewards: List[float] = []
    window_steps: List[int] = []
    window_topic_counts = [0 for _ in range(num_topics)]
    window_tutor_hl = 0
    window_tutee_hl = 0

    tutor_action_names = tutor_agents[0].get_action_meanings()
    window_tutor_action_counts = {a: 0 for a in tutor_action_names}

    if tutee_agent is not None:
        tutee_action_names = tutee_agent.get_action_meanings()
        window_tutee_action_counts = {a: 0 for a in tutee_action_names}
    else:
        window_tutee_action_counts = {}

    eps_start = 0.2
    eps_end = 0.0
    eps_decay_episodes = max(1, args.episodes)

    for episode in tqdm(range(1, args.episodes + 1), desc="Training"):
        (
            total_reward,
            steps,
            topic_counts,
            tutor_action_counts,
            tutee_action_counts,
            tutor_hl_count,
            tutee_hl_count,
            hl_trace,
        ) = run_episode(env, high_level_agent, tutor_agents, tutee_agent, train=True)

        window_rewards.append(float(total_reward))
        window_steps.append(int(steps))
        for i in range(num_topics):
            window_topic_counts[i] += int(topic_counts[i])
        for a in tutor_action_names:
            window_tutor_action_counts[a] += int(tutor_action_counts[a])
        for a in window_tutee_action_counts:
            window_tutee_action_counts[a] += int(tutee_action_counts.get(a, 0))

        window_tutor_hl += int(tutor_hl_count)
        window_tutee_hl += int(tutee_hl_count)

        # log
        if episode % args.log_window == 0:
            w = args.log_window
            mean_reward = sum(window_rewards) / max(1, len(window_rewards))
            mean_steps = sum(window_steps) / max(1, len(window_steps))

            # KDD env mastery: directly from simulator state
            mean_mastery = float(np.mean(env.model.state.mastery))

            total_topic_choices = sum(window_topic_counts) or 1
            topic_freqs = [c / total_topic_choices for c in window_topic_counts]

            total_tutor_actions = sum(window_tutor_action_counts.values()) or 1
            tutor_action_freqs = {a: window_tutor_action_counts[a] / total_tutor_actions for a in tutor_action_names}

            print(f"[Episode {episode:4d}]")
            print(f"  Mean reward (last {w}):          {mean_reward: .4f}")
            print(f"  Mean steps per episode:         {mean_steps: .1f}")
            print(f"  Mean learner mastery:           {mean_mastery: .3f}")
            print("  Topic choice frequencies:")
            for i, f in enumerate(topic_freqs):
                print(f"    - Topic {i}: {f * 100:5.1f}% of high-level choices")

            print("  Tutor action frequencies:")
            for a, f in tutor_action_freqs.items():
                print(f"    - {a:20s}: {f * 100:5.1f}% of tutor actions")

            if args.use_tutee and window_tutee_action_counts:
                total_tutee_actions = sum(window_tutee_action_counts.values()) or 1
                print("  Tutee action frequencies:")
                for a, c in window_tutee_action_counts.items():
                    f = c / total_tutee_actions
                    print(f"    - {a:20s}: {f * 100:5.1f}% of tutee actions")

            total_hl = window_tutor_hl + window_tutee_hl or 1
            print(f"  High-level mode frequencies (last {w} episodes):")
            print(f"    - tutor: {window_tutor_hl / total_hl * 100:5.1f}% of high-level decisions")
            if args.use_tutee:
                print(f"    - tutee: {window_tutee_hl / total_hl * 100:5.1f}% of high-level decisions")

            if hl_trace:
                print("  High-level decision sequence (last episode):")
                print(f"    --> {' --> '.join(hl_trace)}")
            print()

            # reset window
            window_rewards.clear()
            window_steps.clear()
            window_topic_counts = [0 for _ in range(num_topics)]
            window_tutor_action_counts = {a: 0 for a in tutor_action_names}
            window_tutor_hl = 0
            window_tutee_hl = 0
            if args.use_tutee and tutee_agent is not None:
                window_tutee_action_counts = {a: 0 for a in tutee_action_names}

            # epsilon schedule
            progress = min(1.0, episode / eps_decay_episodes)
            eps = eps_start + (eps_end - eps_start) * progress
            high_level_agent.set_epsilon(eps)
            for ag in tutor_agents:
                ag.set_epsilon(eps)
            if tutee_agent is not None:
                tutee_agent.set_epsilon(eps)

    print("Training finished.")


if __name__ == "__main__":
    main()
