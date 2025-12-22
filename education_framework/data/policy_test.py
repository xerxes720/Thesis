# education_framework/scripts/eval_quality_policies.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import joblib

from education_framework.environment.learner_model import (
    KDDLearnerModel,
    KDDLearnerConfig,
    ActionMeta,
    LowLevelAction,
)

QUALITY_ORDER = {
    "very_bad": 0,
    "bad": 1,
    "neutral": 2,
    "good": 3,
    "very_good": 4,
}

ALL_ACTION_IDS = list(range(8))


@dataclass
class EpisodeResult:
    policy: str
    seed: int
    steps: int
    done: bool
    final_global_mastery: float
    mastered_topics: int


def choose_topic_lowest_mastery(env: KDDLearnerModel) -> int:
    m = env.state.mastery
    return int(np.argmin(m))


def pick_action_by_quality(env: KDDLearnerModel, topic: int, mode: str) -> ActionMeta:
    """
    mode in {"best", "worst", "random"}
    """
    if mode == "random":
        a = int(env.np_rng.integers(0, 8))
        return ActionMeta(action=LowLevelAction(a), is_tutee=(a >= 5), force_generation=False)

    qb = env.bundle.quality_bank
    assert qb is not None, "bundle has no quality_bank"

    x_state = env._state_features(env.state, topic)  # pylint: disable=protected-access

    # collect qualities and optional scores
    items: List[Tuple[int, int, float]] = []
    for a in ALL_ACTION_IDS:
        q = qb.predict_quality(topic_id=topic, action_id=a, x_state=x_state)
        qv = QUALITY_ORDER.get(q, 2)
        s = qb.predict_score_mean(topic_id=topic, action_id=a, x_state=x_state)
        score = float(s) if s is not None else 0.0
        items.append((a, qv, score))

    # rank
    if mode == "best":
        # max quality, then max score
        a, _, _ = max(items, key=lambda t: (t[1], t[2]))
    else:
        # min quality, then min score
        a, _, _ = min(items, key=lambda t: (t[1], t[2]))

    return ActionMeta(action=LowLevelAction(a), is_tutee=(a >= 5), force_generation=False)


def run_policy_episode(bundle, policy_name: str, seed: int, max_steps: int) -> EpisodeResult:
    env = KDDLearnerModel(cfg=KDDLearnerConfig(n_topics=bundle.n_topics), bundle=bundle, seed=seed)
    env.reset(initial_mastery=0.2)

    done = False
    steps = 0

    for t in range(max_steps):
        topic = choose_topic_lowest_mastery(env)

        if policy_name == "BEST":
            am = pick_action_by_quality(env, topic, "best")
        elif policy_name == "WORST":
            am = pick_action_by_quality(env, topic, "worst")
        else:
            am = pick_action_by_quality(env, topic, "random")

        _, info = env.step(topic_id=topic, action_meta=am)
        steps += 1
        done = bool(info.get("done", False))
        if done:
            break

    final_global_mastery = float(np.mean(env.state.mastery))
    mastered_topics = int(np.sum(env.state.mastery >= env.cfg.mastery_threshold))

    return EpisodeResult(
        policy=policy_name,
        seed=seed,
        steps=steps,
        done=done,
        final_global_mastery=final_global_mastery,
        mastered_topics=mastered_topics,
    )


def summarize(results: List[EpisodeResult]) -> None:
    # group by policy
    by: Dict[str, List[EpisodeResult]] = {}
    for r in results:
        by.setdefault(r.policy, []).append(r)

    print("\nRESULT SUMMARY (higher mastery, more mastered_topics, fewer steps is better)\n")
    print(f"{'Policy':<8} {'Episodes':>8} {'Done%':>8} {'AvgSteps':>10} {'AvgMastery':>12} {'AvgMasteredTopics':>16}")
    print("-" * 72)

    for pol in ["BEST", "RANDOM", "WORST"]:
        xs = by.get(pol, [])
        if not xs:
            continue
        done_rate = 100.0 * np.mean([1.0 if x.done else 0.0 for x in xs])
        avg_steps = float(np.mean([x.steps for x in xs]))
        avg_mastery = float(np.mean([x.final_global_mastery for x in xs]))
        avg_mastered = float(np.mean([x.mastered_topics for x in xs]))
        print(f"{pol:<8} {len(xs):>8d} {done_rate:>7.1f}% {avg_steps:>10.1f} {avg_mastery:>12.4f} {avg_mastered:>16.2f}")

    # simple “effect sizes” for thesis narration
    b = by.get("BEST", [])
    w = by.get("WORST", [])
    if b and w:
        diff_mastery = float(np.mean([x.final_global_mastery for x in b]) - np.mean([x.final_global_mastery for x in w]))
        diff_mastered = float(np.mean([x.mastered_topics for x in b]) - np.mean([x.mastered_topics for x in w]))
        print("\nBEST - WORST differences:")
        print(f"  Δ global mastery: {diff_mastery:.4f}")
        print(f"  Δ mastered topics: {diff_mastered:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="kdd_bundle.joblib")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)

    results: List[EpisodeResult] = []
    for i in range(args.episodes):
        seed = args.seed0 + i
        results.append(run_policy_episode(bundle, "BEST", seed, args.max_steps))
        results.append(run_policy_episode(bundle, "RANDOM", seed, args.max_steps))
        results.append(run_policy_episode(bundle, "WORST", seed, args.max_steps))

    summarize(results)


if __name__ == "__main__":
    main()
