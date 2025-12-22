# education_framework/scripts/test_kdd_bundle.py
from __future__ import annotations
import joblib

b = joblib.load("kdd_bundle.joblib")

print("n_topics:", b.n_topics)
print("has quality_bank:", hasattr(b, "quality_bank") and b.quality_bank is not None)
print("has transition_models:", hasattr(b, "transition_models"))
print("betas:", b.quality_bank.mastery_params)
print("topics in bank:", sorted(b.quality_bank.bank.keys()))

import argparse
import math
from collections import Counter, defaultdict

import joblib
import numpy as np

from education_framework.environment.learner_model import (
    KDDLearnerModel,
    KDDLearnerConfig,
    ActionMeta,
    LowLevelAction,
)

QUALITY_LEVELS = ["very_bad", "bad", "neutral", "good", "very_good"]


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_bundle_integrity(bundle) -> None:
    print_header("1) Bundle integrity")
    assert_true(bundle.n_topics == 7, f"Expected n_topics=7, got {bundle.n_topics}")
    assert_true(hasattr(bundle, "quality_bank") and bundle.quality_bank is not None, "Missing quality_bank")
    assert_true(len(bundle.quality_bank.bank) == 7,
                f"Expected 7 topic models in quality bank, got {len(bundle.quality_bank.bank)}")
    assert_true(len(bundle.response_models) >= 5,
                f"Expected at least 5 response models, got {len(bundle.response_models)}")

    # optional: aux models
    if bundle.hints_models is not None:
        assert_true(isinstance(bundle.hints_models, dict), "hints_models should be a dict or None")
    if bundle.time_models is not None:
        assert_true(isinstance(bundle.time_models, dict), "time_models should be a dict or None")
    if bundle.inc_models is not None:
        assert_true(isinstance(bundle.inc_models, dict), "inc_models should be a dict or None")

    betas = bundle.quality_bank.mastery_params
    print("Mastery betas:", betas)
    assert_true(0.0 < betas.beta_good <= 0.25, "beta_good out of range")
    assert_true(0.0 < betas.beta_very_good <= 0.25, "beta_very_good out of range")
    assert_true(0.0 < betas.beta_bad <= 0.25, "beta_bad out of range")
    assert_true(0.0 < betas.beta_very_bad <= 0.25, "beta_very_bad out of range")
    print("OK")


def test_leaf_tables_non_degenerate(bundle) -> None:
    print_header("2) Leaf tables sanity (action influence exists)")
    qb = bundle.quality_bank

    total_leaves = 0
    leaves_with_variation = 0
    per_topic_stats = []

    for k, model in sorted(qb.bank.items()):
        leaf_map = model.leaf_to_action_quality
        assert_true(len(leaf_map) > 0, f"Topic {k}: no leaves stored")

        topic_leaves = len(leaf_map)
        topic_var = 0
        for leaf, amap in leaf_map.items():
            total_leaves += 1
            quals = [amap.get(a, "neutral") for a in range(8)]
            uniq = set(quals)
            if len(uniq) > 1:
                leaves_with_variation += 1
                topic_var += 1

            # check all qualities are in allowed set
            for q in uniq:
                assert_true(q in QUALITY_LEVELS, f"Topic {k}, leaf {leaf}: invalid quality {q}")

        per_topic_stats.append((k, topic_leaves, topic_var))

    print("Per-topic leaf variation (topic, leaves, leaves_with_action_variation):")
    for row in per_topic_stats:
        print("  ", row)

    frac = leaves_with_variation / max(1, total_leaves)
    print(f"Leaves with action variation: {leaves_with_variation}/{total_leaves} = {frac:.3f}")
    assert_true(frac >= 0.50, "Too many leaves have identical quality for all actions (degenerate mapping).")
    print("OK")


def test_quality_correlates_with_score(bundle) -> None:
    """
    Validity check using the stored leaf_to_action_score_mean (mean delta_mastery proxy).
    Expect monotonic relationship: very_good > good > neutral > bad > very_bad on average.
    """
    print_header("3) Validity: quality should correlate with higher mean score")
    qb = bundle.quality_bank

    scores_by_q = defaultdict(list)

    for k, model in sorted(qb.bank.items()):
        for leaf, amap in model.leaf_to_action_quality.items():
            smap = model.leaf_to_action_score_mean.get(leaf, {})
            for a in range(8):
                q = amap.get(a, model.default_quality)
                s = smap.get(a, None)
                if s is None:
                    continue
                scores_by_q[q].append(float(s))

    for q in QUALITY_LEVELS:
        arr = np.asarray(scores_by_q.get(q, []), dtype=np.float32)
        print(f"{q:9s}  n={arr.size:6d}  mean={arr.mean() if arr.size else float('nan'):.6f}")

    # basic monotonic checks if enough samples exist
    means = {q: (np.mean(scores_by_q[q]) if scores_by_q[q] else None) for q in QUALITY_LEVELS}
    if means["very_good"] is not None and means["good"] is not None:
        assert_true(means["very_good"] >= means["good"] - 1e-9, "Expected very_good >= good in mean score")
    if means["good"] is not None and means["neutral"] is not None:
        assert_true(means["good"] >= means["neutral"] - 1e-9, "Expected good >= neutral in mean score")
    if means["neutral"] is not None and means["bad"] is not None:
        assert_true(means["neutral"] >= means["bad"] - 1e-9, "Expected neutral >= bad in mean score")

    print("OK (if monotonic constraints passed; if they fail, your leaf ranking/score is inconsistent)")


def test_mastery_update_direction(bundle) -> None:
    print_header("4) Mastery update direction check")
    cfg = KDDLearnerConfig(n_topics=bundle.n_topics)
    env = KDDLearnerModel(cfg=cfg, bundle=bundle, seed=0)
    env.reset(initial_mastery=0.3)

    # pick topic 0
    t = 0
    m0 = float(env.state.mastery[t])

    # Force quality updates directly by calling internal method (safe for test)
    # We only verify direction / bounds.
    def apply(q):
        env2 = env  # same env
        before = float(env2.state.mastery[t])
        env2._apply_mastery_quality_update(t, q)  # pylint: disable=protected-access
        after = float(env2.state.mastery[t])
        env2.state.mastery[t] = before  # reset
        return before, after

    for q in ["very_good", "good", "neutral", "bad", "very_bad"]:
        b, a = apply(q)
        print(f"{q:9s}  {b:.4f} -> {a:.4f}")

    b, a = apply("good")
    assert_true(a >= b, "good should not decrease mastery")
    b, a = apply("very_good")
    assert_true(a >= b, "very_good should not decrease mastery")
    b, a = apply("bad")
    assert_true(a <= b + 1e-9, "bad should not increase mastery")
    b, a = apply("very_bad")
    assert_true(a <= b + 1e-9, "very_bad should not increase mastery")
    print("OK")


def test_step_outputs_are_finite(bundle, n_steps=200) -> None:
    print_header("5) Runtime step sanity (finite outcomes, mastery within [0,1])")
    cfg = KDDLearnerConfig(n_topics=bundle.n_topics)
    env = KDDLearnerModel(cfg=cfg, bundle=bundle, seed=123)
    env.reset(initial_mastery=0.2)

    actions = [
        ActionMeta(LowLevelAction.TUTOR_QUIZ, False, False),
        ActionMeta(LowLevelAction.TUTOR_HINT, False, False),
        ActionMeta(LowLevelAction.TUTOR_WORKED_EXAMPLE, False, False),
        ActionMeta(LowLevelAction.TUTOR_REMEDIATION, False, False),
        ActionMeta(LowLevelAction.TUTOR_REVIEW, False, False),
        ActionMeta(LowLevelAction.TUTEE_QUIZ, True, False),
        ActionMeta(LowLevelAction.TUTEE_EXPLAIN, True, False),
        ActionMeta(LowLevelAction.TUTEE_FIX, True, False),
    ]

    q_counts = Counter()
    for i in range(n_steps):
        topic = i % bundle.n_topics
        am = actions[i % len(actions)]
        s, info = env.step(topic_id=topic, action_meta=am)
        q_counts[str(info.get("quality", "neutral"))] += 1

        # finite checks
        for k in ["p_correct", "duration", "reward"]:
            v = float(info.get(k, 0.0))
            assert_true(math.isfinite(v), f"{k} is not finite: {v}")
        for k in ["cfa", "hints", "incorrects"]:
            assert_true(isinstance(info.get(k), int), f"{k} is not int")

        # bounds checks
        m = float(env.state.mastery[topic])
        assert_true(0.0 <= m <= 1.0, f"mastery out of bounds: {m}")

    print("Quality distribution over steps:", dict(q_counts))
    assert_true(sum(q_counts.values()) == n_steps, "step count mismatch")
    print("OK")


def test_action_changes_quality_same_state(bundle, trials_per_topic=200) -> None:
    """
    Core test: for the same learner state, do different actions yield different quality?
    We do this by querying the quality_bank directly at the current state features.
    """
    print_header("6) Core test: actions change quality for same state")
    qb = bundle.quality_bank

    env = KDDLearnerModel(cfg=KDDLearnerConfig(n_topics=bundle.n_topics), bundle=bundle, seed=42)
    env.reset(initial_mastery=0.2)

    varied = 0
    total = 0

    # generate a bunch of random states by running random steps, then probe quality
    actions = [
        ActionMeta(LowLevelAction.TUTOR_QUIZ, False, False),
        ActionMeta(LowLevelAction.TUTOR_HINT, False, False),
        ActionMeta(LowLevelAction.TUTOR_WORKED_EXAMPLE, False, False),
        ActionMeta(LowLevelAction.TUTOR_REMEDIATION, False, False),
        ActionMeta(LowLevelAction.TUTOR_REVIEW, False, False),
        ActionMeta(LowLevelAction.TUTEE_QUIZ, True, False),
        ActionMeta(LowLevelAction.TUTEE_EXPLAIN, True, False),
        ActionMeta(LowLevelAction.TUTEE_FIX, True, False),
    ]

    for _ in range(trials_per_topic * bundle.n_topics):
        topic = env.np_rng.integers(0, bundle.n_topics)
        # advance env a bit
        env.step(topic_id=int(topic), action_meta=actions[int(env.np_rng.integers(0, len(actions)))])
        x_state = env._state_features(env.state, int(topic))  # pylint: disable=protected-access

        # query quality for all actions at same state
        quals = [qb.predict_quality(int(topic), a, x_state) for a in range(8)]
        total += 1
        if len(set(quals)) > 1:
            varied += 1

    frac = varied / max(1, total)
    print(f"States where action changes quality: {varied}/{total} = {frac:.3f}")
    assert_true(frac >= 0.70, "Action does not change quality often enough; mapping may be degenerate.")
    print("OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="kdd_bundle.joblib")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)

    test_bundle_integrity(bundle)
    test_leaf_tables_non_degenerate(bundle)
    test_quality_correlates_with_score(bundle)
    test_mastery_update_direction(bundle)
    test_step_outputs_are_finite(bundle, n_steps=args.steps)
    test_action_changes_quality_same_state(bundle, trials_per_topic=120)

    print_header("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
