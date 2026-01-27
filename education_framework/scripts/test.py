# education_framework/scripts/test.py
from __future__ import annotations

"""Expanded test suite for the KDD-based learner simulator and QualityTreeBank.

Run:
  python education_framework/scripts/test.py --bundle education_framework/data/kdd_bundle.joblib --steps 200

The suite is intentionally committee-friendly:
- checks invariants (bounds, monotonicity, non-degeneracy)
- avoids brittle exact-value assertions
"""

import argparse
import math
from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple

import joblib
import numpy as np

from education_framework.environment.learner_model import (
    KDDLearnerModel,
    KDDLearnerConfig,
    ActionMeta,
    LowLevelAction,
    KDDActionSchema,
)

# Optional: agent tests
try:
    from education_framework.agents.high_level_agent import HighLevelAgent, HighLevelAgentConfig
    from education_framework.agents.low_level_agents import (
        TutorLowLevelAgent,
        TuteeLowLevelAgent,
        LowLevelAgentConfig,
        linear_cka,
    )
    import torch
    _HAS_AGENTS = True
except Exception:
    _HAS_AGENTS = False


QUALITY_LEVELS = ["very_bad", "bad", "neutral", "good", "very_good"]
QUALITY_ORDER = {q: i for i, q in enumerate(QUALITY_LEVELS)}


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ----------------------------
# 1) Bundle / bank structural tests
# ----------------------------

# Verify the trained KDDModelBundle is structurally valid and internally consistent.
def test_bundle_integrity(bundle) -> None:
    print_header("1) Bundle integrity")

    assert_true(hasattr(bundle, "n_topics"), "bundle missing n_topics")
    n_topics = int(bundle.n_topics)
    assert_true(n_topics > 0, f"invalid n_topics={n_topics}")

    assert_true(hasattr(bundle, "schema") and bundle.schema is not None, "bundle missing schema")
    assert_true(isinstance(bundle.schema, KDDActionSchema), "bundle.schema is not KDDActionSchema")

    assert_true(hasattr(bundle, "response_models") and isinstance(bundle.response_models, dict), "Missing response_models")
    assert_true(len(bundle.response_models) >= max(1, n_topics // 2),
                f"Too few response models: {len(bundle.response_models)} for n_topics={n_topics}")

    assert_true(hasattr(bundle, "quality_bank") and bundle.quality_bank is not None, "Missing quality_bank")
    qb = bundle.quality_bank
    assert_true(hasattr(qb, "bank") and isinstance(qb.bank, dict), "quality_bank.bank missing or invalid")
    assert_true(len(qb.bank) > 0, "quality_bank has zero topic models")

    # Optional aux models: if present, must be dict.
    for name in ["hints_models", "time_models", "inc_models"]:
        m = getattr(bundle, name, None)
        if m is not None:
            assert_true(isinstance(m, dict), f"{name} must be dict or None")

    # Betas range sanity.
    betas = qb.mastery_params
    print("Mastery params:", betas)
    for fld in ["beta_good", "beta_very_good", "beta_bad", "beta_very_bad"]:
        v = float(getattr(betas, fld))
        assert_true(0.0 < v <= 0.25, f"{fld} out of range: {v}")
    assert_true(float(betas.beta_very_good) >= float(betas.beta_good) - 1e-12, "expected beta_very_good >= beta_good")
    assert_true(float(betas.beta_very_bad) >= float(betas.beta_bad) - 1e-12, "expected beta_very_bad >= beta_bad")

    # Feature names for state features are expected to be length 8.
    assert_true(hasattr(qb, "feature_names"), "quality_bank missing feature_names")
    assert_true(len(qb.feature_names) == 8, f"expected 8 state features, got {len(qb.feature_names)}")

    print("OK")

# Ensure the QualityTreeBank’s topic models are indexed correctly and cover a meaningful subset of topics.
def test_quality_bank_topics_cover_range(bundle) -> None:
    print_header("2) QualityTreeBank topic coverage")

    qb = bundle.quality_bank
    n_topics = int(bundle.n_topics)
    topics = sorted(qb.bank.keys())

    print("Topics in bank:", topics)
    assert_true(all(isinstance(t, int) for t in topics), "quality_bank topic keys must be ints")
    assert_true(min(topics) >= 0, "topic ids must be non-negative")
    assert_true(max(topics) < n_topics, "quality_bank contains topic_id >= n_topics")
    assert_true(len(topics) >= max(1, n_topics // 2), "too few topic models in bank")

    print("OK")

# Confirm the leaf-level action→quality tables are not degenerate (i.e., actions actually matter).
def test_leaf_tables_non_degenerate(bundle) -> None:
    print_header("3) Leaf tables non-degeneracy (actions matter)")

    qb = bundle.quality_bank
    total_leaves = 0
    leaves_with_variation = 0

    for topic_id, model in sorted(qb.bank.items()):
        leaf_map = model.leaf_to_action_quality
        assert_true(len(leaf_map) > 0, f"Topic {topic_id}: no leaves stored")

        for leaf_id, amap in leaf_map.items():
            total_leaves += 1
            quals = [amap.get(a, model.default_quality) for a in range(8)]
            uniq = set(quals)
            if len(uniq) > 1:
                leaves_with_variation += 1
            for q in uniq:
                assert_true(q in QUALITY_LEVELS, f"Topic {topic_id}, leaf {leaf_id}: invalid quality {q}")

    frac = leaves_with_variation / max(1, total_leaves)
    print(f"Leaves with action variation: {leaves_with_variation}/{total_leaves} = {frac:.3f}")
    assert_true(frac >= 0.50, "Too many leaves have identical quality for all actions (degenerate mapping).")

    print("OK")

# Validate that categorical qualities align with the stored numeric mean score proxy (leaf_to_action_score_mean).
def test_quality_correlates_with_score(bundle) -> None:
    print_header("4) Quality monotonicity vs stored mean scores")

    qb = bundle.quality_bank
    scores_by_q = defaultdict(list)

    for _, model in sorted(qb.bank.items()):
        for leaf, amap in model.leaf_to_action_quality.items():
            smap = model.leaf_to_action_score_mean.get(leaf, {})
            for a in range(8):
                q = amap.get(a, model.default_quality)
                s = smap.get(a, None)
                if s is None:
                    continue
                scores_by_q[q].append(float(s))

    means: Dict[str, Optional[float]] = {}
    for q in QUALITY_LEVELS:
        arr = np.asarray(scores_by_q.get(q, []), dtype=np.float32)
        means[q] = float(arr.mean()) if arr.size else None
        print(f"{q:9s}  n={arr.size:7d}  mean={means[q] if means[q] is not None else float('nan'):.6f}")

    def _ge(q1: str, q2: str, eps: float = 1e-9) -> None:
        if means[q1] is None or means[q2] is None:
            return
        assert_true(means[q1] >= means[q2] - eps, f"Expected mean({q1}) >= mean({q2})")

    _ge("very_good", "good")
    _ge("good", "neutral")
    _ge("neutral", "bad")
    _ge("bad", "very_bad")

    print("OK")

# Sanity-check the fitted KDD quantile thresholds used in action labeling and tutee proxies.
def test_schema_quantiles_sane(bundle) -> None:
    print_header("5) KDDActionSchema sanity")

    schema: KDDActionSchema = bundle.schema
    assert_true(schema.dur_q25 <= schema.dur_q50 + 1e-12, "dur_q25 > dur_q50")
    assert_true(schema.dur_q50 <= schema.dur_q75 + 1e-12, "dur_q50 > dur_q75")
    assert_true(schema.dur_q75 <= schema.dur_q90 + 1e-12, "dur_q75 > dur_q90")
    assert_true(schema.hints_q50 >= 0.0, "hints_q50 negative")
    assert_true(schema.inc_q50 >= 0.0, "inc_q50 negative")
    print("Schema:", schema)

    print("OK")


# ----------------------------
# 2) Simulator tests
# ----------------------------

# Verify feature engineering has the expected dimensionality (important for model compatibility).
def test_feature_dimensions(bundle) -> None:
    print_header("6) Feature engineering dimensions")

    env = KDDLearnerModel(cfg=KDDLearnerConfig(n_topics=bundle.n_topics), bundle=bundle, seed=0)
    env.reset(initial_mastery=0.2)

    topic = 0
    am = ActionMeta(LowLevelAction.TUTOR_QUIZ, is_tutee=False, force_generation=False)

    x_state = env._state_features(env.state, topic)  # pylint: disable=protected-access
    assert_true(x_state.shape == (8,), f"_state_features expected (8,), got {x_state.shape}")

    x = env._build_features(  # pylint: disable=protected-access
        s=env.state,
        topic_id=topic,
        action_meta=am,
        generation_mode=0,
        include_outcome=False,
    )
    assert_true(x.shape == (22,), f"_build_features expected (22,), got {x.shape}")

    print("OK")

# Validate the mastery update rule is directionally correct and bounded.
def test_mastery_update_direction(bundle) -> None:
    print_header("7) Mastery update direction + bounds")

    env = KDDLearnerModel(cfg=KDDLearnerConfig(n_topics=bundle.n_topics), bundle=bundle, seed=1)
    env.reset(initial_mastery=0.3)

    topic = 0

    def apply(q: str) -> Tuple[float, float]:
        before = float(env.state.mastery[topic])
        env._apply_mastery_quality_update(topic, q)  # pylint: disable=protected-access
        after = float(env.state.mastery[topic])
        env.state.mastery[topic] = before  # reset
        return before, after

    for q in ["very_good", "good", "neutral", "bad", "very_bad"]:
        b, a = apply(q)
        print(f"{q:9s}  {b:.4f} -> {a:.4f}")
        assert_true(0.0 <= a <= 1.0, f"mastery out of bounds after {q}: {a}")

    b, a = apply("good")
    assert_true(a >= b - 1e-12, "good should not decrease mastery")
    b, a = apply("very_good")
    assert_true(a >= b - 1e-12, "very_good should not decrease mastery")
    b, a = apply("bad")
    assert_true(a <= b + 1e-12, "bad should not increase mastery")
    b, a = apply("very_bad")
    assert_true(a <= b + 1e-12, "very_bad should not increase mastery")

    print("OK")

# Runtime smoke test for numerical stability and bounds during actual environment stepping.
def test_step_outputs_are_finite(bundle, n_steps: int = 200) -> None:
    print_header("8) Runtime step sanity (finite outputs)")

    env = KDDLearnerModel(cfg=KDDLearnerConfig(n_topics=bundle.n_topics), bundle=bundle, seed=123)
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

    for i in range(int(n_steps)):
        topic = i % int(bundle.n_topics)
        am = actions[i % len(actions)]
        _, info = env.step(topic_id=topic, action_meta=am)
        q_counts[str(info.get("quality", "neutral"))] += 1

        for k in ["p_correct", "duration", "reward"]:
            v = float(info.get(k, 0.0))
            assert_true(math.isfinite(v), f"{k} is not finite: {v}")
        for k in ["cfa", "hints", "incorrects"]:
            assert_true(isinstance(info.get(k), int), f"{k} is not int")

        m = float(env.state.mastery[topic])
        assert_true(0.0 <= m <= 1.0, f"mastery out of bounds: {m}")
        pc = float(info.get("p_correct", 0.0))
        assert_true(0.0 <= pc <= 1.0, f"p_correct out of bounds: {pc}")

    print("Quality distribution:", dict(q_counts))
    assert_true(sum(q_counts.values()) == int(n_steps), "step count mismatch")

    print("OK")

# Core behavioral validation: for the same state, different actions should yield different predicted qualities frequently.
def test_action_changes_quality_same_state(bundle, trials_per_topic: int = 120) -> None:
    print_header("9) Core: actions change quality at the same state")

    qb = bundle.quality_bank
    assert_true(qb is not None, "quality_bank missing")

    env = KDDLearnerModel(cfg=KDDLearnerConfig(n_topics=bundle.n_topics), bundle=bundle, seed=42)
    env.reset(initial_mastery=0.2)

    advance_actions = [
        ActionMeta(LowLevelAction.TUTOR_QUIZ, False, False),
        ActionMeta(LowLevelAction.TUTOR_HINT, False, False),
        ActionMeta(LowLevelAction.TUTOR_WORKED_EXAMPLE, False, False),
        ActionMeta(LowLevelAction.TUTOR_REMEDIATION, False, False),
        ActionMeta(LowLevelAction.TUTOR_REVIEW, False, False),
        ActionMeta(LowLevelAction.TUTEE_QUIZ, True, False),
        ActionMeta(LowLevelAction.TUTEE_EXPLAIN, True, False),
        ActionMeta(LowLevelAction.TUTEE_FIX, True, False),
    ]

    varied = 0
    total = 0
    for _ in range(int(trials_per_topic) * int(bundle.n_topics)):
        topic = int(env.np_rng.integers(0, int(bundle.n_topics)))
        env.step(topic_id=topic, action_meta=advance_actions[int(env.np_rng.integers(0, len(advance_actions)))])
        x_state = env._state_features(env.state, topic)  # pylint: disable=protected-access
        quals = [qb.predict_quality(topic_id=topic, action_id=a, x_state=x_state) for a in range(8)]
        total += 1
        if len(set(quals)) > 1:
            varied += 1

    frac = varied / max(1, total)
    print(f"States where action changes quality: {varied}/{total} = {frac:.3f}")
    assert_true(frac >= 0.70, "Action does not change quality often enough; mapping may be degenerate.")

    print("OK")


# Validate the response model outputs valid probabilities.
def test_response_model_probability_bounds(bundle, probes_per_topic: int = 50) -> None:
    print_header("10) Response model outputs are probabilities")

    env = KDDLearnerModel(cfg=KDDLearnerConfig(n_topics=bundle.n_topics), bundle=bundle, seed=7)
    env.reset(initial_mastery=0.2)

    for topic in range(int(bundle.n_topics)):
        model = bundle.response_models.get(topic)
        if model is None:
            continue
        for _ in range(int(probes_per_topic)):
            am = ActionMeta(LowLevelAction.TUTOR_WORKED_EXAMPLE, is_tutee=False, force_generation=False)
            x = env._build_features(  # pylint: disable=protected-access
                s=env.state,
                topic_id=topic,
                action_meta=am,
                generation_mode=0,
                include_outcome=False,
            )
            p = env._predict_proba_1(model, x)  # pylint: disable=protected-access
            assert_true(0.0 <= float(p) <= 1.0, f"topic {topic}: p out of bounds: {p}")
            env.step(topic_id=topic, action_meta=am)

    print("OK")

# Validate auxiliary outcome models produce physically meaningful values.
def test_aux_models_non_negative(bundle, probes: int = 120) -> None:
    print_header("11) Aux model outputs are non-negative")

    env = KDDLearnerModel(cfg=KDDLearnerConfig(n_topics=bundle.n_topics), bundle=bundle, seed=11)
    env.reset(initial_mastery=0.2)

    am = ActionMeta(LowLevelAction.TUTOR_QUIZ, is_tutee=False, force_generation=False)
    for i in range(int(probes)):
        topic = i % int(bundle.n_topics)
        x = env._build_features(  # pylint: disable=protected-access
            s=env.state,
            topic_id=topic,
            action_meta=am,
            generation_mode=0,
            include_outcome=False,
        )

        hints = env._predict_aux_int(topic, x, bundle.hints_models, default=0)  # pylint: disable=protected-access
        inc = env._predict_aux_int(topic, x, bundle.inc_models, default=0)  # pylint: disable=protected-access
        dur = env._predict_aux_float(topic, x, bundle.time_models, default=float(bundle.schema.dur_q50))  # pylint: disable=protected-access

        assert_true(isinstance(hints, int) and hints >= 0, f"hints invalid: {hints}")
        assert_true(isinstance(inc, int) and inc >= 0, f"incorrects invalid: {inc}")
        assert_true(isinstance(dur, float) and dur >= 0.0, f"duration invalid: {dur}")

        env.step(topic_id=topic, action_meta=am)

    print("OK")

# Ensure the simulator’s completion criterion is achievable under a reasonable policy.
def test_done_condition_achievable(bundle, max_steps: int = 5000) -> None:
    print_header("12) Done-condition achievability smoke test")

    qb = bundle.quality_bank
    assert_true(qb is not None, "quality_bank missing")

    env = KDDLearnerModel(cfg=KDDLearnerConfig(n_topics=bundle.n_topics), bundle=bundle, seed=99)
    env.reset(initial_mastery=0.2)

    for step in range(int(max_steps)):
        if env.is_done():
            print(f"Reached done in {step} steps")
            print("Final mastery:", env.state.mastery)
            print("Opp:", env.state.opp)
            print("OK")
            return

        # Prioritize topics that still violate done-condition (opp deficit first, then mastery deficit)
        mastery = env.state.mastery
        opp = env.state.opp
        thr = float(env.cfg.mastery_threshold)
        opp_min = int(env.cfg.opp_min)

        need_opp = np.where(opp < opp_min)[0]
        if need_opp.size > 0:
            # among under-covered topics, focus the weakest mastery
            topic = int(need_opp[np.argmin(mastery[need_opp])])
        else:
            need_mastery = np.where(mastery < thr)[0]
            if need_mastery.size > 0:
                topic = int(need_mastery[np.argmin(mastery[need_mastery])])
            else:
                # already satisfies mastery everywhere; just finish coverage if needed (shouldn't happen if need_opp handled)
                topic = int(np.argmin(opp))

        x_state = env._state_features(env.state, topic)  # pylint: disable=protected-access

        best_a = 0
        best_key = (-1, -1e9)
        for a in range(8):
            q = qb.predict_quality(topic_id=topic, action_id=a, x_state=x_state)
            qv = QUALITY_ORDER.get(q, 2)
            s = qb.predict_score_mean(topic_id=topic, action_id=a, x_state=x_state)
            key = (qv, float(s) if s is not None else 0.0)
            if key > best_key:
                best_key = key
                best_a = a

        am = ActionMeta(action=LowLevelAction(best_a), is_tutee=(best_a >= 5), force_generation=False)
        env.step(topic_id=topic, action_meta=am)

    raise AssertionError(
        f"Did not reach done within {max_steps} steps. "
        "This may indicate thresholds are too strict or quality updates too weak."
    )


# ----------------------------
# 3) Optional tests: agents + sharing
# ----------------------------

# Verify HRL agent action spaces match the intended design.
def test_agent_action_spaces() -> None:
    if not _HAS_AGENTS:
        print_header("13) Agent tests skipped (torch/agents unavailable)")
        return

    print_header("13) Agent action spaces")
    num_topics = 7

    hl0 = HighLevelAgent(HighLevelAgentConfig(num_topics=num_topics, use_tutee=False))
    assert_true(hl0.num_actions == num_topics, "HL without tutee should have num_topics actions")

    hl1 = HighLevelAgent(HighLevelAgentConfig(num_topics=num_topics, use_tutee=True))
    assert_true(hl1.num_actions == 2 * num_topics, "HL with tutee should have 2*num_topics actions")

    ll_cfg = LowLevelAgentConfig(num_topics=num_topics)
    tutor = TutorLowLevelAgent(ll_cfg)
    tutee = TuteeLowLevelAgent(ll_cfg)

    assert_true(len(tutor.get_action_meanings()) == 5, "Tutor LL should have 5 actions")
    assert_true(len(tutee.get_action_meanings()) == 4, "Tutee LL should have 4 actions")

    print("OK")

# Sanity-check the experience-sharing similarity metric implementation (Linear CKA).
def test_cka_in_range() -> None:
    if not _HAS_AGENTS:
        print_header("14) CKA tests skipped (torch/agents unavailable)")
        return

    print_header("14) Linear CKA is in [0,1]")

    X = torch.randn(64, 32)
    Y = torch.randn(64, 16)
    v = float(linear_cka(X, Y).item())
    print("CKA:", v)
    assert_true(0.0 <= v <= 1.0, "CKA out of bounds")

    print("OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="../data/kdd_bundle.joblib")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max_done_steps", type=int, default=5000)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)

    test_bundle_integrity(bundle)
    test_quality_bank_topics_cover_range(bundle)
    test_leaf_tables_non_degenerate(bundle)
    test_quality_correlates_with_score(bundle)

    test_schema_quantiles_sane(bundle)
    test_feature_dimensions(bundle)
    test_mastery_update_direction(bundle)
    test_step_outputs_are_finite(bundle, n_steps=int(args.steps))
    test_action_changes_quality_same_state(bundle, trials_per_topic=120)
    test_response_model_probability_bounds(bundle, probes_per_topic=50)
    test_aux_models_non_negative(bundle, probes=120)
    test_done_condition_achievable(bundle, max_steps=int(args.max_done_steps))

    test_agent_action_spaces()
    test_cka_in_range()

    print_header("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
