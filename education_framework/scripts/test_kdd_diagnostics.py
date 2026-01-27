# education_framework/scripts/test_kdd_diagnostics.py
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from dataclasses import is_dataclass
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np


# Run with
# python -m education_framework.scripts.test_kdd_diagnostics --bundle education_framework/data/kdd_bundle.joblib --out runs/diag_kdd.json --n_state_samples 500 --seed 0





QUALITY_ORDER = {"very_bad": 0, "bad": 1, "neutral": 2, "good": 3, "very_good": 4}
TUTOR_ACTIONS = list(range(0, 5))
TUTEE_ACTIONS = list(range(5, 8))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_attr(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def quality_to_int(q: str) -> int:
    return QUALITY_ORDER.get(str(q), QUALITY_ORDER["neutral"])


def best_quality(leaf_map: Dict[int, str], action_ids: List[int]) -> int:
    vals = [quality_to_int(leaf_map.get(a, "neutral")) for a in action_ids]
    return max(vals) if vals else QUALITY_ORDER["neutral"]


def count_quality_levels(leaf_map: Dict[int, str], action_ids: List[int]) -> Dict[str, int]:
    out = {k: 0 for k in QUALITY_ORDER.keys()}
    for a in action_ids:
        q = leaf_map.get(int(a), "neutral")
        q = q if q in QUALITY_ORDER else "neutral"
        out[q] += 1
    return out


def synth_states(
    n: int,
    mastery_k: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build synthetic x_state vectors in the same 8D format used by your QualityTreeBank training:
      [mastery_k, opp_k_norm, cfa_ema_k, hint_ema_k, time_ema_k, inc_ema_k, global_mastery, total_steps_norm]
    We randomize the other 7 dimensions in [0, 1] for coverage.
    """
    X = rng.random((n, 8), dtype=np.float32)
    X[:, 0] = float(mastery_k)
    X[:, 6] = np.clip(X[:, 6], 0.0, 1.0)  # global mastery
    return X.astype(np.float32)


def audit_bank(bundle: Any, *, n_state_samples: int, seed: int) -> Dict[str, Any]:
    qbank = get_attr(bundle, "quality_bank", None)
    if qbank is None:
        raise RuntimeError("Bundle has no quality_bank; cannot audit.")

    bank = get_attr(qbank, "bank", {})
    num_topics = int(get_attr(bundle, "n_topics", len(bank) if bank else 0))

    # Basic invariants / summary
    topic_summaries: Dict[int, Any] = {}
    overall = {
        "topics_present": sorted(list(bank.keys())),
        "num_topics_in_bundle": num_topics,
        "invariant_all_leaves_have_0_7": True,
    }

    # Dominance counters
    total_leaves = 0
    tutee_wins = 0
    tutor_wins = 0
    ties = 0

    # Low mastery readiness audit accumulators
    rng = np.random.default_rng(seed)
    low_mastery_points = [0.05, 0.10, 0.20]
    low_mastery_stats = {m: {"tutee_good_or_better_rate": 0.0, "tutee_very_good_rate": 0.0} for m in low_mastery_points}

    for topic_id, model in bank.items():
        leaf_to_action_quality = get_attr(model, "leaf_to_action_quality", {}) or {}
        leaves = list(leaf_to_action_quality.keys())
        total_leaves += len(leaves)

        # Invariant: each leaf has 0..7 keys (or at least defaulted)
        missing_counts = 0

        # Per-topic dominance
        t_wins = 0
        r_wins = 0
        t_ties = 0

        # Per-topic quality distribution
        tutor_quality_hist = {k: 0 for k in QUALITY_ORDER.keys()}
        tutee_quality_hist = {k: 0 for k in QUALITY_ORDER.keys()}

        for leaf in leaves:
            amap = leaf_to_action_quality.get(int(leaf), {}) or {}

            # Check keys 0..7
            for a in range(8):
                if int(a) not in amap:
                    missing_counts += 1
                    overall["invariant_all_leaves_have_0_7"] = False
                    break

            btutor = best_quality(amap, TUTOR_ACTIONS)
            btutee = best_quality(amap, TUTEE_ACTIONS)

            if btutee > btutor:
                t_wins += 1
            elif btutor > btutee:
                r_wins += 1
            else:
                t_ties += 1

            # hist by label count
            tq = count_quality_levels(amap, TUTOR_ACTIONS)
            uq = count_quality_levels(amap, TUTEE_ACTIONS)
            for k in QUALITY_ORDER.keys():
                tutor_quality_hist[k] += tq[k]
                tutee_quality_hist[k] += uq[k]

        # add to overall
        tutee_wins += t_wins
        tutor_wins += r_wins
        ties += t_ties

        # Low-mastery readiness audit using synthetic states routed through the tree
        # We evaluate how often any tutee action is labeled good/very_good when mastery is low.
        tree = get_attr(model, "tree", None)
        if tree is not None and n_state_samples > 0 and len(leaves) > 0:
            for m in low_mastery_points:
                X = synth_states(n_state_samples, mastery_k=m, rng=rng)
                # route to leaves for this topic
                leaf_ids = tree.apply(X).astype(int)
                good_or_better = 0
                very_good = 0
                for lid, x in zip(leaf_ids, X):
                    amap = leaf_to_action_quality.get(int(lid), None)
                    if not amap:
                        continue
                    best_tutee = best_quality(amap, TUTEE_ACTIONS)
                    if best_tutee >= QUALITY_ORDER["good"]:
                        good_or_better += 1
                    if best_tutee == QUALITY_ORDER["very_good"]:
                        very_good += 1
                denom = max(1, len(leaf_ids))
                low_mastery_stats[m]["tutee_good_or_better_rate"] += good_or_better / denom
                low_mastery_stats[m]["tutee_very_good_rate"] += very_good / denom

        topic_summaries[int(topic_id)] = {
            "num_leaves": len(leaves),
            "tutee_best_quality_gt_tutor_best_quality_leaves": t_wins,
            "tutor_best_quality_gt_tutee_best_quality_leaves": r_wins,
            "ties_leaves": t_ties,
            "tutee_win_rate": (t_wins / len(leaves)) if leaves else 0.0,
            "missing_action_key_count_estimate": missing_counts,
            "tutor_quality_hist_counts": tutor_quality_hist,
            "tutee_quality_hist_counts": tutee_quality_hist,
        }

    # finalize low mastery stats (average across topics that had trees)
    topics_with_models = max(1, len(bank))
    for m in low_mastery_points:
        low_mastery_stats[m]["tutee_good_or_better_rate"] /= topics_with_models
        low_mastery_stats[m]["tutee_very_good_rate"] /= topics_with_models

    overall.update({
        "total_leaves": total_leaves,
        "overall_tutee_win_rate": (tutee_wins / total_leaves) if total_leaves else 0.0,
        "overall_tutor_win_rate": (tutor_wins / total_leaves) if total_leaves else 0.0,
        "overall_tie_rate": (ties / total_leaves) if total_leaves else 0.0,
        "low_mastery_readiness_audit": low_mastery_stats,
    })

    return {"overall": overall, "topics": topic_summaries}


def neutralize_tutee_qualities(bundle: Any) -> Any:
    """
    Returns a deep-copied bundle where, in every topic/leaf, action 5..7 is forced to 'neutral'.
    This is the "smoking gun" ablation: if tutee advantage disappears, the gain is from the tutee leaf tables.
    """
    b2 = copy.deepcopy(bundle)
    qbank = get_attr(b2, "quality_bank", None)
    if qbank is None:
        return b2
    bank = get_attr(qbank, "bank", {})
    for _, model in bank.items():
        leaf_to_action_quality = get_attr(model, "leaf_to_action_quality", {}) or {}
        for leaf, amap in leaf_to_action_quality.items():
            for a in TUTEE_ACTIONS:
                amap[int(a)] = "neutral"
    return b2


def bundle_fingerprint(bundle_path: str, bundle: Any) -> Dict[str, Any]:
    fp = {
        "bundle_path": os.path.abspath(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_bytes": os.path.getsize(bundle_path),
    }

    # Try to extract mastery betas if present
    qbank = get_attr(bundle, "quality_bank", None)
    mp = get_attr(qbank, "mastery_params", None)
    if mp is not None:
        fp["mastery_params"] = {
            "beta_good": float(get_attr(mp, "beta_good", 0.0)),
            "beta_very_good": float(get_attr(mp, "beta_very_good", 0.0)),
            "beta_bad": float(get_attr(mp, "beta_bad", 0.0)),
            "beta_very_bad": float(get_attr(mp, "beta_very_bad", 0.0)),
            "mastery_jump": float(get_attr(mp, "mastery_jump", 0.0)),
        }

    fp["n_topics"] = int(get_attr(bundle, "n_topics", -1))
    fp["ema_alpha"] = float(get_attr(bundle, "ema_alpha", -1.0))
    fp["one_hot_actions"] = bool(get_attr(bundle, "one_hot_actions", False))
    return fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, default="education_framework/data/kdd_bundle.joblib")
    ap.add_argument("--out", default="runs/diag_kdd.json")
    ap.add_argument("--n_state_samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    bundle = joblib.load(args.bundle)
    report: Dict[str, Any] = {}
    report["fingerprint"] = bundle_fingerprint(args.bundle, bundle)

    report["audit_original"] = audit_bank(bundle, n_state_samples=args.n_state_samples, seed=args.seed)

    bundle_neutral = neutralize_tutee_qualities(bundle)
    report["audit_tutee_neutralized"] = audit_bank(bundle_neutral, n_state_samples=args.n_state_samples, seed=args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # concise console summary
    o = report["audit_original"]["overall"]
    n = report["audit_tutee_neutralized"]["overall"]
    print("=== Bundle fingerprint ===")
    print(json.dumps(report["fingerprint"], indent=2))
    print("\n=== Original bank dominance ===")
    print(f"Total leaves: {o['total_leaves']}")
    print(f"Tutee win rate (best tutee quality > best tutor quality): {o['overall_tutee_win_rate']:.3f}")
    print(f"Tutor win rate: {o['overall_tutor_win_rate']:.3f} | Tie rate: {o['overall_tie_rate']:.3f}")
    print(f"All leaves have action keys 0..7: {o['invariant_all_leaves_have_0_7']}")
    print("Low-mastery readiness (avg across topics):")
    for m, d in o["low_mastery_readiness_audit"].items():
        print(f"  mastery={m}: tutee>=good {d['tutee_good_or_better_rate']:.3f}, tutee==very_good {d['tutee_very_good_rate']:.3f}")

    print("\n=== After neutralizing tutee qualities ===")
    print(f"Tutee win rate: {n['overall_tutee_win_rate']:.3f} (should drop near 0)")
    print(f"Tutor win rate: {n['overall_tutor_win_rate']:.3f} | Tie rate: {n['overall_tie_rate']:.3f}")

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
