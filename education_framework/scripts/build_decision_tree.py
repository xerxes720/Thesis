from __future__ import annotations
# education_framework/scripts/build_decision_tree.py
"""build_decision_tree.py

Train KDD-based per-topic models and a *paper-style* QualityTreeBank.

Outputs (joblib):
- KDDModelBundle, containing:
    * KDDActionSchema (quantiles)
    * kc_to_topic mapping (clustered into n_topics)
    * response_models: per-topic DecisionTreeClassifier for P(CFA=1)
    * optional auxiliary outcome models (hints/inc/duration)
    * quality_bank: per-topic routing tree + leaf/action quality categories

This replaces the earlier "transition regressor" with an interpretable leaf-quality simulator,
which is easier to explain and avoids EMA double-counting.
"""

"""
Run with: 
python -m education_framework.scripts.build_decision_tree --csv education_framework/data/algebra_2005_2006_train.txt --out education_framework/data/kdd_bundle.joblib --n_topics 7 --max_depth 7 --min_leaf 50 --ema_alpha 0.2 --seed 0 

"""



import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from education_framework.utils.kdd_utils import group_rows_by_student_ordered, read_kdd_table

from education_framework.environment.learner_model import (
    KDDActionSchema,
    KDDModelBundle,
    KDDTrajectoryBuilder,
    LowLevelAction,
)
from education_framework.models.quality_tree_bank import QualityTreeBank, LeafQualityModel, MasteryUpdateParams, QUALITY_LEVELS


# ----------------------------
# KC -> Topic mapping
# ----------------------------

def _extract_primary_kc(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if "~~" in s:
        s = s.split("~~")[0].strip()
    return s if s else None


def build_kc_to_topic_from_json(path: str, n_topics: int) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("kc_map_json must be a JSON object mapping KC string -> topic_id")
    out: Dict[str, int] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        tid = int(v)
        if tid < 0 or tid >= n_topics:
            raise ValueError(f"Topic id out of range for KC={k}: {tid}")
        out[k.strip()] = tid
    return out


def build_kc_to_topic_by_clustering(
    df: pd.DataFrame,
    n_topics: int,
    kc_col: str,
    problem_col_candidates: Sequence[str] = ("Problem Name", "problem_id", "Problem Id", "Problem"),
    max_kcs: int = 250,
    min_kc_freq: int = 50,
    seed: int = 0,
) -> Dict[str, int]:
    if kc_col not in df.columns:
        raise ValueError(f"KC column not found: {kc_col}")

    problem_col = None
    for c in problem_col_candidates:
        if c in df.columns:
            problem_col = c
            break
    if problem_col is None:
        problem_col = "Step Name" if "Step Name" in df.columns else "__row_bucket__"
        if problem_col == "__row_bucket__":
            df = df.copy()
            df[problem_col] = (np.arange(len(df)) // 10).astype(int)

    kcs = df[kc_col].apply(_extract_primary_kc)
    freq = kcs.value_counts(dropna=True)
    keep = [kc for kc, c in freq.items() if (kc is not None) and (int(c) >= min_kc_freq)]
    keep = keep[:max_kcs]
    if len(keep) < max(10, n_topics):
        # fallback: keep more if sparse
        keep = [kc for kc, _ in freq.items() if kc is not None][: max(10, n_topics)]

    df2 = df.copy()
    df2["__kc__"] = kcs
    df2 = df2[df2["__kc__"].isin(keep)]

    problems = df2[problem_col].astype(str).tolist()
    kc_list = df2["__kc__"].astype(str).tolist()

    prob_index: Dict[str, int] = {}
    for p in problems:
        if p not in prob_index:
            prob_index[p] = len(prob_index)

    kc_index: Dict[str, int] = {kc: i for i, kc in enumerate(sorted(set(kc_list)))}

    X = np.zeros((len(kc_index), len(prob_index)), dtype=np.float32)
    for kc, p in zip(kc_list, problems):
        X[kc_index[kc], prob_index[p]] = 1.0

    # Normalize rows (cosine-ish)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm < 1e-6] = 1.0
    Xn = X / norm

    km = KMeans(n_clusters=n_topics, random_state=seed, n_init=10)
    labels = km.fit_predict(Xn)

    return {kc: int(labels[i]) for kc, i in kc_index.items()}


# ----------------------------
# Training
# ----------------------------

@dataclass
class TrainConfig:
    n_topics: int = 7
    max_depth: int = 7
    min_samples_leaf: int = 50
    ema_alpha: float = 0.2
    seed: int = 0

    schema_max_rows: int = 2_000_000

    # quality tree constraints
    min_leaf_action_count: int = 25

    # quality labeling: treat small deltas as neutral to avoid sign/semantic mismatch
    quality_eps: float = 0.003  # mastery-delta noise floor for leaf-wise action scoring

    # NOTE: KDD has no explicit tutee signal. Tutee effects are modeled mechanistically at runtime.

# def _rank_to_quality(action_ids: List[int], scores: List[float]) -> Dict[int, str]:
#     """Map 8 actions to 5 categories by rank: 1/2/2/2/1 buckets."""
#     pairs = sorted(zip(action_ids, scores), key=lambda x: x[1])  # ascending
#     # worst..best
#     buckets = {
#         "very_bad": [pairs[0][0]],
#         "bad": [pairs[1][0], pairs[2][0]],
#         "neutral": [pairs[3][0], pairs[4][0]],
#         "good": [pairs[5][0], pairs[6][0]],
#         "very_good": [pairs[7][0]],
#     }
#     out: Dict[int, str] = {}
#     for q, ids in buckets.items():
#         for a in ids:
#             out[int(a)] = q
#     return out

# def _rank_to_quality(action_ids: List[int], scores: List[float], eps: float = 0.003) -> Dict[int, str]:
#     """Map actions to {very_good, good, neutral, bad, very_bad} using a magnitude-aware hybrid rule.
#
#     Rationale:
#       - Avoid labeling empirically positive actions as 'bad/very_bad', because downstream transitions
#         treat bad/very_bad as mastery decay.
#       - Preserve the policy-shaping benefit of relative ranking by ranking within the positive set
#         (and within the negative set), while assigning near-zero effects to 'neutral'.
#
#     Rule (per leaf):
#       - If score >  +eps  => positive set  (eligible for good/very_good)
#       - If score <  -eps  => negative set  (eligible for bad/very_bad)
#       - Otherwise         => neutral
#
#     Within sets:
#       - positive: best -> very_good (if ≥2 positives), remaining -> good
#       - negative: worst -> very_bad (if ≥2 negatives), remaining -> bad
#
#     Notes:
#       - This may yield fewer than 5 labels in a given leaf (e.g., no negatives).
#       - All actions always receive a label.
#     """
#     if len(action_ids) != len(scores):
#         raise ValueError(f"action_ids and scores must have same length, got {len(action_ids)} and {len(scores)}")
#
#     out: Dict[int, str] = {int(a): "neutral" for a in action_ids}
#
#     pos = [(int(a), float(s)) for a, s in zip(action_ids, scores) if s > eps]
#     neg = [(int(a), float(s)) for a, s in zip(action_ids, scores) if s < -eps]
#
#     if pos:
#         pos_sorted = sorted(pos, key=lambda x: x[1])  # ascending; last is best
#         best_a = pos_sorted[-1][0]
#         out[best_a] = "very_good" if len(pos_sorted) >= 2 else "good"
#         for a, _ in pos_sorted[:-1]:
#             out[a] = "good"
#
#     if neg:
#         neg_sorted = sorted(neg, key=lambda x: x[1])  # ascending; first is worst (most negative)
#         worst_a = neg_sorted[0][0]
#         out[worst_a] = "very_bad" if len(neg_sorted) >= 2 else "bad"
#         for a, _ in neg_sorted[1:]:
#             out[a] = "bad"
#
#     return out

def _compute_global_cutoffs(deltas: List[float], *, neutral_eps: float) -> tuple[float, float, float, float]:
    """Compute global cutoffs for 5-way quality binning.

    Use global quantiles when there is enough data; otherwise fall back to symmetric
    thresholds around 0 (scaled by neutral_eps).

    Returns:
        (q20, q40, q60, q80)
    """
    arr = np.asarray(deltas, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size >= 200:
        q20, q40, q60, q80 = np.quantile(arr, [0.2, 0.4, 0.6, 0.8]).tolist()
        return float(q20), float(q40), float(q60), float(q80)

    # Fallback: keep a neutral band around 0, and define tails relative to it.
    q20 = -3.0 * neutral_eps
    q40 = -1.0 * neutral_eps
    q60 = +1.0 * neutral_eps
    q80 = +3.0 * neutral_eps
    return float(q20), float(q40), float(q60), float(q80)


def _scores_to_quality_global(action_ids, scores, *, cutoffs, neutral_eps=0.003):
    q20, q40, q60, q80 = cutoffs
    out = {}
    for a, s in zip(action_ids, scores):
        a = int(a)
        s = float(s)

        # Neutral only if globally "middle"
        if q40 <= s <= q60:
            out[a] = "neutral"
        elif s <= q20:
            out[a] = "very_bad"
        elif s <= q40:
            out[a] = "bad"
        elif s <= q80:
            out[a] = "good"
        else:
            out[a] = "very_good"
    return out



def train_bundle_from_kdd_csv(
    csv_path: str,
    out_path: str,
    cfg: TrainConfig,
    *,
    kc_col: str,
    student_col: str,
    order_cols: Optional[List[str]],
    cfa_col: str,
    duration_col: str,
    hints_col: str,
    incorrects_col: str,
    corrects_col: str,
    kc_map_json: Optional[str] = None,
    cluster_max_kcs: int = 250,
    cluster_min_kc_freq: int = 50,
    train_aux_models: bool = True,
) -> None:
    df = read_kdd_table(csv_path)

    # fit schema
    schema = KDDActionSchema.fit_from_rows(
        rows=df.to_dict(orient="records"),
        duration_col=duration_col,
        hints_col=hints_col,
        incorrects_col=incorrects_col,
        corrects_col=corrects_col,
        max_rows=cfg.schema_max_rows,
    )

    for col in [cfa_col, duration_col, hints_col, incorrects_col, corrects_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # kc->topic
    if kc_map_json:
        kc_to_topic = build_kc_to_topic_from_json(kc_map_json, cfg.n_topics)
    else:
        kc_to_topic = build_kc_to_topic_by_clustering(
            df=df,
            n_topics=cfg.n_topics,
            kc_col=kc_col,
            max_kcs=cluster_max_kcs,
            min_kc_freq=cluster_min_kc_freq,
            seed=cfg.seed,
        )

    # rows by student
    if order_cols:
        order_cols = [c for c in order_cols if c in df.columns]
        if not order_cols:
            order_cols = None

    rows_by_student = group_rows_by_student_ordered(df, student_col=student_col, order_cols=order_cols)

    builder = KDDTrajectoryBuilder(
        n_topics=cfg.n_topics,
        kc_to_topic=kc_to_topic,
        schema=schema,
        ema_alpha=cfg.ema_alpha,
        seed=cfg.seed,
    )

    # collect per-topic data
    X_resp: Dict[int, List[np.ndarray]] = {k: [] for k in range(cfg.n_topics)}
    y_cfa: Dict[int, List[int]] = {k: [] for k in range(cfg.n_topics)}

    X_state: Dict[int, List[np.ndarray]] = {k: [] for k in range(cfg.n_topics)}
    mastery_pre: Dict[int, List[float]] = {k: [] for k in range(cfg.n_topics)}
    delta_m: Dict[int, List[float]] = {k: [] for k in range(cfg.n_topics)}
    act_id: Dict[int, List[int]] = {k: [] for k in range(cfg.n_topics)}

    hints_y: Dict[int, List[int]] = {k: [] for k in range(cfg.n_topics)}
    inc_y: Dict[int, List[int]] = {k: [] for k in range(cfg.n_topics)}
    dur_y: Dict[int, List[float]] = {k: [] for k in range(cfg.n_topics)}

    # KDD has no explicit tutee outcomes; we do not derive or proxy them from the dataset.
    # The tutee effect is modeled mechanistically in the simulator.
    tutee_outcome_topic = None
    tutee_outcome_leaf = None

    for tr in builder.iter_training_rows(
        rows_by_student,
        kc_col=kc_col,
        cfa_col=cfa_col,
        duration_col=duration_col,
        hints_col=hints_col,
        incorrects_col=incorrects_col,
    ):
        k = int(tr.topic_id)
        X_resp[k].append(tr.x_resp)
        y_cfa[k].append(int(tr.cfa))

        X_state[k].append(tr.x_state)
        mastery_pre[k].append(float(tr.mastery_pre))
        delta_m[k].append(float(tr.delta_mastery))
        act_id[k].append(int(tr.action_id))

        hints_y[k].append(int(tr.hints))
        inc_y[k].append(int(tr.incorrects))
        dur_y[k].append(float(tr.duration))

    # train response and aux models
    response_models: Dict[int, Any] = {}
    hints_models: Dict[int, Any] = {}
    inc_models: Dict[int, Any] = {}
    time_models: Dict[int, Any] = {}

    for k in range(cfg.n_topics):
        Xk = np.asarray(X_resp[k], dtype=np.float32)
        yk = np.asarray(y_cfa[k], dtype=np.int32)
        if Xk.shape[0] < max(200, cfg.min_samples_leaf * 4):
            continue
        clf = DecisionTreeClassifier(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf, random_state=cfg.seed)
        clf.fit(Xk, yk)
        response_models[k] = clf

        if train_aux_models:
            y_h = np.asarray(hints_y[k], dtype=np.float32)
            y_i = np.asarray(inc_y[k], dtype=np.float32)
            y_t = np.asarray(dur_y[k], dtype=np.float32)

            m_h = np.isfinite(y_h)
            m_i = np.isfinite(y_i)
            m_t = np.isfinite(y_t)

            # fit each model on its own valid subset
            if m_h.sum() >= cfg.min_samples_leaf * 4:
                hr = DecisionTreeRegressor(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
                                           random_state=cfg.seed)
                hr.fit(Xk[m_h], y_h[m_h])
                hints_models[k] = hr

            if m_i.sum() >= cfg.min_samples_leaf * 4:
                ir = DecisionTreeRegressor(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
                                           random_state=cfg.seed)
                ir.fit(Xk[m_i], y_i[m_i])
                inc_models[k] = ir

            if m_t.sum() >= cfg.min_samples_leaf * 4:
                trr = DecisionTreeRegressor(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
                                            random_state=cfg.seed)
                trr.fit(Xk[m_t], y_t[m_t])
                time_models[k] = trr


    # ----------------------------
    # Train paper-style QualityTreeBank
    # ----------------------------
    feature_names = [
        "mastery_k",
        "opp_k_norm",
        "cfa_ema_k",
        "hint_ema_k",
        "time_ema_k",
        "inc_ema_k",
        "global_mastery",
        "total_steps_norm",
    ]

    qbank = QualityTreeBank(num_topics=cfg.n_topics, feature_names=feature_names)

    # topic-level fallbacks
    topic_action_mean: Dict[int, Dict[int, float]] = {}
    # topic-level global cutoffs (for absolute quality labels)
    topic_cutoffs: Dict[int, tuple[float, float, float, float]] = {}

    for k in range(cfg.n_topics):
        topic_action_mean[k] = {}
        if not delta_m[k]:
            # fallback cutoffs if there is literally no data
            topic_cutoffs[k] = _compute_global_cutoffs([], neutral_eps=cfg.quality_eps)
            continue

        # existing topic_action_mean computation (keep it)
        for a in range(5):
            vals = [dm for dm, aa in zip(delta_m[k], act_id[k]) if aa == a]
            if vals:
                topic_action_mean[k][a] = float(np.mean(vals))

        # NEW: global distribution over tutor actions for this topic
        all_tutor_deltas = [dm for dm, aa in zip(delta_m[k], act_id[k]) if aa in (0, 1, 2, 3, 4)]
        topic_cutoffs[k] = _compute_global_cutoffs(all_tutor_deltas, neutral_eps=cfg.quality_eps)
        # NOTE: KDD has no explicit tutee interactions; we do not infer tutee effects from data.
        # Keep a neutral (0) topic-level fallback for tutee actions (ids 5..7).
        for a_tutee in (
            int(LowLevelAction.TUTEE_QUIZ),
            int(LowLevelAction.TUTEE_EXPLAIN),
            int(LowLevelAction.TUTEE_FIX),
        ):
            topic_action_mean[k][a_tutee] = 0.0

    # --- NEW: per-topic per-action gain multipliers (data-driven) ---
    # Stable normalization: use relative advantages, not raw std (avoids exploding when variance is tiny).
    # topic_tutor_action_gain = np.ones((cfg.n_topics, 5), dtype=np.float32)
    #
    # G = 0.15  # strength (0.10..0.20 recommended)
    # LO, HI = 0.85, 1.15
    #
    # for k in range(cfg.n_topics):
    #     means = []
    #     for a in range(5):
    #         vals = [dm for dm, aa in zip(delta_m[k], act_id[k]) if aa == a]
    #         means.append(float(np.mean(vals)) if vals else float(topic_action_mean.get(k, {}).get(a, 0.0)))
    #
    #     mu = float(np.mean(means))
    #     adv = np.asarray([m - mu for m in means], dtype=np.float32)
    #     denom = float(np.max(np.abs(adv))) + 1e-8  # scale by max deviation
    #     rel = adv / denom  # in [-1,1]
    #     gains = 1.0 + G * rel
    #     gains = np.clip(gains, LO, HI)
    #     topic_tutor_action_gain[k, :] = gains

    # print(topic_tutor_action_gain.shape)
    for k in range(cfg.n_topics):
        # NOTE: we no longer compute any tutee outcome tables from the dataset.
        Xs = np.asarray(X_state[k], dtype=np.float32)
        yk = np.asarray(y_cfa[k], dtype=np.int32)
        if Xs.shape[0] < max(500, cfg.min_samples_leaf * 8):
            continue

        route = DecisionTreeClassifier(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf, random_state=cfg.seed)
        route.fit(Xs, yk)
        leaf_ids = route.apply(Xs).astype(int)

        # pre-index samples per leaf
        idx_by_leaf: Dict[int, np.ndarray] = {}
        for leaf in np.unique(leaf_ids):
            idx_by_leaf[int(leaf)] = np.where(leaf_ids == leaf)[0]

        leaf_to_action_score: Dict[int, Dict[int, float]] = {}
        leaf_to_action_quality: Dict[int, Dict[int, str]] = {}

        for leaf, idxs in idx_by_leaf.items():
            # action scores for 0..4 (tutor-labelled)
            scores: Dict[int, float] = {}
            for a in range(5):
                vals = [delta_m[k][i] for i in idxs if act_id[k][i] == a]
                if len(vals) >= cfg.min_leaf_action_count:
                    scores[a] = float(np.mean(vals))

            # NOTE: KDD has no explicit tutee interactions; we do not infer tutee effects from data.
            # Keep neutral (0) scores for the three tutee actions in the QualityTreeBank.
            scores[int(LowLevelAction.TUTEE_QUIZ)] = 0.0
            scores[int(LowLevelAction.TUTEE_EXPLAIN)] = 0.0
            scores[int(LowLevelAction.TUTEE_FIX)] = 0.0

            # fill missing actions with topic-level means (or 0)
            for a in range(8):
                if a in scores:
                    continue
                scores[a] = float(topic_action_mean.get(k, {}).get(a, 0.0))

            # store scores for all actions (unchanged)
            action_ids = list(range(8))
            leaf_to_action_score[leaf] = {a: float(scores[a]) for a in action_ids}

            # Tutor qualities come from KDD-derived leaf scores.
            tutor_ids = [0, 1, 2, 3, 4]
            tutor_scores = [float(scores[a]) for a in tutor_ids]
            tutor_q = _scores_to_quality_global(
                tutor_ids,
                tutor_scores,
                cutoffs=topic_cutoffs[k],
                neutral_eps=cfg.quality_eps,
            )
            # Tutee qualities are NOT derived from KDD (no direct tutee signal).
            # We keep them neutral in the bank; the tutee learning effect is modeled
            # mechanistically in the simulator via a bounded mastery bonus + explicit cost.
            leaf_quality = dict(tutor_q)
            for a_tutee in (5, 6, 7):
                leaf_quality[a_tutee] = "neutral"

            leaf_to_action_quality[leaf] = leaf_quality

            # (intentionally no dataset-derived tutee outcome proxies)

        # NOTE: We no longer compute dataset-derived tutee outcome proxies.
        # KDD has no explicit tutee interactions; the tutee effect is modeled
        # mechanistically in the simulator (bounded mastery bonus + explicit cost).

        qbank.bank[k] = LeafQualityModel(
            tree=route,
            leaf_to_action_quality=leaf_to_action_quality,
            leaf_to_action_score_mean=leaf_to_action_score,
            default_quality="neutral",
        )

    # ----------------------------
    # Calibrate betas from tutor-labelled data (0..4)
    # ----------------------------
    # Use the quality mapping induced by the bank on the same samples.
    beta_pos_by_cat: Dict[str, List[float]] = {"good": [], "very_good": []}
    beta_neg_by_cat: Dict[str, List[float]] = {"bad": [], "very_bad": []}

    neutral_deltas: List[float] = []

    for k in range(cfg.n_topics):
        m = qbank.bank.get(k)
        if m is None:
            continue
        Xs = np.asarray(X_state[k], dtype=np.float32)
        leaf_ids = m.tree.apply(Xs).astype(int)

        for i, leaf in enumerate(leaf_ids):
            a = int(act_id[k][i])
            if a < 0 or a > 4:
                continue
            q = m.leaf_to_action_quality.get(int(leaf), {}).get(a, "neutral")
            dm = float(delta_m[k][i])
            mp = float(mastery_pre[k][i])

            if q == "neutral":
                neutral_deltas.append(dm)
            if q in ("good", "very_good") and dm > 0:
                denom = max(1e-6, (1.0 - mp))
                beta_pos_by_cat[q].append(dm / denom)
            if q in ("bad", "very_bad") and dm < 0:
                denom = max(1e-6, mp)
                beta_neg_by_cat[q].append((-dm) / denom)

    def _robust_mean(xs: List[float]) -> float:
        if not xs:
            return 0.0
        arr = np.asarray(xs, dtype=np.float32)
        lo, hi = np.quantile(arr, [0.10, 0.90])
        arr = arr[(arr >= lo) & (arr <= hi)]
        return float(arr.mean()) if arr.size else float(np.mean(xs))

    def _robust_std(xs: List[float]) -> float:
        if not xs:
            return 0.005  # small default
        arr = np.asarray(xs, dtype=np.float32)
        lo, hi = np.quantile(arr, [0.10, 0.90])
        arr = arr[(arr >= lo) & (arr <= hi)]
        return float(arr.std()) if arr.size else float(np.std(xs))

    neutral_noise_std = float(np.clip(_robust_std(neutral_deltas), 0.001, 0.02))


    beta_good = _robust_mean(beta_pos_by_cat["good"])
    beta_vgood = _robust_mean(beta_pos_by_cat["very_good"])
    beta_bad = _robust_mean(beta_neg_by_cat["bad"])
    beta_vbad = _robust_mean(beta_neg_by_cat["very_bad"])

    # clamp to reasonable simulator ranges
    beta_good = float(np.clip(beta_good, 0.005, 0.15))
    beta_vgood = float(np.clip(beta_vgood, beta_good, 0.25))
    beta_bad = float(np.clip(beta_bad, 0.005, 0.15))
    beta_vbad = float(np.clip(beta_vbad, beta_bad, 0.25))

    # Enforce that "very" is meaningfully stronger than non-very
    MIN_RATIO = 1.35  # 1.25–1.5 is reasonable; pick one and justify in thesis

    beta_vgood = max(beta_vgood, min(0.25, beta_good * MIN_RATIO))
    beta_vbad = max(beta_vbad, min(0.25, beta_bad * MIN_RATIO))

    qbank.mastery_params = MasteryUpdateParams(
        mastery_jump=0.95,
        beta_good=beta_good,
        beta_very_good=beta_vgood,
        beta_bad=beta_bad,
        beta_very_bad=beta_vbad,
        neutral_noise_std=neutral_noise_std,
    )

    # ----------------------------
    # NEW: per-topic beta multipliers (data-driven heterogeneity)
    # ----------------------------
    def _topic_strength(k: int) -> float:
        """Return a stable [-1, +1] strength score from tutor-only deltas for topic k.
        +1 => topic responds well to tutoring (good dominates), -1 => topic is harder / negative dominates.
        """
        vals = [dm for dm, aa in zip(delta_m[k], act_id[k]) if aa in (0, 1, 2, 3, 4)]
        if len(vals) < 200:
            return 0.0
        arr = np.asarray(vals, dtype=np.float32)
        # robust mean / std
        lo, hi = np.quantile(arr, [0.10, 0.90])
        arr = arr[(arr >= lo) & (arr <= hi)]
        mu = float(arr.mean()) if arr.size else float(np.mean(vals))
        sig = float(arr.std()) if arr.size else float(np.std(vals))
        sig = max(1e-6, sig)
        z = mu / sig
        # squash to [-1, 1]
        return float(np.tanh(z))

    # Multipliers: good updates stronger on "easy" topics, bad updates stronger on "hard" topics.
    # Keep within tight bounds so you don't destabilize training.
    good_lo, good_hi = 0.80, 1.20
    bad_lo, bad_hi = 0.80, 1.20
    alpha = 0.20  # sensitivity (0.10-0.30 recommended)

    topic_beta_good_mult = np.ones(cfg.n_topics, dtype=np.float32)
    topic_beta_very_good_mult = np.ones(cfg.n_topics, dtype=np.float32)
    topic_beta_bad_mult = np.ones(cfg.n_topics, dtype=np.float32)
    topic_beta_very_bad_mult = np.ones(cfg.n_topics, dtype=np.float32)

    for k in range(cfg.n_topics):
        s = _topic_strength(k)  # [-1, 1]
        # easy (s>0): increase positive betas, decrease negative betas
        topic_beta_good_mult[k] = np.clip(1.0 + alpha * s, good_lo, good_hi)
        topic_beta_very_good_mult[k] = np.clip(1.0 + alpha * s, good_lo, good_hi)
        topic_beta_bad_mult[k] = np.clip(1.0 - alpha * s, bad_lo, bad_hi)
        topic_beta_very_bad_mult[k] = np.clip(1.0 - alpha * s, bad_lo, bad_hi)


    # ----------------------------
    # Save bundle
    # ----------------------------
    bundle = KDDModelBundle(
        n_topics=cfg.n_topics,
        schema=schema,
        kc_to_topic=kc_to_topic,
        response_models=response_models,
        hints_models=hints_models if train_aux_models else None,
        time_models=time_models if train_aux_models else None,
        inc_models=inc_models if train_aux_models else None,
        quality_bank=qbank,
        one_hot_actions=True,
        ema_alpha=cfg.ema_alpha,
        tutee_outcome_topic=tutee_outcome_topic,
        tutee_outcome_leaf=tutee_outcome_leaf,
        topic_beta_good_mult=topic_beta_good_mult,
        topic_beta_very_good_mult=topic_beta_very_good_mult,
        topic_beta_bad_mult=topic_beta_bad_mult,
        topic_beta_very_bad_mult=topic_beta_very_bad_mult,
        # topic_tutor_action_gain=topic_tutor_action_gain,  # <-- add this

    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Saved KDDModelBundle to {out_path}")
    print(f"- n_topics: {cfg.n_topics}")
    print(f"- response models: {len(response_models)}")
    print(f"- aux models: {train_aux_models}")
    print(f"- quality bank topics: {len(qbank.bank)}")
    print(f"- mastery betas: good={qbank.mastery_params.beta_good:.4f}, very_good={qbank.mastery_params.beta_very_good:.4f}, bad={qbank.mastery_params.beta_bad:.4f}, very_bad={qbank.mastery_params.beta_very_bad:.4f}")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--n_topics", type=int, default=7)
    ap.add_argument("--max_depth", type=int, default=7)
    ap.add_argument("--min_leaf", type=int, default=50)
    ap.add_argument("--ema_alpha", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--kc_col", default="KC(Default)")
    ap.add_argument("--student_col", default="Anon Student Id")
    ap.add_argument("--order_cols", default="", help="comma-separated ordering columns")

    ap.add_argument("--cfa_col", default="Correct First Attempt")
    ap.add_argument("--duration_col", default="Step Duration (sec)")
    ap.add_argument("--hints_col", default="Hints")
    ap.add_argument("--incorrects_col", default="Incorrects")
    ap.add_argument("--corrects_col", default="Corrects")

    ap.add_argument("--kc_map_json", default=None)
    ap.add_argument("--cluster_max_kcs", type=int, default=250)
    ap.add_argument("--cluster_min_kc_freq", type=int, default=50)

    ap.add_argument("--no_aux", action="store_true")

    args = ap.parse_args()

    order_cols = [c.strip() for c in args.order_cols.split(",") if c.strip()] if args.order_cols else None

    cfg = TrainConfig(
        n_topics=int(args.n_topics),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_leaf),
        ema_alpha=float(args.ema_alpha),
        seed=int(args.seed),
    )

    train_bundle_from_kdd_csv(
        csv_path=args.csv,
        out_path=args.out,
        cfg=cfg,
        kc_col=args.kc_col,
        student_col=args.student_col,
        order_cols=order_cols,
        cfa_col=args.cfa_col,
        duration_col=args.duration_col,
        hints_col=args.hints_col,
        incorrects_col=args.incorrects_col,
        corrects_col=args.corrects_col,
        kc_map_json=args.kc_map_json,
        cluster_max_kcs=int(args.cluster_max_kcs),
        cluster_min_kc_freq=int(args.cluster_min_kc_freq),
        train_aux_models=(not args.no_aux),
    )


if __name__ == "__main__":
    main()
