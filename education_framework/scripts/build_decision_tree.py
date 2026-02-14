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
python -m education_framework.scripts.build_decision_tree   --csv education_framework/data/algebra_2005_2006_train.tx
t   --out education_framework/data/kdd_bundle.joblib   --n_topics 7 --max_depth 7 --min_leaf 50 --ema_alpha 0.2 --seed 0   --cluster_mode behavior -
-action_mode discover --n_actions 5   --action_features hints,incorrects,duration,opp   --action_min_cluster_frac 0.06 --action_label_mode schema   --leaf_shrinkage_prior 10   --topic_gain_mode balanced_primary   --quality_eps_frac 0.05 --quality_eps_min 0.0001

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
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from education_framework.utils.kdd_utils import group_rows_by_student_ordered, read_kdd_table

from education_framework.environment.learner_model import (
    KDDActionSchema,
    KDDModelBundle,
    KDDTrajectoryBuilder,
    LowLevelAction,
)
from education_framework.models.quality_tree_bank import QualityTreeBank, LeafQualityModel, MasteryUpdateParams, \
    QUALITY_LEVELS


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


def _action_signature(hints: int, incorrects: int, duration: float) -> np.ndarray:
    return np.asarray([
        float(hints) / 5.0,
        float(incorrects) / 5.0,
        float(duration) / 120.0,
    ], dtype=np.float32)


def fit_global_action_kmeans(rows_by_student, builder, cfg, seed: int):
    Z, T = [], []
    for tr in builder.iter_training_rows(
            rows_by_student,
            kc_col=cfg._kc_col,
            cfa_col=cfg._cfa_col,
            duration_col=cfg._duration_col,
            hints_col=cfg._hints_col,
            incorrects_col=cfg._incorrects_col,
    ):
        Z.append(_action_signature(tr.hints, tr.incorrects, tr.duration))
        T.append(int(tr.topic_id))

    Z = np.vstack(Z) if len(Z) else np.zeros((0, 3), dtype=np.float32)
    T = np.asarray(T, dtype=np.int32)

    # safety: drop non-finite
    if len(Z):
        mask = np.isfinite(Z).all(axis=1)
        Z = Z[mask]
        T = T[mask]

    scaler = StandardScaler(with_mean=True, with_std=True)
    Zs = scaler.fit_transform(Z) if len(Z) else Z

    K = int(cfg.cluster_k)
    if K != 5:
        raise ValueError(f"Option B (fixed) expects cluster_k=5, got {K}")

    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=int(seed),
        batch_size=4096,
        n_init="auto",
    )
    lab = km.fit_predict(Zs)

    # optional: warn if some topics don't contain enough actions (we do NOT shrink K)
    min_cnt = int(getattr(cfg, "cluster_min_count_per_action", 50))
    min_actions = int(getattr(cfg, "cluster_min_actions_per_topic", 3))
    bad = []
    for topic_id in range(int(builder.n_topics)):
        cnt = np.bincount(lab[T == topic_id], minlength=K)
        present = int(np.sum(cnt >= min_cnt))
        if present < min_actions:
            bad.append((topic_id, present, cnt.tolist()))
    if bad:
        print("[cluster][WARN] Some topics have low cluster coverage; continuing with fixed K=5.")
        for topic_id, present, cnts in bad[:10]:
            print(f"  topic={topic_id} present={present} counts={cnts}")

    # IMPORTANT: map modes -> real tutor actions 0..4 (bijective)
    # Our cluster features are [hints, incorrects, duration] so define feature_names accordingly.
    feature_names = ["hints", "incorrects", "duration"]
    mode_to_action = _assign_modes_to_actions_bijective_k5(km.cluster_centers_.astype(np.float32), feature_names)

    return scaler, km, mode_to_action


def build_kc_to_topic_by_clustering(
        df: pd.DataFrame,
        n_topics: int,
        kc_col: str,
        problem_col_candidates: Sequence[str] = ("Problem Name", "problem_id", "Problem Id", "Problem"),
        max_kcs: int = 250,
        min_kc_freq: int = 50,
        seed: int = 0,
        mode: str = "behavior",
        cfa_col: Optional[str] = None,
        duration_col: Optional[str] = None,
        hints_col: Optional[str] = None,
        incorrects_col: Optional[str] = None,
        corrects_col: Optional[str] = None,
        add_stds: bool = True,
        include_log_freq: bool = True,
) -> Dict[str, int]:
    """Cluster KCs into `n_topics` and return kc->topic mapping.
     Modes:
      - "cooccur" : KC x problem co-occurrence (previous default).
      - "behavior": KC behavior-signature from KDD columns (duration/hints/incorrects/CFA/...).
     Behavior mode is intended to produce *more heterogeneous* topics so that
    topic-specialized LL policies differ more meaningfully.
    """
    if kc_col not in df.columns:
        raise ValueError(f"KC column not found: {kc_col}")

    # problem_col = None
    # for c in problem_col_candidates:
    #     if c in df.columns:
    #         problem_col = c
    #         break
    # if problem_col is None:
    #     problem_col = "Step Name" if "Step Name" in df.columns else "__row_bucket__"
    #     if problem_col == "__row_bucket__":
    #         df = df.copy()
    #         df[problem_col] = (np.arange(len(df)) // 10).astype(int)

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

    kc_index: Dict[str, int] = {kc: i for i, kc in enumerate(sorted(df2["__kc__"].astype(str).unique().tolist()))}
    if len(kc_index) < n_topics:
        raise ValueError(
            f"Not enough KCs to cluster into n_topics={n_topics}. "
            f"Got {len(kc_index)} KCs after filtering (min_kc_freq={min_kc_freq}, max_kcs={max_kcs})."
        )

    # ----------------------------
    # Behavior-signature clustering
    # ----------------------------
    if str(mode).lower().strip() == "behavior":
        # pick available numeric columns
        candidates = [
            ("cfa", cfa_col),
            ("duration", duration_col),
            ("hints", hints_col),
            ("incorrects", incorrects_col),
            ("corrects", corrects_col),
        ]

        used: List[tuple[str, str]] = []
        for name, col in candidates:
            if col and (col in df2.columns):
                used.append((name, col))

        if len(used) > 0:
            # coerce to numeric
            for _, col in used:
                df2[col] = pd.to_numeric(df2[col], errors="coerce")

            g = df2.groupby("__kc__", sort=True)
            feat_parts: List[pd.DataFrame] = []

            for name, col in used:
                m = g[col].mean().rename(f"{name}_mean")
                feat_parts.append(m.to_frame())
                if add_stds:
                    s = g[col].std(ddof=0).rename(f"{name}_std")
                    feat_parts.append(s.to_frame())

            if include_log_freq:
                cnt = g.size().astype(float)
                feat_parts.append(np.log1p(cnt).rename("log_freq").to_frame())

            feats = pd.concat(feat_parts, axis=1)
            feats = feats.reindex(list(kc_index.keys()))

            X = feats.to_numpy(dtype=np.float32)

            # impute NaNs with column means (then standardize)
            col_mean = np.nanmean(X, axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
            inds = ~np.isfinite(X)

            if inds.any():
                X[inds] = np.take(col_mean, np.where(inds)[1])

            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True)
            sd[sd < 1e-6] = 1.0
            Xs = (X - mu) / sd

            km = KMeans(n_clusters=n_topics, random_state=seed, n_init=10)
            labels = km.fit_predict(Xs)
            return {kc: int(labels[i]) for kc, i in kc_index.items()}

    # if behavior columns missing, fall back to co-occurrence

    # ----------------------------
    # Co-occurrence clustering (previous behavior)
    # ----------------------------
    problem_col = None
    for c in problem_col_candidates:
        if c in df2.columns:
            problem_col = c
            break

    if problem_col is None:
        problem_col = "Step Name" if "Step Name" in df2.columns else "__row_bucket__"
        if problem_col == "__row_bucket__":
            df2 = df2.copy()
            df2[problem_col] = (np.arange(len(df2)) // 10).astype(int)

    problems = df2[problem_col].astype(str).tolist()
    kc_list = df2["__kc__"].astype(str).tolist()

    prob_index: Dict[str, int] = {}
    for p in problems:
        if p not in prob_index:
            prob_index[p] = len(prob_index)

    # kc_index: Dict[str, int] = {kc: i for i, kc in enumerate(sorted(set(kc_list)))}

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
    n_topics: int = 5
    max_depth: int = 6
    min_samples_leaf: int = 80
    ema_alpha: float = 0.2
    seed: int = 0

    schema_max_rows: int = 2_000_000

    # quality tree constraints
    min_leaf_action_count: int = 150

    # quality labeling: treat small deltas as neutral to avoid sign/semantic mismatch
    # quality_eps: float = 0.01  # mastery-delta noise floor for leaf-wise action scoring

    quality_eps: float = 0.01  # upper cap (kept for backward compat)
    quality_eps_min: float = 1e-3  # floor to avoid exact-zero issues
    quality_eps_frac: float = 0.10  # neutral_eps_k = min(cap, max(floor, frac*(q80-q20)))

    # build_decision_tree.py (where your build config lives)
    action_label_mode: str = "schema"  # "schema" | "cluster"
    cluster_k: int = 5
    cluster_min_actions_per_topic: int = 3
    cluster_min_count_per_action: int = 50  # per topic, for counting "present"
    leaf_shrinkage_prior: float = 20.0  # pseudo-count for per-leaf smoothing
    force_leaf_preference: bool = True  # if leaf becomes all-neutral, force best/worst
    force_good_if_none: bool = True
    force_bad_if_none: bool = True
    # Topic-level sparsifying / diversifying action gains (optional but recommended)
    topic_gain_mode: str = "balanced_primary"  # "linear" | "exp_z" | "balanced_primary"
    topic_gain_G: float = 0.25
    topic_gain_lo: float = 0.80
    topic_gain_hi: float = 1.20
    topic_gain_temp: float = 2.0

    # --- mastery-aware action effect estimation ---
    gain_mpre_max: float = 0.75  # rows above this mastery_pre do NOT drive per-topic gains
    cutoffs_mpre_max: float = 0.75  # rows above this do NOT drive topic cutoffs
    leaf_score_mpre_max: float = 0.80  # rows above this do NOT drive leaf action scores

    mpre_weight_power: float = 1.0  # weight ∝ (1 - mastery_pre)^power (set 0.0 to disable)
    residualize_by_mpre_bins: bool = True  # subtract baseline Δmastery per mastery bin

    # --- mastery-bin conditioned tutor gains (data-driven, defensible) ---
    use_gain_by_mpre_bins: bool = True
    mpre_bins: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)  # edges, len = n_bins + 1

    # For bin-conditioned gains we avoid "balanced_primary" forcing.
    # Use only "linear" or "exp_z" here.
    topic_gain_mode_bins: str = "linear"  # "linear" | "exp_z"
    min_gain_bin_action_count: int = 50   # min samples per (topic, bin, action) to trust bin mean

    apply_topic_gain_inside_bank: bool = False



def _mpre_bin_id(x: float, edges: np.ndarray) -> int:
    # edges: length n_bins+1
    if not np.isfinite(x):
        return 0
    n_bins = int(edges.size - 1)
    if n_bins <= 1:
        return 0
    # digitize against interior edges
    b = int(np.digitize([float(x)], edges[1:-1], right=False)[0])
    return int(np.clip(b, 0, n_bins - 1))

def _compute_topic_action_gains(means_by_topic: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    n_topics, n_actions = means_by_topic.shape
    LO = float(getattr(cfg, "topic_gain_lo", 0.80))
    HI = float(getattr(cfg, "topic_gain_hi", 1.20))
    mode = str(getattr(cfg, "topic_gain_mode", "linear"))

    if mode == "linear":
        G = float(getattr(cfg, "topic_gain_G", 0.25))
        gains = np.ones_like(means_by_topic, dtype=np.float32)
        for k in range(n_topics):
            m = means_by_topic[k]
            mu = float(np.mean(m))
            dev = m - mu
            denom = float(np.max(np.abs(dev))) if np.max(np.abs(dev)) > 1e-9 else 1.0
            rel = dev / denom
            gains[k] = np.clip(1.0 + G * rel, LO, HI)
        return gains

    if mode == "exp_z":
        temp = float(getattr(cfg, "topic_gain_temp", 2.0))
        gains = np.ones_like(means_by_topic, dtype=np.float32)
        for k in range(n_topics):
            m = means_by_topic[k]
            z = (m - m.mean()) / (m.std() + 1e-6)
            gains[k] = np.clip(np.exp(temp * z), LO, HI)
        return gains

    if mode == "balanced_primary":
        # Balanced assignment: ensure different topics get different "primary" best actions (as much as possible).
        adv = means_by_topic - means_by_topic.mean(axis=1, keepdims=True)
        cap = int(np.ceil(n_topics / float(n_actions)))
        used = np.zeros(n_actions, dtype=np.int32)
        primary = np.full(n_topics, -1, dtype=np.int32)

        order = np.argsort(-np.max(adv, axis=1))  # topics with strongest preference first
        for k in order:
            cands = np.argsort(-adv[k])  # best->worst
            for a in cands:
                if used[a] < cap:
                    primary[k] = int(a)
                    used[a] += 1
                    break
            if primary[k] < 0:
                primary[k] = int(np.argmax(adv[k]))

        # ensure each action is used at least once if possible
        for a in range(n_actions):
            if used[a] == 0:
                best_k = None
                best_loss = 1e9
                for k in range(n_topics):
                    cur = int(primary[k])
                    if used[cur] <= 1:
                        continue
                    loss = float(adv[k, cur] - adv[k, a])
                    if loss < best_loss:
                        best_loss = loss
                        best_k = k
                if best_k is not None:
                    used[int(primary[best_k])] -= 1
                    primary[best_k] = int(a)
                    used[a] += 1

        gains = np.ones_like(means_by_topic, dtype=np.float32)
        for k in range(n_topics):
            a_hi = int(primary[k])
            a_lo = int(np.argmin(adv[k]))
            gains[k, a_hi] = HI
            gains[k, a_lo] = LO
        return gains

    raise ValueError(f"Unknown topic_gain_mode={mode}")


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


def _topic_neutral_eps(cfg: TrainConfig, cutoffs: tuple[float, float, float, float]) -> float:
    q20, q40, q60, q80 = cutoffs
    span = float(q80) - float(q20)
    cap = float(cfg.quality_eps)
    floor = float(getattr(cfg, "quality_eps_min", 1e-4))
    frac = float(getattr(cfg, "quality_eps_frac", 0.05))
    return float(min(cap, max(floor, frac * max(0.0, span))))


def _scores_to_quality_global(action_ids, scores, *, cutoffs, neutral_eps=0.003):
    q20, q40, q60, q80 = cutoffs
    out = {}
    for a, s in zip(action_ids, scores):
        s = float(s)

        # KEY FIX: explicit neutral band
        if abs(s) <= neutral_eps:
            out[a] = "neutral"
        elif s <= q20:
            out[a] = "very_bad"
        elif s <= q40:
            out[a] = "bad"
        elif s <= q60:
            out[a] = "neutral"
        elif s <= q80:
            out[a] = "good"
        else:
            out[a] = "very_good"
    return out


def scores_to_quality_with_fallback(action_ids, scores, *, cutoffs, neutral_eps: float, topic_id: int, cfg, seed: int):
    q = _scores_to_quality_global(action_ids, scores, cutoffs=cutoffs, neutral_eps=neutral_eps)

    if getattr(cfg, "force_good_if_none", True):

        if not any(v in ("good", "very_good") for v in q.values()):
            best = int(action_ids[int(np.argmax(np.asarray(scores, dtype=np.float32)))])
            q[best] = "good"

    if getattr(cfg, "force_bad_if_none", True):

        if not any(v in ("bad", "very_bad") for v in q.values()):
            worst = int(action_ids[int(np.argmin(np.asarray(scores, dtype=np.float32)))])
            q[worst] = "bad"

    if not getattr(cfg, "force_leaf_preference", True):
        return q

    if all(v == "neutral" for v in q.values()):
        rng = np.random.RandomState(int(seed) + 1009 * int(topic_id))
        order = rng.permutation(len(action_ids))
        jitter = (order.astype(np.float32) - order.mean()) * 1e-6
        sj = np.asarray(scores, dtype=np.float32) + jitter
        best = int(np.argmax(sj))
        worst = int(np.argmin(sj))

        q[action_ids[best]] = "good"
        q[action_ids[worst]] = "bad"
    return q


def _detect_opp_col(df: pd.DataFrame) -> Optional[str]:
    cands = ["Opportunity(Default)", "Opportunity(KC)", "Opportunity", "Opportunity (Default)", "Opportunity (KC)"]
    for c in cands:
        if c in df.columns:
            return c
    return None


def _build_action_feature_matrix(
        df: pd.DataFrame,
        hints_col: str,
        incorrects_col: str,
        duration_col: str,
        opp_col: Optional[str],
        feature_names: List[str],
) -> np.ndarray:
    # numeric + light transforms for stability
    X_parts = []

    if "hints" in feature_names:
        h = pd.to_numeric(df[hints_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
        X_parts.append(np.log1p(h).reshape(-1, 1))

    if "incorrects" in feature_names:
        inc = pd.to_numeric(df[incorrects_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
        X_parts.append(np.log1p(inc).reshape(-1, 1))

    if "duration" in feature_names:
        dur = pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
        X_parts.append(np.log1p(np.clip(dur, 0.0, None)).reshape(-1, 1))

    if "opp" in feature_names and opp_col:
        opp = pd.to_numeric(df[opp_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
        X_parts.append(np.log1p(opp).reshape(-1, 1))

    if len(X_parts) == 0:
        raise ValueError("discover mode: no valid action_features found / available in df")

    X = np.concatenate(X_parts, axis=1).astype(np.float32)

    # standardize
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    Xs = (X - mu) / sd
    return Xs, mu.astype(np.float32), sd.astype(np.float32)


def _merge_tiny_clusters(labels: np.ndarray, centroids: np.ndarray, min_frac: float) -> np.ndarray:
    n = labels.size
    if n == 0:
        return labels
    counts = np.bincount(labels, minlength=centroids.shape[0]).astype(np.float32)
    frac = counts / float(n)

    tiny = np.where(frac < min_frac)[0]
    if tiny.size == 0:
        return labels

    # reassign tiny cluster points to nearest non-tiny centroid
    keep = np.where(frac >= min_frac)[0]
    if keep.size == 0:
        return labels  # degenerate; do nothing

    for c in tiny:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        # nearest keep centroid by euclidean distance
        d = ((centroids[keep] - centroids[c]) ** 2).sum(axis=1)
        new_c = int(keep[int(np.argmin(d))])
        labels[idx] = new_c

    # relabel to contiguous 0..K'-1
    uniq = np.unique(labels)
    remap = {int(u): i for i, u in enumerate(uniq)}
    labels2 = np.array([remap[int(x)] for x in labels], dtype=np.int32)
    return labels2


def _assign_discovered_modes_to_action_ids(centroids: np.ndarray, feature_names: List[str], n_actions: int) -> Dict[int, int]:
    """
    Robust bijection: assign each centroid to one of {quiz, hint, worked_example, remediation, review}
    using independent semantic scores, then resolve collisions by priority.
    """
    K = int(centroids.shape[0])
    if K != n_actions:
        raise ValueError(f"Expected K=5 centroids, got K={K}")

    fn = [f.lower().strip() for f in feature_names]
    def col(name: str):
        return fn.index(name) if name in fn else None

    idx_h = col("hints")
    idx_i = col("incorrects")
    idx_d = col("duration")
    idx_o = col("opp")

    H = centroids[:, idx_h] if idx_h is not None else np.zeros((K,), dtype=np.float32)
    I = centroids[:, idx_i] if idx_i is not None else np.zeros((K,), dtype=np.float32)
    D = centroids[:, idx_d] if idx_d is not None else np.zeros((K,), dtype=np.float32)
    O = centroids[:, idx_o] if idx_o is not None else np.zeros((K,), dtype=np.float32)

    # candidate picks (may collide)
    pick = {
        "review": int(np.argmax(O)) if idx_o is not None else int(np.argmax(D)),
        "quiz": int(np.argmin(H + I + D)),
        "hint": int(np.argmax(H)),
        "remediation": int(np.argmax(I)),
        "worked_example": int(np.argmax(D)),
    }

    # resolve collisions by priority: review > remediation > worked_example > hint > quiz
    priority = ["review", "remediation", "worked_example", "hint", "quiz"]
    used_modes = set()
    final = {}

    remaining = set(range(K))

    for name in priority:
        m = pick[name]
        if m in remaining:
            final[name] = m
            remaining.remove(m)

    # fill any missing roles with leftovers (stable)
    for name in priority:
        if name not in final:
            m = min(remaining)  # deterministic
            final[name] = m
            remaining.remove(m)

    mode_to_action = {
        final["quiz"]: 0,
        final["hint"]: 1,
        final["worked_example"]: 2,
        final["remediation"]: 3,
        final["review"]: 4,
    }

    used = sorted(mode_to_action.values())
    if used != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Non-bijective mapping produced: {mode_to_action}")
    return mode_to_action




def _assign_modes_to_actions_bijective_k5(
        centroids: np.ndarray,
        feature_names: List[str],
) -> Dict[int, int]:
    """
    Bijective map: 5 modes -> 5 tutor actions (0..4).
    Ensures coverage (no many-to-one collapse).

    Heuristic:
      - If 'opp' exists: highest opp centroid => REVIEW(4)
      - Remaining 4 modes ordered by "support index" (hints+incorrects+duration+0.5*opp):
          lowest => QUIZ(0)
          highest => REMEDIATION(3)
          second-highest => WORKED_EXAMPLE(2)
          remaining => HINT(1)
    """
    K = int(centroids.shape[0])
    if K != 5:
        raise ValueError(f"Expected K=5 centroids, got K={K}")

    w = np.ones((centroids.shape[1],), dtype=np.float32)
    if "opp" in feature_names:
        w[feature_names.index("opp")] = 0.5

    support = (centroids * w.reshape(1, -1)).sum(axis=1)

    mode_to_action: Dict[int, int] = {}
    remaining = list(range(K))

    # REVIEW from opp if available
    if "opp" in feature_names:
        opp_idx = feature_names.index("opp")
        review_mode = int(np.argmax(centroids[:, opp_idx]))
    else:
        # fallback: use highest duration as a weak "review-ish" proxy
        if "duration" in feature_names:
            dur_idx = feature_names.index("duration")
            review_mode = int(np.argmax(centroids[:, dur_idx]))
        else:
            review_mode = int(np.argmax(support))

    mode_to_action[review_mode] = 4  # REVIEW
    remaining = [m for m in remaining if m != review_mode]

    # order remaining by support
    ordered = sorted(remaining, key=lambda m: float(support[m]))
    # now len(ordered)=4
    mode_to_action[ordered[0]] = 0  # QUIZ
    mode_to_action[ordered[-1]] = 3  # REMEDIATION
    mode_to_action[ordered[-2]] = 2  # WORKED_EXAMPLE
    # the only leftover
    leftover = [m for m in ordered if m not in (ordered[0], ordered[-1], ordered[-2])]
    mode_to_action[leftover[0]] = 1  # HINT

    # sanity: bijection check
    used = sorted(mode_to_action.values())
    if used != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Non-bijective mapping produced: {mode_to_action}")

    return mode_to_action


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
        cluster_mode: str = "behavior",
        train_aux_models: bool = True,
        action_mode: str = 'classic',
        n_actions: int = 4,
        action_min_cluster_frac: float = 0.05,
        action_features: str = "hints,incorrects,duration,opp",
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
            mode=cluster_mode,
            cfa_col=cfa_col,
            duration_col=duration_col,
            hints_col=hints_col,
            incorrects_col=incorrects_col,
            corrects_col=corrects_col,
        )

    # rows by student
    if order_cols:
        order_cols = [c for c in order_cols if c in df.columns]
        if not order_cols:
            order_cols = None

    rows_by_student = group_rows_by_student_ordered(df, student_col=student_col, order_cols=order_cols)

    action_mode = str(action_mode).lower().strip()  # ensure you added arg to signature
    schema.action_mode = action_mode  # works if schema accepts dynamic attrs; ok if not slots

    if action_mode == "discover":
        feat_names = [s.strip().lower() for s in str(action_features).split(",") if s.strip()]
        opp_col = _detect_opp_col(df)

        Xs, mu, sd = _build_action_feature_matrix(
            df=df,
            hints_col=hints_col,
            incorrects_col=incorrects_col,
            duration_col=duration_col,
            opp_col=opp_col,
            feature_names=feat_names,
        )

        km = KMeans(n_clusters=int(n_actions), random_state=cfg.seed, n_init=10)
        labels = km.fit_predict(Xs).astype(np.int32)
        labels = _merge_tiny_clusters(labels, km.cluster_centers_.astype(np.float32),
                                      min_frac=float(action_min_cluster_frac))

        # recompute centroids after merge (simple mean per cluster)
        K = int(labels.max()) + 1
        centroids = np.zeros((K, Xs.shape[1]), dtype=np.float32)
        for k in range(K):
            idx = np.where(labels == k)[0]
            centroids[k] = Xs[idx].mean(axis=0) if idx.size else 0.0

        mode_to_action = _assign_discovered_modes_to_action_ids(centroids, feat_names, n_actions=K)

        # Attach to schema so trajectory builder can use it
        schema.discovered_action = {
            "feature_names": feat_names,
            "opp_col": opp_col or "",
            "mu": mu,
            "sd": sd,
            "centroids": centroids,
            "mode_to_action": mode_to_action,
        }

        print("[DISCOVER actions] K=", K, "features=", feat_names, "opp_col=", opp_col)
        # Optional: print cluster sizes
        counts = np.bincount(labels, minlength=K)
        print("[DISCOVER actions] cluster sizes:", counts.tolist())

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

    # stash column names into cfg so fit_global_action_kmeans can call iter_training_rows
    cfg._kc_col = kc_col
    cfg._cfa_col = cfa_col
    cfg._duration_col = duration_col
    cfg._hints_col = hints_col
    cfg._incorrects_col = incorrects_col

    action_labeler = None

    if str(cfg.action_label_mode).lower().strip() == "cluster":
        scaler, km, mode_to_action = fit_global_action_kmeans(rows_by_student, builder, cfg, cfg.seed)

        def action_labeler(*, row, topic_id, s_pre, x_state, cfa, hints, incorrects, duration):
            z = _action_signature(hints, incorrects, duration).reshape(1, -1)
            zs = scaler.transform(z)
            mode = int(km.predict(zs)[0])
            return int(mode_to_action[mode])  # <-- mapped to tutor action id 0..4

    for tr in builder.iter_training_rows(
            rows_by_student,
            kc_col=kc_col,
            cfa_col=cfa_col,
            duration_col=duration_col,
            hints_col=hints_col,
            incorrects_col=incorrects_col,
            action_labeler=action_labeler,  # <-- requires your learner_model.py patch
    ):
        ...

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

    # ============================
    # DEBUG: delta_mastery health by topic/action
    # ============================
    def _summ(arr):
        arr = np.asarray(arr, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p10": float(np.quantile(arr, 0.10)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
            "zero_frac": float(np.mean(np.isclose(arr, 0.0, atol=1e-12))),
            "neg_frac": float(np.mean(arr < 0.0)),
            "pos_frac": float(np.mean(arr > 0.0)),
        }

    print("\n=== DEBUG: delta_mastery stats (tutor actions only) ===")
    for k in range(cfg.n_topics):
        dmk = np.asarray(delta_m[k], dtype=np.float32)
        aak = np.asarray(act_id[k], dtype=np.int32)
        mpre = np.asarray(mastery_pre[k], dtype=np.float32)

        tutor_mask = (aak >= 0) & (aak <= 4) & np.isfinite(dmk)
        d_tutor = dmk[tutor_mask]
        mp_tutor = mpre[tutor_mask]

        print(f"\n[Topic {k}] total_n={len(dmk)} tutor_n={int(d_tutor.size)} "
              f"mpre_mean={float(np.nanmean(mp_tutor)) if mp_tutor.size else float('nan'):.3f}")

        s_all = _summ(d_tutor)
        print("  tutor_delta:", s_all)

        # per-action
        for a in range(5):
            mask_a = tutor_mask & (aak == a)
            s_a = _summ(dmk[mask_a])
            if s_a.get("n", 0) > 0:
                print(f"  a={a}: {s_a}")

        # delta conditioned on mastery_pre bins (to detect floor/clip effects)
        if d_tutor.size > 0:
            bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
            for lo, hi in bins:
                bm = (mp_tutor >= lo) & (mp_tutor < hi)
                s_b = _summ(d_tutor[bm])
                if s_b.get("n", 0) >= 200:
                    print(f"  mpre in [{lo:.2f},{hi:.2f}): {s_b}")

    # train response and aux models
    response_models: Dict[int, Any] = {}
    hints_models: Dict[int, Any] = {}
    inc_models: Dict[int, Any] = {}
    time_models: Dict[int, Any] = {}

    for k in range(cfg.n_topics):
        Xk = np.asarray(X_resp[k], dtype=np.float32)
        yk = np.asarray(y_cfa[k], dtype=np.int32)
        u = np.unique(yk)
        if u.size < 2:
            # Degenerate: only one label present -> predict_proba would be (n,1)
            # Skip storing a response model for this topic; runtime will fall back to mastery-based p_correct.
            continue
        if Xk.shape[0] < max(200, cfg.min_samples_leaf * 4):
            continue
        clf = DecisionTreeClassifier(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
                                     random_state=cfg.seed)
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
    topic_neutral_eps: Dict[int, float] = {}
    dm_used_by_topic: Dict[int, np.ndarray] = {}
    mpre_edges = np.asarray(getattr(cfg, "mpre_bins", (0.0, 0.25, 0.5, 0.75, 1.0)), dtype=np.float32)
    if mpre_edges.ndim != 1 or mpre_edges.size < 2:
        raise ValueError("mpre_bins must be a 1D sequence of edges with length >= 2.")
    n_mpre_bins = int(mpre_edges.size - 1)

    # topic -> bin -> action -> mean residual Δmastery
    topic_action_mean_bin: Dict[int, Dict[int, Dict[int, float]]] = {}


    for k in range(cfg.n_topics):
        topic_action_mean[k] = {}
        if not delta_m[k]:
            # fallback cutoffs if there is literally no data
            topic_cutoffs[k] = _compute_global_cutoffs([], neutral_eps=cfg.quality_eps)
            topic_neutral_eps[k] = _topic_neutral_eps(cfg, topic_cutoffs[k])
            continue
        dm_used_by_topic[k] = np.asarray(delta_m[k], dtype=np.float32)

        # # existing topic_action_mean computation (keep it)
        # for a in range(5):
        #     vals = [dm for dm, aa in zip(delta_m[k], act_id[k]) if aa == a]
        #     if vals:
        #         topic_action_mean[k][a] = float(np.mean(vals))

        mp = np.asarray(mastery_pre[k], dtype=np.float32)
        dm = np.asarray(delta_m[k], dtype=np.float32)
        aa = np.asarray(act_id[k], dtype=np.int32)

        tutor_mask = (aa >= 0) & (aa <= 4) & np.isfinite(dm) & np.isfinite(mp)

        # ---- baseline by mastery bin (removes ceiling/floor artifacts) ----
        dm_used = dm.copy()

        if cfg.residualize_by_mpre_bins:
            bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
            base = np.zeros(len(bins), dtype=np.float32)

            for b, (lo, hi) in enumerate(bins):
                m = tutor_mask & (mp >= lo) & (mp < hi) & (mp <= cfg.cutoffs_mpre_max)
                base[b] = float(dm[m].mean()) if int(m.sum()) >= 50 else 0.0

            for b, (lo, hi) in enumerate(bins):
                m = tutor_mask & (mp >= lo) & (mp < hi)
                dm_used[m] = dm[m] - base[b]  # residual Δmastery
        dm_used_by_topic[k] = dm_used.astype(np.float32, copy=False)
        # optional weights to emphasize low mastery
        w = (np.clip(1.0 - mp, 0.05, 1.0) ** float(cfg.mpre_weight_power))
        topic_action_mean[k] = {}

        # topic_action_mean: only from learning region (mpre <= gain_mpre_max)
        for a in range(5):
            m = tutor_mask & (aa == a) & (mp <= cfg.gain_mpre_max)
            if int(m.sum()) > 0:
                # weighted mean; if you want unweighted, just use dm_used[m].mean()
                topic_action_mean[k][a] = float(np.average(dm_used[m], weights=w[m]))

        topic_action_mean_bin[k] = {}
        for b in range(n_mpre_bins):
            lo = float(mpre_edges[b])
            hi = float(mpre_edges[b + 1])
            topic_action_mean_bin[k][b] = {}
            for a in range(5):
                mb = tutor_mask & (aa == a) & (mp >= lo) & (mp < hi) & (mp <= cfg.gain_mpre_max)
                if int(mb.sum()) > 0:
                    topic_action_mean_bin[k][b][a] = float(np.average(dm_used[mb], weights=w[mb]))
                else:
                    # fallback to topic mean (keeps heterogeneity without inventing bin effects)
                    topic_action_mean_bin[k][b][a] = float(topic_action_mean[k].get(a, 0.0))


        # topic_cutoffs: only from learning region (mpre <= cutoffs_mpre_max)
        m_cut = tutor_mask & (mp <= cfg.cutoffs_mpre_max)
        all_tutor_deltas = dm_used[m_cut].tolist()
        topic_cutoffs[k] = _compute_global_cutoffs(all_tutor_deltas, neutral_eps=cfg.quality_eps)
        topic_neutral_eps[k] = _topic_neutral_eps(cfg, topic_cutoffs[
            k])  # keep your existing line :contentReference[oaicite:6]{index=6}

        # keep tutee fallbacks
        for a_tutee in (
        int(LowLevelAction.TUTEE_QUIZ), int(LowLevelAction.TUTEE_EXPLAIN), int(LowLevelAction.TUTEE_FIX)):
            topic_action_mean[k][a_tutee] = 0.0

        # NEW: global distribution over tutor actions for this topic
        # all_tutor_deltas = [dm for dm, aa in zip(delta_m[k], act_id[k]) if aa in (0, 1, 2, 3, 4)]
        # topic_cutoffs[k] = _compute_global_cutoffs(all_tutor_deltas, neutral_eps=cfg.quality_eps)

        # ✅ ADD THIS LINE (this is what you’re missing)
        # topic_neutral_eps[k] = _topic_neutral_eps(cfg, topic_cutoffs[k])

        q20, q40, q60, q80 = topic_cutoffs[k]
        span = q80 - q20
        print(f"[DEBUG cutoffs] topic={k} q20={q20:+.6f} q40={q40:+.6f} q60={q60:+.6f} q80={q80:+.6f} span={span:.6f}")

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
    topic_tutor_action_gain = np.ones((cfg.n_topics, 5), dtype=np.float32)
    #
    # G = 0.25  # strength (0.10..0.20 recommended)
    # LO, HI = 0.80, 1.20
    #
    # --- NEW: per-topic per-action gain multipliers (data-driven) ---
    means_by_topic = np.zeros((cfg.n_topics, 5), dtype=np.float32)
    means_by_topic_bin = np.zeros((cfg.n_topics, n_mpre_bins, 5), dtype=np.float32)
    counts_by_topic_bin_action = np.zeros((cfg.n_topics, n_mpre_bins, 5), dtype=np.int32)

    for k in range(cfg.n_topics):
        mp = np.asarray(mastery_pre[k], dtype=np.float32)
        dm_used = np.asarray(dm_used_by_topic.get(k, np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        aa = np.asarray(act_id[k], dtype=np.int32)
        tutor_mask = (aa >= 0) & (aa <= 4) & np.isfinite(dm_used) & np.isfinite(mp)

        # overall means (kept for backward compatibility + logging)
        for a in range(5):
            m = tutor_mask & (aa == a) & (mp <= cfg.gain_mpre_max)
            means_by_topic[k, a] = float(dm_used[m].mean()) if int(m.sum()) else float(
                topic_action_mean.get(k, {}).get(a, 0.0)
            )

        # bin-conditioned means (defensible state dependence)
        for b in range(n_mpre_bins):
            lo = float(mpre_edges[b])
            hi = float(mpre_edges[b + 1])
            for a in range(5):
                m = tutor_mask & (aa == a) & (mp >= lo) & (mp < hi) & (mp <= cfg.gain_mpre_max)
                c = int(m.sum())
                counts_by_topic_bin_action[k, b, a] = c
                if c >= int(getattr(cfg, "min_gain_bin_action_count", 75)):
                    means_by_topic_bin[k, b, a] = float(dm_used[m].mean())
                else:
                    # conservative fallback: do not fabricate bin effects
                    means_by_topic_bin[k, b, a] = float(means_by_topic[k, a])

    # base (topic-only) gains as before
    topic_tutor_action_gain = _compute_topic_action_gains(means_by_topic, cfg)

    # bin-conditioned gains (no forced diversity)
    topic_tutor_action_gain_by_mpre_bin = None
    if bool(getattr(cfg, "use_gain_by_mpre_bins", True)) and n_mpre_bins > 1:
        tmp_cfg = cfg
        # temporarily use bin-specific mode (linear/exp_z) to avoid balanced_primary forcing
        setattr(tmp_cfg, "topic_gain_mode", str(getattr(cfg, "topic_gain_mode_bins", "linear")))
        topic_tutor_action_gain_by_mpre_bin = np.ones((cfg.n_topics, n_mpre_bins, 5), dtype=np.float32)
        for b in range(n_mpre_bins):
            topic_tutor_action_gain_by_mpre_bin[:, b, :] = _compute_topic_action_gains(means_by_topic_bin[:, b, :], tmp_cfg)
        # restore original topic_gain_mode (important if later code assumes it)
        setattr(tmp_cfg, "topic_gain_mode", str(getattr(cfg, "topic_gain_mode", "linear")))

    if topic_tutor_action_gain_by_mpre_bin is not None:
        thr = int(getattr(cfg, "min_gain_bin_action_count", 75))
        print("[DEBUG bin action counts + trust + primary]:")
        for k in range(cfg.n_topics):
            for b in range(n_mpre_bins):
                lo = float(mpre_edges[b]); hi = float(mpre_edges[b+1])
                cnts = counts_by_topic_bin_action[k, b].tolist()
                trusted = [int(c >= thr) for c in cnts]
                prim = int(np.argmax(topic_tutor_action_gain_by_mpre_bin[k, b]))
                print(f"  topic {k} bin {b} [{lo:.2f},{hi:.2f}): counts={cnts} trusted={trusted} primary={prim}")

    if topic_tutor_action_gain_by_mpre_bin is not None:
        print("[DEBUG gain primary action per topic per mastery-bin]:")
        for k in range(cfg.n_topics):
            per_bin = {}
            for b in range(n_mpre_bins):
                per_bin[b] = int(np.argmax(topic_tutor_action_gain_by_mpre_bin[k, b]))
            print(f"  topic {k}:", per_bin)


    print("[DEBUG gain primary action per topic]:",
          {k: int(np.argmax(topic_tutor_action_gain[k])) for k in range(cfg.n_topics)})
    print("[DEBUG gain matrix]:")
    for k in range(cfg.n_topics):
        print(f"  topic {k} gains:", np.round(topic_tutor_action_gain[k], 3))

    # mu = float(np.mean(means))
    # adv = np.asarray([m - mu for m in means], dtype=np.float32)
    # denom = float(np.max(np.abs(adv))) + 1e-8  # scale by max deviation
    # rel = adv / denom  # in [-1,1]
    # gains = 1.0 + G * rel
    # gains = np.clip(gains, LO, HI)
    # topic_tutor_action_gain[k, :] = gains

    # print(topic_tutor_action_gain.shape)
    for k in range(cfg.n_topics):
        dmk_used = dm_used_by_topic[k]
        # NOTE: we no longer compute any tutee outcome tables from the dataset.
        Xs = np.asarray(X_state[k], dtype=np.float32)
        yk = np.asarray(y_cfa[k], dtype=np.int32)
        if Xs.shape[0] < max(500, cfg.min_samples_leaf * 8):
            continue

        route = DecisionTreeClassifier(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
                                       random_state=cfg.seed)
        route.fit(Xs, yk)
        leaf_ids = route.apply(Xs).astype(int)

        # pre-index samples per leaf
        idx_by_leaf: Dict[int, np.ndarray] = {}
        for leaf in np.unique(leaf_ids):
            idx_by_leaf[int(leaf)] = np.where(leaf_ids == leaf)[0]

        leaf_to_action_score: Dict[int, Dict[int, float]] = {}
        leaf_to_action_quality: Dict[int, Dict[int, str]] = {}

        print(f"\n=== DEBUG: leaf action-score health (topic {k}) ===")

        for leaf, idxs in idx_by_leaf.items():
            # # action scores for 0..4 (tutor-labelled)
            # scores: Dict[int, float] = {}
            # for a in range(5):
            #     vals = [delta_m[k][i] for i in idxs if act_id[k][i] == a]
            #     if len(vals) >= cfg.min_leaf_action_count:
            #         scores[a] = float(np.mean(vals))
            prior = float(cfg.leaf_shrinkage_prior)

            scores: Dict[int, float] = {}

            idxs2 = [i for i in idxs if float(mastery_pre[k][i]) <= cfg.leaf_score_mpre_max]

            # tutor actions 0..4
            for a in range(5):
                vals = [float(dmk_used[i]) for i in idxs2 if act_id[k][i] == a]
                cnt = len(vals)
                ssum = float(np.sum(vals)) if cnt else 0.0
                # topic_mean = float(topic_action_mean.get(k, {}).get(a, 0.0))
                # shrink toward the topic mean of the leaf's typical mastery region (bin-consistent smoothing)
                if len(idxs2) > 0 and k in topic_action_mean_bin:
                    leaf_meds = float(np.median([float(mastery_pre[k][i]) for i in idxs2]))
                    b_leaf = _mpre_bin_id(leaf_meds, mpre_edges)
                    topic_mean = float(topic_action_mean_bin[k].get(b_leaf, {}).get(a, topic_action_mean.get(k, {}).get(a, 0.0)))
                else:
                    topic_mean = float(topic_action_mean.get(k, {}).get(a, 0.0))

                # scores[a] = (ssum + prior * topic_mean) / (cnt + prior)
                prior0 = float(cfg.leaf_shrinkage_prior)
                # shrink more when cnt is tiny; shrink less when cnt is large
                prior_eff = prior0 * (float(cfg.min_samples_leaf) / float(cnt + cfg.min_samples_leaf))
                scores[a] = (ssum + prior_eff * topic_mean) / (cnt + prior_eff)

            # DEBUG: how often leaf/action deltas are effectively zero
            tutor_scores = [scores.get(a, None) for a in range(5)]
            tutor_scores_filled = [float(s) if s is not None else float('nan') for s in tutor_scores]
            tutor_scores_arr = np.asarray(tutor_scores_filled, dtype=np.float32)
            finite = np.isfinite(tutor_scores_arr)
            if finite.any():
                zfrac = float(np.mean(np.isclose(tutor_scores_arr[finite], 0.0, atol=1e-12)))
                if zfrac > 0.9:
                    # print only suspicious leaves to avoid spam
                    print(f"  leaf={leaf} n={len(idxs)} tutor_score_zero_frac={zfrac:.2f} "
                          f"scores={[(a, scores.get(a, None)) for a in range(5)]}")

            # NOTE: KDD has no explicit tutee interactions; we do not infer tutee effects from data.
            # Keep neutral (0) scores for the three tutee actions in the QualityTreeBank.
            scores[int(LowLevelAction.TUTEE_QUIZ)] = 0.0
            scores[int(LowLevelAction.TUTEE_EXPLAIN)] = 0.0
            scores[int(LowLevelAction.TUTEE_FIX)] = 0.0

            # fill missing actions with topic-level means (or 0)
            # for a in range(8):
            #     if a in scores:
            #         continue
            #     scores[a] = float(topic_action_mean.get(k, {}).get(a, 0.0))

            # store scores for all actions (unchanged)
            action_ids = list(range(8))
            leaf_to_action_score[leaf] = {a: float(scores[a]) for a in action_ids}

            if bool(getattr(cfg, "apply_topic_gain_inside_bank", False)):
                for a in range(5):
                    scores[a] = float(scores[a]) * float(topic_tutor_action_gain[k, a])

            # Tutor qualities come from KDD-derived leaf scores.
            tutor_ids = [0, 1, 2, 3, 4]
            tutor_scores = [float(scores[a]) for a in tutor_ids]
            tutor_q = scores_to_quality_with_fallback(
                tutor_ids,
                tutor_scores,
                cutoffs=topic_cutoffs[k],
                # neutral_eps=cfg.quality_eps,
                neutral_eps=topic_neutral_eps[k],
                topic_id=k,
                cfg=cfg,
                seed=cfg.seed,
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
    beta_bad = min(beta_bad, 2.0 * beta_good)
    beta_vbad = min(beta_vbad, 2.0 * beta_vgood)

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
    for k, m in sorted(response_models.items()):
        cls = getattr(m, "classes_", None)
        print(k, cls, "n_classes=", None if cls is None else len(cls))
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
        topic_tutor_action_gain=topic_tutor_action_gain,
        # topic_tutor_action_gain=topic_tutor_action_gain,
        topic_tutor_action_gain_by_mpre_bin=topic_tutor_action_gain_by_mpre_bin,
        tutor_gain_mpre_bins=mpre_edges,

    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Saved KDDModelBundle to {out_path}")
    print(f"- n_topics: {cfg.n_topics}")
    print(f"- response models: {len(response_models)}")
    print(f"- aux models: {train_aux_models}")
    print(f"- quality bank topics: {len(qbank.bank)}")
    print(
        f"- mastery betas: good={qbank.mastery_params.beta_good:.4f}, very_good={qbank.mastery_params.beta_very_good:.4f}, bad={qbank.mastery_params.beta_bad:.4f}, very_bad={qbank.mastery_params.beta_very_bad:.4f}")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--n_topics", type=int, default=7)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--min_leaf", type=int, default=150)
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
    ap.add_argument("--cluster_max_kcs", type=int, default=300)
    ap.add_argument("--cluster_min_kc_freq", type=int, default=250)

    ap.add_argument("--no_aux", action="store_true")
    ap.add_argument("--cluster_mode", default="behavior", choices=["behavior", "cooccur"],
                    help="KC->topic clustering mode: behavior-signature (recommended) or KC×problem co-occurrence")
    ap.add_argument("--action_mode",
                    choices=["classic", "kdd_oriented", "discover"],
                    default="kdd_oriented",
                    help="How to label tutor actions from KDD rows. 'discover' learns K latent modes from data.")

    ap.add_argument("--n_actions",
                    type=int,
                    default=4,
                    help="Number of discovered latent actions when action_mode=discover (3 or 4 recommended).")

    ap.add_argument("--action_min_cluster_frac",
                    type=float,
                    default=0.05,
                    help="Minimum cluster fraction; small clusters will be merged into nearest centroid (discover mode).")

    ap.add_argument("--action_features",
                    type=str,
                    default="hints,incorrects,duration,opp",
                    help="Comma-separated features for discover mode. Supported: hints,incorrects,duration,opp.")
    ap.add_argument("--action_label_mode", default="schema", choices=["schema", "cluster"])
    ap.add_argument("--cluster_k", type=int, default=5)
    ap.add_argument("--cluster_min_actions_per_topic", type=int, default=3)
    ap.add_argument("--cluster_min_count_per_action", type=int, default=50)
    ap.add_argument("--leaf_shrinkage_prior", type=float, default=20.0)
    ap.add_argument("--no_force_leaf_preference", action="store_true",
                    help="Disable forcing best/worst when leaf would be degenerate.")
    ap.add_argument("--no_force_good_if_none", action="store_true",
                    help="Disable forcing a GOOD action in a leaf if none exists.")
    ap.add_argument("--no_force_bad_if_none", action="store_true",
                    help="Disable forcing a BAD action in a leaf if none exists.")

    ap.add_argument("--topic_gain_mode", type=str, default="balanced_primary",
                    choices=["linear", "exp_z", "balanced_primary"])
    ap.add_argument("--topic_gain_G", type=float, default=0.25)
    ap.add_argument("--topic_gain_lo", type=float, default=0.80)
    ap.add_argument("--topic_gain_hi", type=float, default=1.20)
    ap.add_argument("--topic_gain_temp", type=float, default=2.0)
    ap.add_argument("--quality_eps_min", type=float, default=1e-3)
    ap.add_argument("--quality_eps_frac", type=float, default=0.10)
    ap.add_argument("--no_gain_by_mpre_bins", action="store_true",
                    help="Disable topic×mastery-bin tutor action gains.")
    ap.add_argument("--topic_gain_mode_bins", type=str, default="linear", choices=["linear", "exp_z"],
                    help="Gain mode for mastery-bin gains (avoid balanced_primary here).")
    ap.add_argument("--min_gain_bin_action_count", type=int, default=75,
                    help="Min count for (topic, bin, action) when estimating bin-conditioned gains.")
    ap.add_argument("--apply_topic_gain_inside_bank", action="store_true",
                    help="Multiply leaf action scores by topic gain inside the quality bank (not recommended if gains are also applied at runtime).")

    args = ap.parse_args()

    order_cols = [c.strip() for c in args.order_cols.split(",") if c.strip()] if args.order_cols else None

    cfg = TrainConfig(
        n_topics=int(args.n_topics),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_leaf),
        ema_alpha=float(args.ema_alpha),
        seed=int(args.seed),
        action_label_mode=str(args.action_label_mode),
        cluster_k=int(args.cluster_k),
        cluster_min_actions_per_topic=int(args.cluster_min_actions_per_topic),
        cluster_min_count_per_action=int(args.cluster_min_count_per_action),
        leaf_shrinkage_prior=float(args.leaf_shrinkage_prior),
        # force_leaf_preference=bool(args.force_leaf_preference),
        force_leaf_preference=(not bool(args.no_force_leaf_preference)),
        force_good_if_none=(not bool(args.no_force_good_if_none)),
        force_bad_if_none=(not bool(args.no_force_bad_if_none)),
        topic_gain_mode=str(args.topic_gain_mode),
        topic_gain_G=float(args.topic_gain_G),
        topic_gain_lo=float(args.topic_gain_lo),
        topic_gain_hi=float(args.topic_gain_hi),
        topic_gain_temp=float(args.topic_gain_temp),
        quality_eps_min=float(args.quality_eps_min),
        quality_eps_frac=float(args.quality_eps_frac),
        use_gain_by_mpre_bins=(not bool(args.no_gain_by_mpre_bins)),
        topic_gain_mode_bins=str(args.topic_gain_mode_bins),
        min_gain_bin_action_count=int(args.min_gain_bin_action_count),
        apply_topic_gain_inside_bank=bool(args.apply_topic_gain_inside_bank),

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
        cluster_mode=str(args.cluster_mode),
        train_aux_models=(not args.no_aux),
        action_mode=str(args.action_mode),
        n_actions=int(args.n_actions),
        action_min_cluster_frac=float(args.action_min_cluster_frac),
        action_features=str(args.action_features),

    )


if __name__ == "__main__":
    main()
