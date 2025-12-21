# build_decision_tree.py
"""
Train KDD-based response + transition decision-tree models and save a KDDModelBundle.

This script produces:
- schema (quantile thresholds) for action labeling + generation_mode proxy
- kc_to_topic mapping (8 macro-topics)
- per-topic response model: DecisionTreeClassifier(max_depth=7)
- per-topic transition model: DecisionTreeRegressor(max_depth=7) predicting delta vector (dim=5)
- optional per-topic auxiliary models for hints, incorrects, duration (regressors)

Output:
- joblib file containing KDDModelBundle (see learner_model.py)

Usage (example):
  python build_decision_tree.py --csv path/to/kdd.csv --out kdd_bundle.joblib \
      --kc_col "KC(Default)" --student_col "Anon Student Id"

Notes:
- If you already have an 8-topic mapping, pass --kc_map_json.
- Otherwise, the script clusters KCs into 8 macro-topics using co-occurrence + KMeans.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from education_framework.utils.kdd_utils import group_rows_by_student_ordered
from education_framework.utils.kdd_utils import read_kdd_table



import joblib

from education_framework.environment.learner_model import (
    KDDActionSchema,
    KDDModelBundle,
    KDDTrajectoryBuilder,
    group_rows_by_student,
)


# ----------------------------
# KC -> Topic mapping
# ----------------------------
def filter_finite_xy(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[mask], Y[mask]

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
        raise ValueError("kc_map_json must contain a JSON object mapping KC string -> topic_id.")
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
    """
    Data-driven mapping: KCs clustered into n_topics using co-occurrence across problems.

    Steps:
    1) Extract primary KC for each row.
    2) Keep top KCs by frequency.
    3) Build KC-by-problem incidence matrix (binary).
    4) Cluster KCs with KMeans(n_clusters=n_topics) on normalized incidence vectors.

    This is transparent and reproducible.
    """
    if kc_col not in df.columns:
        raise ValueError(f"KC column not found: {kc_col}")

    problem_col = None
    for c in problem_col_candidates:
        if c in df.columns:
            problem_col = c
            break
    if problem_col is None:
        # fallback: use "Step Name" if present, else row index bucket
        if "Step Name" in df.columns:
            problem_col = "Step Name"
        else:
            problem_col = "__row_bucket__"
            df = df.copy()
            df[problem_col] = (np.arange(len(df)) // 10_000).astype(int)

    kcs = df[kc_col].map(_extract_primary_kc)
    df2 = df.copy()
    df2["__kc__"] = kcs
    df2 = df2.dropna(subset=["__kc__"])
    kc_counts = df2["__kc__"].value_counts()
    kc_keep = kc_counts[kc_counts >= min_kc_freq].head(max_kcs).index.tolist()
    if len(kc_keep) < n_topics:
        raise RuntimeError(f"Not enough KCs after filtering to form {n_topics} topics (kept={len(kc_keep)}).")

    df2 = df2[df2["__kc__"].isin(kc_keep)]

    # Map problems to ids
    problems = df2[problem_col].astype(str)
    prob_ids, prob_uniques = pd.factorize(problems, sort=False)
    kc_ids, kc_uniques = pd.factorize(df2["__kc__"].astype(str), sort=False)

    n_kc = len(kc_uniques)
    n_prob = len(prob_uniques)

    # Sparse build: incidence[kc, prob] = 1 if appeared
    # We'll build as dense float32 for simplicity (n_kc <= 250)
    incidence = np.zeros((n_kc, n_prob), dtype=np.float32)
    incidence[kc_ids, prob_ids] = 1.0

    # Normalize rows (avoid bias toward frequent KCs)
    row_norm = np.linalg.norm(incidence, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    X = incidence / row_norm

    km = KMeans(n_clusters=n_topics, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    mapping: Dict[str, int] = {}
    for kc, lab in zip(kc_uniques.tolist(), labels.tolist()):
        mapping[str(kc)] = int(lab)

    return mapping


# ----------------------------
# Training per-topic trees
# ----------------------------

@dataclass
class TrainConfig:
    n_topics: int = 8
    max_depth: int = 7
    min_samples_leaf: int = 50
    ema_alpha: float = 0.2
    one_hot_actions: bool = True
    seed: int = 0

    # for action schema fitting
    schema_max_rows: int = 2_000_000


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
    # --- fit action schema quantiles ---
    # Use dict-record iteration to avoid copying large arrays.
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
    # --- build KC->topic mapping ---
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

    # --- group rows by student and order ---
    records = df.to_dict(orient="records")
    # Choose order cols that exist
    if order_cols:
        order_cols = [c for c in order_cols if c in df.columns]
        if not order_cols:
            order_cols = None
    rows_by_student = group_rows_by_student_ordered(
        df,
        student_col=student_col,
        order_cols=order_cols,
    )

    # --- build supervised rows ---
    builder = KDDTrajectoryBuilder(
        n_topics=cfg.n_topics,
        kc_to_topic=kc_to_topic,
        schema=schema,
        ema_alpha=cfg.ema_alpha,
        one_hot_actions=cfg.one_hot_actions,
        seed=cfg.seed,
    )

    X_resp_by_topic: Dict[int, List[np.ndarray]] = {k: [] for k in range(cfg.n_topics)}
    y_cfa_by_topic: Dict[int, List[int]] = {k: [] for k in range(cfg.n_topics)}
    X_trans_by_topic: Dict[int, List[np.ndarray]] = {k: [] for k in range(cfg.n_topics)}
    Y_delta_by_topic: Dict[int, List[np.ndarray]] = {k: [] for k in range(cfg.n_topics)}

    # Aux targets
    aux_hints_by_topic: Dict[int, List[float]] = {k: [] for k in range(cfg.n_topics)}
    aux_inc_by_topic: Dict[int, List[float]] = {k: [] for k in range(cfg.n_topics)}
    aux_time_by_topic: Dict[int, List[float]] = {k: [] for k in range(cfg.n_topics)}

    for tr in builder.iter_training_rows(
        rows_by_student,
        kc_col=kc_col,
        cfa_col=cfa_col,
        duration_col=duration_col,
        hints_col=hints_col,
        incorrects_col=incorrects_col,
    ):
        k = int(tr.topic_id)
        X_resp_by_topic[k].append(tr.x_resp)
        y_cfa_by_topic[k].append(int(tr.cfa))
        X_trans_by_topic[k].append(tr.x_trans)
        Y_delta_by_topic[k].append(tr.delta)

        # aux targets are embedded in x_trans outcome features; but for separate aux models,
        # store raw values from those feature slots.
        # x_trans layout ends with [cfa, hints_norm, incorrects_norm, duration_norm]
        # We'll train aux on the same x_resp features (pre-outcome) so it can be predicted before sampling outcome.
        if train_aux_models:
            # Use normalized values times typical norms would require cfg; we just keep normalized targets.
            hints_norm = float(tr.x_trans[-3])
            inc_norm = float(tr.x_trans[-2])
            time_norm = float(tr.x_trans[-1])
            aux_hints_by_topic[k].append(float(tr.hints))
            aux_inc_by_topic[k].append(float(tr.incorrects))
            aux_time_by_topic[k].append(float(tr.duration))

    response_models: Dict[int, Any] = {}
    transition_models: Dict[int, Any] = {}
    hints_models: Dict[int, Any] = {}
    inc_models: Dict[int, Any] = {}
    time_models: Dict[int, Any] = {}

    for k in range(cfg.n_topics):
        if len(X_resp_by_topic[k]) < 1_000:
            # Still build, but warn via simple print
            print(f"[WARN] Topic {k}: low sample count for training ({len(X_resp_by_topic[k])}).")

        Xr = np.vstack(X_resp_by_topic[k]).astype(np.float32)
        yr = np.asarray(y_cfa_by_topic[k], dtype=np.int32)

        mask = np.isfinite(Xr).all(axis=1)
        Xr = Xr[mask]
        yr = yr[mask]

        # If you train aux models, apply SAME mask to aux targets
        if train_aux_models and k in hints_models:  # or just check list lengths
            pass

        Xt = np.vstack(X_trans_by_topic[k]).astype(np.float32)
        Yd = np.vstack(Y_delta_by_topic[k]).astype(np.float32)

        Xt, Yd = filter_finite_xy(Xt, Yd)

        # Response: classifier
        clf = DecisionTreeClassifier(
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
        )
        clf.fit(Xr, yr)
        response_models[k] = clf

        print("[DEBUG] NaNs per delta-dim:", np.isnan(Yd).sum(axis=0).tolist())
        if Xt.shape[0] < 200:  # pick a small floor so you don't fit garbage
            print(f"[WARN] Topic {k}: too few finite transition samples after filtering ({Xt.shape[0]}). Skipping.")
        else:
            reg = DecisionTreeRegressor(
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=cfg.seed,
            )
            reg.fit(Xt, Yd)
            transition_models[k] = reg

        if train_aux_models and len(aux_hints_by_topic[k]) == len(X_resp_by_topic[k]):
            # Train aux regressors to predict expected hints/inc/time_norm from x_resp
            y_h = np.asarray(aux_hints_by_topic[k], dtype=np.float32)
            y_i = np.asarray(aux_inc_by_topic[k], dtype=np.float32)
            y_t = np.asarray(aux_time_by_topic[k], dtype=np.float32)

            reg_h = DecisionTreeRegressor(
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=cfg.seed,
            )
            reg_i = DecisionTreeRegressor(
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=cfg.seed,
            )
            reg_t = DecisionTreeRegressor(
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=cfg.seed,
            )
            reg_h.fit(Xr, y_h)
            reg_i.fit(Xr, y_i)
            reg_t.fit(Xr, y_t)

            hints_models[k] = reg_h
            inc_models[k] = reg_i
            time_models[k] = reg_t

    bundle = KDDModelBundle(
        n_topics=cfg.n_topics,
        schema=schema,
        kc_to_topic=kc_to_topic,
        response_models=response_models,
        transition_models=transition_models,
        hints_models=(hints_models if train_aux_models else None),
        inc_models=(inc_models if train_aux_models else None),
        time_models=(time_models if train_aux_models else None),
        one_hot_actions=cfg.one_hot_actions,
        ema_alpha=cfg.ema_alpha,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"[OK] Saved KDDModelBundle to: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to KDD Algebra CSV")
    ap.add_argument("--out", required=True, help="Output .joblib bundle path")
    ap.add_argument("--kc_col", default="KC(Default)")
    ap.add_argument("--student_col", default="Anon Student Id")
    ap.add_argument("--order_cols", default="", help="Comma-separated order columns (e.g., 'Row,Time')")
    ap.add_argument("--cfa_col", default="Correct First Attempt")
    ap.add_argument("--duration_col", default="Step Duration (sec)")
    ap.add_argument("--hints_col", default="Hints")
    ap.add_argument("--incorrects_col", default="Incorrects")
    ap.add_argument("--corrects_col", default="Corrects")
    ap.add_argument("--kc_map_json", default="", help="Optional KC->topic mapping JSON")
    ap.add_argument("--n_topics", type=int, default=8)
    ap.add_argument("--max_depth", type=int, default=7)
    ap.add_argument("--min_samples_leaf", type=int, default=50)
    ap.add_argument("--ema_alpha", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster_max_kcs", type=int, default=250)
    ap.add_argument("--cluster_min_kc_freq", type=int, default=50)
    ap.add_argument("--no_aux", action="store_true", help="Disable auxiliary hint/time/inc models")
    args = ap.parse_args()

    cfg = TrainConfig(
        n_topics=args.n_topics,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        ema_alpha=args.ema_alpha,
        one_hot_actions=True,
        seed=args.seed,
    )
    order_cols = [c.strip() for c in args.order_cols.split(",") if c.strip()] or None
    kc_map_json = args.kc_map_json.strip() or None

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
        kc_map_json=kc_map_json,
        cluster_max_kcs=args.cluster_max_kcs,
        cluster_min_kc_freq=args.cluster_min_kc_freq,
        train_aux_models=(not args.no_aux),
    )



if __name__ == "__main__":
    main()
