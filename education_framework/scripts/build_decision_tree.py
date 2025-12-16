"""
Build per-topic, per-leaf action quality trees from ASSISTments logs and cache to disk.

Pipeline:
1) Read ASSISTments CSV (2009-2010 Combined or Skill Builder).
2) Store minimal columns into SQLite (data/edm.db).
3) Create per-interaction "state" features using rolling / EMA proxies:
     - mastery proxy (EMA of correctness)
     - avg response time proxy
     - hint-rate proxy
     - attempt-rate proxy
4) Derive an "action" label from the log (practice vs hint vs worked_example proxy).
5) Fit per-topic DecisionTreeRegressor that partitions state space.
6) For each leaf, map actions into 5 quality bins based on mean outcome in that leaf.
7) Save trees + leaf maps to models/quality_trees_assistments.joblib

Why this is defensible:
- ASSISTments provides correctness, attempt counts, response-time fields like ms_first_response,
  and hint-related fields like hint_count/hint_total (dataset-dependent). :contentReference[oaicite:1]{index=1}
- This mirrors the paper’s pattern: state partitions + leaf-level action quality categories.

Usage:
  python scripts/build_quality_trees_from_assistments.py --csv path/to/assistments.csv

Notes:
- You must download the dataset yourself and point --csv to it.
- Install dependencies: pip install pandas scikit-learn joblib
"""

from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib


QUALITY_LEVELS = ["very_bad", "bad", "neutral", "good", "very_good"]


# -----------------------------
# Helpers: action quality bins
# -----------------------------

def map_means_to_quality(action_to_mean: Dict[str, float], eps: float = 1e-6) -> Dict[str, str]:
    """
    Convert per-action mean outcome within a leaf into 5 discrete bins.
    Normalize means in [0,1] within the leaf, then threshold into bins.
    """
    if not action_to_mean:
        return {}

    means = np.array(list(action_to_mean.values()), dtype=np.float32)
    mn, mx = float(np.min(means)), float(np.max(means))
    if (mx - mn) < eps:
        return {a: "neutral" for a in action_to_mean.keys()}

    out: Dict[str, str] = {}
    for a, m in action_to_mean.items():
        s = (float(m) - mn) / (mx - mn)
        if s >= 0.85:
            out[a] = "very_good"
        elif s >= 0.65:
            out[a] = "good"
        elif s >= 0.35:
            out[a] = "neutral"
        elif s >= 0.15:
            out[a] = "bad"
        else:
            out[a] = "very_bad"
    return out


# -----------------------------
# Model container (saved to disk)
# -----------------------------

@dataclass
class LeafQualityModel:
    tree: DecisionTreeRegressor
    leaf_to_action_quality: Dict[int, Dict[str, str]]
    default_quality: str = "neutral"


class QualityTreeBank:
    def __init__(self, num_topics: int, feature_names: List[str]):
        self.num_topics = num_topics
        self.feature_names = feature_names
        self.bank: Dict[int, LeafQualityModel] = {}  # topic_id -> model

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "num_topics": self.num_topics,
            "feature_names": self.feature_names,
            "bank": self.bank,
            "version": 1,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "QualityTreeBank":
        payload = joblib.load(path)
        qt = QualityTreeBank(payload["num_topics"], payload["feature_names"])
        qt.bank = payload["bank"]
        return qt

    def predict_quality(self, topic_id: int, action: str, x: np.ndarray) -> str:
        m = self.bank.get(topic_id)
        if m is None:
            return "neutral"
        leaf = int(m.tree.apply(x.reshape(1, -1))[0])
        leaf_map = m.leaf_to_action_quality.get(leaf) or {}
        return leaf_map.get(action, m.default_quality)


# -----------------------------
# Step 1: Load ASSISTments CSV
# -----------------------------

def load_assistments_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # Common columns across 2009-2010 Combined/SkillBuilder:
    # user_id, skill_id (or skill), correct, attempt_count, ms_first_response,
    # hint_count, hint_total, overlap_time, (sometimes) bottom_hint, start_time/end_time.
    required = ["user_id", "correct"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column: {r}")

    # Normalize skill identifier
    if "skill_id" not in df.columns:
        if "skill" in df.columns:
            df["skill_id"] = df["skill"].astype(str)
        elif "skill name" in df.columns:
            df["skill_id"] = df["skill name"].astype(str)
        else:
            raise ValueError("Could not find skill column: expected skill_id or skill")

    # Normalize time proxy
    if "ms_first_response" not in df.columns:
        if "overlap_time" in df.columns:
            df["ms_first_response"] = df["overlap_time"]
        else:
            # still runnable: fallback to NaN and fill later
            df["ms_first_response"] = np.nan

    if "attempt_count" not in df.columns:
        # Some releases have attempt_count; others may use "attempts"
        if "attempts" in df.columns:
            df["attempt_count"] = df["attempts"]
        else:
            df["attempt_count"] = 1

    # Hint columns are dataset-dependent
    if "hint_count" not in df.columns:
        df["hint_count"] = 0
    if "hint_total" not in df.columns:
        df["hint_total"] = 0

    # Optional "bottom_hint" for worked-solution proxy (present in some ASSISTments years)
    if "bottom_hint" not in df.columns:
        df["bottom_hint"] = 0

    # Ensure numeric
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df["attempt_count"] = pd.to_numeric(df["attempt_count"], errors="coerce").fillna(1).astype(int)
    df["ms_first_response"] = pd.to_numeric(df["ms_first_response"], errors="coerce")
    df["hint_count"] = pd.to_numeric(df["hint_count"], errors="coerce").fillna(0).astype(int)
    df["hint_total"] = pd.to_numeric(df["hint_total"], errors="coerce").fillna(0).astype(int)
    df["bottom_hint"] = pd.to_numeric(df["bottom_hint"], errors="coerce").fillna(0).astype(int)

    # Basic cleanup
    df = df.dropna(subset=["user_id", "skill_id"]).copy()
    df["user_id"] = df["user_id"].astype(str)
    df["skill_id"] = df["skill_id"].astype(str)

    return df


# -----------------------------
# Step 2: Store to SQLite
# -----------------------------

def init_sqlite(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        skill_id TEXT NOT NULL,
        correct INTEGER NOT NULL,
        attempt_count INTEGER NOT NULL,
        ms_first_response REAL,
        hint_count INTEGER NOT NULL,
        hint_total INTEGER NOT NULL,
        bottom_hint INTEGER NOT NULL
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_skill ON interactions(user_id, skill_id)")
    con.commit()
    return con


def upsert_interactions(con: sqlite3.Connection, df: pd.DataFrame) -> None:
    cur = con.cursor()
    rows = df[[
        "user_id", "skill_id", "correct", "attempt_count",
        "ms_first_response", "hint_count", "hint_total", "bottom_hint"
    ]].values.tolist()

    cur.executemany("""
    INSERT INTO interactions
      (user_id, skill_id, correct, attempt_count, ms_first_response, hint_count, hint_total, bottom_hint)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    con.commit()


# -----------------------------
# Step 3: Skill -> topic mapping (to match your 8 topics)
# -----------------------------

def build_skill_to_topic(df: pd.DataFrame, num_topics: int, out_json: str) -> Dict[str, int]:
    # Map the most frequent skills to topics [0..num_topics-1].
    top_skills = df["skill_id"].value_counts().head(num_topics).index.tolist()
    mapping = {s: i for i, s in enumerate(top_skills)}

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    return mapping


def load_skill_to_topic(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f).items()}


# -----------------------------
# Step 4: Build features + transitions
# -----------------------------

def derive_action(row: pd.Series) -> str:
    """
    Map log to your project’s action vocabulary (minimal set).

    - bottom_hint == 1  -> worked_example (proxy: full solution / bottom-out hint)
    - hint_count > 0    -> hint
    - otherwise         -> quiz (practice)
    """
    if int(row.get("bottom_hint", 0)) == 1:
        return "worked_example"
    if int(row.get("hint_count", 0)) > 0:
        return "hint"
    return "quiz"


def build_transitions_from_df(
    df: pd.DataFrame,
    skill_to_topic: Dict[str, int],
    num_topics: int,
    ema_alpha: float = 0.15,
    rt_clip_ms: float = 300_000.0,  # 5 min
) -> List[Dict[str, Any]]:
    """
    Build per-interaction transitions with a compact state vector x and an outcome y.

    State x (per topic interaction):
      - mastery_ema (EMA of correctness for that user-topic)
      - rt_ema (EMA of response time, normalized)
      - hint_rate_ema
      - attempt_ema (normalized)
      - global mastery_ema (across topics)  [optional but cheap]

    Outcome y:
      - delta_mastery_ema (after - before)  [paper-like "improvement signal"]
      - minus small time penalty (encourages efficiency but not dominant)
    """
    # Sort to approximate chronology: if there is no timestamp, order in file is used
    # (good enough for a first pass).
    df = df.copy()

    # Keep only skills we mapped (others dropped to keep topic count stable)
    df["topic_id"] = df["skill_id"].map(skill_to_topic)
    df = df.dropna(subset=["topic_id"]).copy()
    df["topic_id"] = df["topic_id"].astype(int)

    # Fill response time if missing
    df["ms_first_response"] = df["ms_first_response"].fillna(df["ms_first_response"].median())

    # Clip RT to avoid extreme outliers dominating EMA
    df["ms_first_response"] = df["ms_first_response"].clip(lower=0.0, upper=rt_clip_ms)

    # Per-user, per-topic EMA trackers
    mastery = {}   # (user, topic) -> ema
    rt = {}        # (user, topic) -> ema
    hint_rate = {} # (user, topic) -> ema
    attempt = {}   # (user, topic) -> ema
    global_mastery = {}  # user -> ema

    transitions: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        user = str(row["user_id"])
        topic = int(row["topic_id"])
        correct = int(row["correct"])
        a_count = int(row["attempt_count"])
        h_count = int(row["hint_count"])
        h_total = int(row["hint_total"])
        rt_ms = float(row["ms_first_response"])

        key = (user, topic)

        # Before
        m0 = mastery.get(key, 0.5)
        rt0 = rt.get(key, 0.5)  # store normalized in [0,1]
        hr0 = hint_rate.get(key, 0.0)
        at0 = attempt.get(key, 0.0)
        gm0 = global_mastery.get(user, 0.5)

        # Normalize features into [0,1]
        # - RT: map 0..300000ms -> 0..1, then invert so higher is "better" like paper normalization
        rt_norm = max(0.0, min(1.0, rt_ms / rt_clip_ms))
        rt_good = 1.0 - rt_norm

        # - hint rate: fraction of possible hints used, or 0..1 if hint_total unavailable
        if h_total > 0:
            hr_val = max(0.0, min(1.0, h_count / float(h_total)))
        else:
            hr_val = 1.0 if h_count > 0 else 0.0

        # - attempts: cap at 5 for normalization
        at_val = max(0.0, min(1.0, a_count / 5.0))

        # Update EMAs
        m1 = (1 - ema_alpha) * m0 + ema_alpha * float(correct)
        rt1 = (1 - ema_alpha) * rt0 + ema_alpha * rt_good
        hr1 = (1 - ema_alpha) * hr0 + ema_alpha * hr_val
        at1 = (1 - ema_alpha) * at0 + ema_alpha * at_val

        gm1 = (1 - ema_alpha) * gm0 + ema_alpha * float(correct)

        mastery[key] = m1
        rt[key] = rt1
        hint_rate[key] = hr1
        attempt[key] = at1
        global_mastery[user] = gm1

        action = derive_action(row)

        # Outcome: improvement in mastery proxy, with a small RT shaping term
        delta_m = m1 - m0
        y = float(delta_m + 0.05 * (rt1 - rt0))

        x = np.array([m0, rt0, hr0, at0, gm0], dtype=np.float32)

        transitions.append({
            "topic_id": topic,
            "action": action,
            "x": x,
            "y": y,
        })

    return transitions


# -----------------------------
# Step 5: Fit trees per topic
# -----------------------------

def fit_quality_trees(
    transitions: List[Dict[str, Any]],
    num_topics: int,
    actions: List[str],
    max_depth: int = 5,
    min_leaf: int = 200,
    seed: int = 0,
) -> QualityTreeBank:
    feature_names = ["mastery_ema", "rt_good_ema", "hint_rate_ema", "attempt_ema", "global_mastery_ema"]
    bank = QualityTreeBank(num_topics=num_topics, feature_names=feature_names)

    # group per topic
    by_topic: Dict[int, List[Dict[str, Any]]] = {t: [] for t in range(num_topics)}
    for tr in transitions:
        by_topic[int(tr["topic_id"])].append(tr)

    for topic_id in range(num_topics):
        rows = by_topic.get(topic_id, [])
        if len(rows) < max(500, min_leaf * 2):
            continue

        X = np.stack([r["x"] for r in rows]).astype(np.float32)
        y = np.array([r["y"] for r in rows], dtype=np.float32)

        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=seed,
        )
        tree.fit(X, y)

        leaf_ids = tree.apply(X).astype(int)

        # leaf -> action -> list[y]
        leaf_action_values: Dict[int, Dict[str, List[float]]] = {}
        for lid, r in zip(leaf_ids, rows):
            a = str(r["action"])
            leaf_action_values.setdefault(int(lid), {}).setdefault(a, []).append(float(r["y"]))

        leaf_to_action_quality: Dict[int, Dict[str, str]] = {}
        for lid, av in leaf_action_values.items():
            action_to_mean = {a: float(np.mean(vals)) for a, vals in av.items()}
            # Keep only actions we care about; missing actions will default at runtime
            action_to_mean = {a: action_to_mean[a] for a in action_to_mean if a in actions}
            leaf_to_action_quality[int(lid)] = map_means_to_quality(action_to_mean)

        bank.bank[topic_id] = LeafQualityModel(
            tree=tree,
            leaf_to_action_quality=leaf_to_action_quality,
            default_quality="neutral",
        )

    return bank


# -----------------------------
# Main
# -----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to ASSISTments CSV file")
    parser.add_argument("--db", type=str, default="data/edm.db")
    parser.add_argument("--out", type=str, default="education_framework/models/quality_trees_assistments.joblib")
    parser.add_argument("--map_out", type=str, default="data/skill_to_topic.json")
    parser.add_argument("--num_topics", type=int, default=8)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_leaf", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        print(f"[SKIP] {args.out} exists. Use --overwrite to refit.")
        return

    print("[1/5] Loading CSV...")
    df = load_assistments_csv(args.csv)
    print(f"  rows: {len(df):,}")

    print("[2/5] Writing to SQLite...")
    con = init_sqlite(args.db)
    upsert_interactions(con, df)
    con.close()
    print(f"  db: {args.db}")

    print("[3/5] Building skill->topic mapping...")
    mapping = build_skill_to_topic(df, args.num_topics, args.map_out)
    print(f"  mapping saved: {args.map_out}")
    print(f"  mapped skills: {len(mapping)}")

    print("[4/5] Building transitions/features...")
    transitions = build_transitions_from_df(df, mapping, num_topics=args.num_topics)
    print(f"  transitions: {len(transitions):,}")

    print("[5/5] Fitting per-topic trees and saving...")
    # Actions aligned with your project’s low-level action meaning set (minimum overlap)
    # You can expand this later.
    actions = ["quiz", "hint", "worked_example"]
    bank = fit_quality_trees(
        transitions=transitions,
        num_topics=args.num_topics,
        actions=actions,
        max_depth=args.max_depth,
        min_leaf=args.min_leaf,
        seed=args.seed,
    )
    bank.save(args.out)
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
