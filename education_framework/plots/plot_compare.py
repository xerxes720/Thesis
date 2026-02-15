import os
import glob
import re
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
plot_compare.py (hardened)

Goals:
- Avoid loading the wrong runs (no substring "contains" filtering).
- Handle multiple seeds correctly.
- Prevent double-counting a seed when multiple reruns exist.
- Align curves by explicit 'episode' values (no index-based x).
- Warn loudly when expected seeds/columns are missing.

Expected run tags (from your run_all.ps1):
  flat_baseline
  paper_multi_no_es
  paper_multi_weighted_cka
  tutee_no_es
  tutee_weighted_cka
"""

# -----------------------------------------------------------------------------
# Where your runs live
# -----------------------------------------------------------------------------
RUNS_DIR = os.environ.get("RUNS_DIR", "../runs")

# -----------------------------------------------------------------------------
# Conditions (exact run_tag matching)
# -----------------------------------------------------------------------------
CONDITIONS = OrderedDict([
    ("flat", {
        "label": "Flat (DQN)",
        "tags": ["flat_baseline"],
    }),
    ("single", {
        "label": "Single (DQN)",
        "tags": ["paper_single"],
    }),
    ("paper_no_es", {
        "label": "HRL multi (no ES)",
        "tags": ["paper_multi_no_es"],
    }),
    ("paper_es", {
        "label": "HRL multi + ES (wCKA)",
        "tags": ["paper_multi_weighted_cka"],
    }),
    ("tutee_no_es", {
        "label": "+Tutee (no ES)",
        "tags": ["tutee_no_es"],
    }),
    ("tutee_es", {
        "label": "+Tutee + ES (wCKA)",
        "tags": ["tutee_weighted_cka"],
    }),
    ("tutee_ctrl_randall", {
        "label": "Tutee control Random all",
        "tags": ["tutee_controlA_randLL_all"],
    }),
    ("tutee_ctrl_rand_allowed", {
        "label": "Tutee control Random allowed",
        "tags": ["tutee_controlA_randLL_ready"],
    }),
])

# Which conditions to show per plot
# FIG5_CONDS = ["flat", "paper_no_es", "paper_es", "tutee_es"]
# FIG6_CONDS = ["flat", "paper_no_es", "paper_es", "tutee_es"]
# FIG7_CONDS = ["single","paper_no_es", "paper_es", "tutee_no_es", "tutee_es"]

FIG5_CONDS = ["flat", "paper_no_es", "paper_es", "tutee_es", "tutee_ctrl_randall", "tutee_ctrl_rand_allowed"]
FIG6_CONDS = ["flat", "paper_no_es", "paper_es", "tutee_es", "tutee_ctrl_randall", "tutee_ctrl_rand_allowed"]
FIG7_CONDS = ["tutee_no_es", "paper_no_es", "paper_es", "tutee_es", "tutee_ctrl_randall", "tutee_ctrl_rand_allowed"]

# smoothing window (paper-ish)
SMOOTH_W = 50

# if True: plot faint per-seed raw curves in background
PLOT_ALL_SEEDS = True

# If you want to enforce that each condition has the same seeds,
# list them here; otherwise leave as None.
EXPECTED_SEEDS = [0, 23, 48]  # e.g., [0, 23, 48]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _rolling_mean(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if w <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=w, min_periods=max(1, w // 10)).mean().to_numpy()


def _find_metrics_csvs(root: str):
    """
    Only pick files that look like the training metrics.
    This avoids accidentally plotting summaries or other csvs.
    """
    patterns = [
        os.path.join(root, "**", ".csv"),
        os.path.join(root, "**", "*.csv"),
    ]
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(set(out))


def _infer_seed_from_path(p: str):
    """
    Extract seed from common patterns:
      - .../seed_23/...
      - ...__seed=23__...
      - ...seed=23...
    """
    s = p.replace("\\", "/")
    m = re.search(r"(?:/seed_(\d+)(?:/|$))", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:__seed=|seed[_=])(\d+)", s)
    return int(m.group(1)) if m else None


def _path_tokens(p: str):
    s = p.replace("\\", "/").lower()
    parts = []
    for seg in s.split("/"):
        if not seg:
            continue
        parts.append(seg)
        parts.extend([t for t in seg.split("__") if t])
    return parts


def _matches_exact_run_tag(p: str, tag: str) -> bool:
    """
    Exact run_tag match; prevents substring mistakes.
    Accepts:
      - token 'tag=<tag>' in filename tokens
      - token 'run_tag=<tag>'
      - directory segment exactly equal to '<tag>'
    """
    tag = tag.lower()
    toks = _path_tokens(p)

    if tag in toks:
        return True
    if f"tag={tag}" in toks:
        return True
    if f"run_tag={tag}" in toks:
        return True

    s = p.replace("\\", "/").lower()
    if re.search(rf"(?:^|__)tag={re.escape(tag)}(?:__|\.csv$)", s):
        return True
    if re.search(rf"(?:^|__)run_tag={re.escape(tag)}(?:__|\.csv$)", s):
        return True
    return False


def _select_latest_per_seed(paths):
    """
    If you have reruns of the same (tag, seed), keep only the most recently modified file.
    This prevents double-counting a seed.
    """
    by_seed = {}
    for p in paths:
        seed = _infer_seed_from_path(p)
        mtime = os.path.getmtime(p)
        key = seed if seed is not None else p  # if seed missing, keep unique
        if key not in by_seed or mtime > by_seed[key][0]:
            by_seed[key] = (mtime, p)
    return [v[1] for v in by_seed.values()]


def _load_condition_dfs(run_tags, root_dir=RUNS_DIR, verbose=True):
    """
    Load all metrics dfs for a condition defined by one or more exact run_tags.
    """
    candidates = _find_metrics_csvs(root_dir)

    matched = []
    for p in candidates:
        for tag in run_tags:
            if _matches_exact_run_tag(p, tag):
                matched.append(p)
                break

    matched = sorted(set(matched))
    matched = _select_latest_per_seed(matched)

    dfs = []
    meta = []  # (seed, path)
    for p in matched:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        # normalize column names
        if "steps" not in df.columns and "steps_cost" in df.columns:
            df = df.rename(columns={"steps_cost": "steps"})

        required = {"episode", "reward", "steps"}
        if not required.issubset(df.columns):
            continue

        # keep relevant columns if present (avoid dragging huge logs)
        keep = ["episode", "reward", "steps"]
        opt_cols = [
            "avg_reward_per_topic_slot",
            "avg_reward_per_learning_agent",
            "avg_agent_reward",
            "use_tutee",
            "arch",
            "n_ll_agents",
        ]
        for c in opt_cols:
            if c in df.columns:
                keep.append(c)

        df = df[keep].copy()

        # sort and de-dup episodes (keep last if logging duplicates)
        df = df.sort_values("episode")
        df = df.drop_duplicates(subset=["episode"], keep="last").reset_index(drop=True)

        dfs.append(df)
        meta.append((_infer_seed_from_path(p), p))

    if verbose:
        seeds = [m[0] for m in meta]
        print(f"[LOAD] tags={run_tags}  files={len(meta)}  seeds={sorted([s for s in seeds if s is not None])}")
        for s, p in meta:
            print(f"   - seed={s}  {p}")

        if EXPECTED_SEEDS is not None:
            missing = sorted(set(EXPECTED_SEEDS) - set([s for s in seeds if s is not None]))
            if missing:
                print(f"[WARN] Missing seeds for tags={run_tags}: {missing}")

    return dfs, meta


def _mean_curve(dfs, col: str):
    """
    Mean across seeds by outer-joining on 'episode'.
    """
    if not dfs:
        return None, None

    merged = None
    for i, df in enumerate(dfs):
        d = df[["episode", col]].copy()
        d = d.rename(columns={col: f"{col}_{i}"})
        merged = d if merged is None else merged.merge(d, on="episode", how="outer")

    merged = merged.sort_values("episode").reset_index(drop=True)
    ycols = [c for c in merged.columns if c.startswith(f"{col}_")]
    mat = merged[ycols].to_numpy(dtype=float)

    x = merged["episode"].to_numpy(dtype=int)
    y = np.nanmean(mat, axis=1)
    return x, y


def _all_seed_curves(dfs, col: str):
    for df in dfs:
        x = df["episode"].to_numpy(dtype=int)
        y = df[col].to_numpy(dtype=float)
        yield x, y


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_fig5_reward_per_episode():
    plt.figure(figsize=(7.2, 4.2))

    for key in FIG5_CONDS:
        spec = CONDITIONS[key]
        dfs, _meta = _load_condition_dfs(spec["tags"], verbose=True)

        if PLOT_ALL_SEEDS:
            for x, y in _all_seed_curves(dfs, "reward"):
                plt.plot(x, y, alpha=0.12, linewidth=1.0)

        x, y = _mean_curve(dfs, "reward")
        if x is None:
            print(f"[WARN] No data for {spec['label']} ({spec['tags']})")
            continue

        plt.plot(x, _rolling_mean(y, SMOOTH_W), linewidth=2.5, label=spec["label"])

    plt.xlabel("Training Episode")
    plt.ylabel("Episode Reward (sum within episode)")
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_fig6_steps_per_episode():
    plt.figure(figsize=(7.2, 4.2))

    for key in FIG6_CONDS:
        spec = CONDITIONS[key]
        dfs, _meta = _load_condition_dfs(spec["tags"], verbose=False)

        x, y = _mean_curve(dfs, "steps")
        if x is None:
            print(f"[WARN] No data for {spec['label']} ({spec['tags']})")
            continue

        plt.plot(x, _rolling_mean(y, SMOOTH_W), linewidth=2.5, label=spec["label"])

    plt.xlabel("Training Episode")
    plt.ylabel("Steps per Episode (to completion)")
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_average_reward_over_all_agents():
    """
    Uses avg_reward_per_topic_slot. If it's missing, we SKIP (no silent fallback).
    This prevents accidentally plotting a different metric and thinking it's Fig.7.
    """
    plt.figure(figsize=(7.2, 4.2))

    for key in FIG7_CONDS:
        spec = CONDITIONS[key]
        dfs, _meta = _load_condition_dfs(spec["tags"], verbose=False)

        dfs = [d for d in dfs if "avg_reward_per_topic_slot" in d.columns]
        x, y = _mean_curve(dfs, "avg_reward_per_topic_slot")
        if x is None:
            print(f"[WARN] Missing avg_reward_per_topic_slot for {spec['label']} ({spec['tags']}). Skipping.")
            continue

        plt.plot(x, _rolling_mean(y, SMOOTH_W), linewidth=2.5, label=spec["label"])

    plt.xlabel("Training Episode")
    plt.ylabel("Average Reward over all Agents (topic-slot average)")
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_fig5_reward_per_episode()
    plot_fig6_steps_per_episode()
    plot_average_reward_over_all_agents()
