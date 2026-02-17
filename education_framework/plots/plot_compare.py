import os
import glob
import re
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
plot_compare.py (paper + tutee suites)

Key properties (defensibility):
- Exact run_tag matching (no substring "contains" accidents).
- De-duplicate reruns: keep latest metrics per (tag, seed).
- Align curves by explicit 'episode' values (outer-join across seeds).
- Optional per-seed traces + mean ± SEM band across seeds.
- Prints end-window (last W episodes) summary table per condition.

Run examples:
  python -m education_framework.plots.plot_compare --suite paper
  python -m education_framework.plots.plot_compare --suite tutee
  python -m education_framework.plots.plot_compare --suite paper --runs_dir education_framework/runs --expected_seeds 0,23,48 --save_dir education_framework/figs

Expected run_tag names (from run_all.ps1):
  flat_baseline
  paper_single
  paper_multi_no_es
  paper_multi_mutual
  paper_multi_weighted_cka

  tutee_no_es
  tutee_weighted_cka
  tutee_controlA_randLL_all_no_es
  tutee_controlA_randLL_ready_no_es
  tutee_controlA_randLL_all
  tutee_controlA_randLL_ready
"""


# -----------------------------------------------------------------------------
# Condition registry (exact run_tag matching)
# -----------------------------------------------------------------------------
CONDITIONS = OrderedDict([
    ("flat", {
        "label": "Flat (DQN)",
        "tags": ["flat_baseline"],
    }),
    ("paper_single", {
        "label": "HRL single-LL (no ES)",
        "tags": ["paper_single"],
    }),
    ("paper_no_es", {
        "label": "HRL multi-LL (no ES)",
        "tags": ["paper_multi_no_es"],
    }),
    ("paper_mutual", {
        "label": "HRL multi-LL + ES (mutual)",
        "tags": ["paper_multi_mutual"],
    }),
    ("paper_cfa", {
        # Your implementation is weighted_cka; label as CFA-like / similarity-weighted ES.
        "label": "HRL multi-LL + ES (sim-weighted)",
        "tags": ["paper_multi_weighted_cka"],
    }),
    ("paper_cfa_forget", {
        # Your implementation is weighted_cka; label as CFA-like / similarity-weighted ES.
        "label": "HRL multi-LL + ES + forget (sim-weighted)",
        "tags": ["paper_multi_weighted_cka_forget"],
    }),


    ("tutee_no_es", {
        "label": "+Tutee (no ES)",
        "tags": ["tutee_no_es"],
    }),
    ("tutee_es", {
        "label": "+Tutee + ES (sim-weighted)",
        "tags": ["tutee_weighted_cka"],
    }),

    # Control A under no-ES backbone (cleanest “random tutee hurts” evidence)
    ("tutee_ctrl_all_no_es", {
        "label": "Tutee control: random_all (no ES)",
        "tags": ["tutee_controlA_randLL_all_no_es"],
    }),
    ("tutee_ctrl_ready_no_es", {
        "label": "Tutee control: random_allowed (no ES)",
        "tags": ["tutee_controlA_randLL_ready_no_es"],
    }),

    # Control A under ES backbone (secondary)
    ("tutee_ctrl_all_es", {
        "label": "Tutee control: random_all (+ES)",
        "tags": ["tutee_controlA_randLL_all"],
    }),
    ("tutee_ctrl_ready_es", {
        "label": "Tutee control: random_allowed (+ES)",
        "tags": ["tutee_controlA_randLL_ready"],
    }),
])


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _rolling_mean(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if w <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=w, min_periods=max(1, w // 10)).mean().to_numpy()


def _find_metrics_csvs(root: str) -> List[str]:
    # Only training metrics. Your main.py writes: <tag>__seed=<seed>.csv
    return sorted(set(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)))


def _infer_seed_from_path(p: str) -> Optional[int]:
    s = p.replace("\\", "/")
    m = re.search(r"(?:/seed_(\d+)(?:/|$))", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:__seed=|seed[_=])(\d+)", s)
    return int(m.group(1)) if m else None


def _path_tokens(p: str) -> List[str]:
    s = p.replace("\\", "/").lower()
    parts: List[str] = []
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
      - filename token exactly equal to <tag> (e.g., paper_multi_no_es__seed=23.csv)
      - token 'tag=<tag>' or 'run_tag=<tag>' if you ever revert to that format
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


def _select_latest_per_seed(paths: List[str]) -> List[str]:
    """
    If you have reruns of the same (tag, seed), keep only the most recently modified file.
    This prevents double-counting a seed.
    """
    by_seed: Dict[object, Tuple[float, str]] = {}
    for p in paths:
        seed = _infer_seed_from_path(p)
        mtime = os.path.getmtime(p)
        key: object = seed if seed is not None else p
        if key not in by_seed or mtime > by_seed[key][0]:
            by_seed[key] = (mtime, p)
    return [v[1] for v in by_seed.values()]


@dataclass
class LoadedCondition:
    key: str
    label: str
    dfs: List[pd.DataFrame]          # per-seed dfs
    seeds: List[Optional[int]]       # aligned to dfs


def _load_condition(run_tags: List[str], root_dir: str, expected_seeds: Optional[List[int]], verbose: bool) -> LoadedCondition:
    candidates = _find_metrics_csvs(root_dir)

    matched: List[str] = []
    for p in candidates:
        for tag in run_tags:
            if _matches_exact_run_tag(p, tag):
                matched.append(p)
                break

    matched = sorted(set(matched))
    matched = _select_latest_per_seed(matched)

    dfs: List[pd.DataFrame] = []
    seeds: List[Optional[int]] = []

    for p in matched:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        # normalize column names (backwards compatibility)
        if "steps" not in df.columns and "steps_cost" in df.columns:
            df = df.rename(columns={"steps_cost": "steps"})

        required = {"episode", "reward", "steps"}
        if not required.issubset(df.columns):
            continue

        keep = ["episode", "reward", "steps"]

        # keep the additional metrics you want to defend with
        for c in ["mastery_mean", "mastery_min", "completed", "avg_reward_per_topic_slot", "avg_reward_per_learning_agent", "avg_agent_reward"]:
            if c in df.columns:
                keep.append(c)

        df = df[keep].copy()
        df = df.sort_values("episode")
        df = df.drop_duplicates(subset=["episode"], keep="last").reset_index(drop=True)

        dfs.append(df)
        seeds.append(_infer_seed_from_path(p))

    if verbose:
        uniq_seeds = sorted([s for s in seeds if s is not None])
        print(f"[LOAD] tags={run_tags}  files={len(dfs)}  seeds={uniq_seeds}")
        for s, p in zip(seeds, matched):
            print(f"   - seed={s}  {p}")

        if expected_seeds is not None:
            missing = sorted(set(expected_seeds) - set([s for s in seeds if s is not None]))
            if missing:
                print(f"[WARN] Missing seeds for tags={run_tags}: {missing}")

    return LoadedCondition(key="__".join(run_tags), label=" / ".join(run_tags), dfs=dfs, seeds=seeds)


def _merge_across_seeds(dfs: List[pd.DataFrame], col: str) -> Optional[pd.DataFrame]:
    if not dfs:
        return None
    merged: Optional[pd.DataFrame] = None
    for i, df in enumerate(dfs):
        if col not in df.columns:
            continue
        d = df[["episode", col]].copy()
        d = d.rename(columns={col: f"{col}_{i}"})
        merged = d if merged is None else merged.merge(d, on="episode", how="outer")
    if merged is None:
        return None
    return merged.sort_values("episode").reset_index(drop=True)


def _mean_sem_curve(dfs: List[pd.DataFrame], col: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    merged = _merge_across_seeds(dfs, col)
    if merged is None:
        return None, None, None, 0
    ycols = [c for c in merged.columns if c.startswith(f"{col}_")]
    mat = merged[ycols].to_numpy(dtype=float)
    x = merged["episode"].to_numpy(dtype=int)

    # count non-nan per episode
    n = np.sum(np.isfinite(mat), axis=1).astype(float)
    y_mean = np.nanmean(mat, axis=1)
    y_std = np.nanstd(mat, axis=1)
    y_sem = np.where(n > 0, y_std / np.sqrt(np.maximum(n, 1.0)), np.nan)
    n_seeds = int(len(ycols))
    return x, y_mean, y_sem, n_seeds


def _end_window_stats(df: pd.DataFrame, window: int) -> Dict[str, float]:
    w = int(max(1, min(window, len(df))))
    tail = df.iloc[-w:]
    out = {
        "reward": float(tail["reward"].mean()),
        "steps": float(tail["steps"].mean()),
    }
    if "mastery_mean" in tail.columns:
        out["mastery_mean"] = float(tail["mastery_mean"].mean())
    if "mastery_min" in tail.columns:
        out["mastery_min"] = float(tail["mastery_min"].mean())
    if "completed" in tail.columns:
        out["completion"] = float(tail["completed"].mean())
    if "avg_reward_per_topic_slot" in tail.columns:
        out["avg_reward_per_topic_slot"] = float(tail["avg_reward_per_topic_slot"].mean())
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _plot_metric(
    *,
    title: str,
    cond_keys: List[str],
    metric: str,
    runs_dir: str,
    expected_seeds: Optional[List[int]],
    smooth_w: int,
    show_seeds: bool,
    show_band: bool,
    verbose: bool,
    save_path: Optional[str],
):
    plt.figure(figsize=(7.6, 4.4))

    for ck in cond_keys:
        spec = CONDITIONS[ck]
        loaded = _load_condition(spec["tags"], root_dir=runs_dir, expected_seeds=expected_seeds, verbose=verbose)

        if not loaded.dfs:
            print(f"[WARN] No data for {spec['label']} ({spec['tags']})")
            continue

        # faint per-seed curves
        if show_seeds:
            for df in loaded.dfs:
                if metric not in df.columns:
                    continue
                x = df["episode"].to_numpy(dtype=int)
                y = df[metric].to_numpy(dtype=float)
                plt.plot(x, y, alpha=0.12, linewidth=1.0)

        x, y_mean, y_sem, n_seeds = _mean_sem_curve(loaded.dfs, metric)
        if x is None or y_mean is None:
            print(f"[WARN] Missing metric '{metric}' for {spec['label']} ({spec['tags']}). Skipping.")
            continue

        y_sm = _rolling_mean(y_mean, smooth_w)
        plt.plot(x, y_sm, linewidth=1.5, label=f"{spec['label']} (n={n_seeds})")

        if show_band and y_sem is not None:
            sem_sm = _rolling_mean(y_sem, smooth_w)
            plt.fill_between(x, y_sm - sem_sm, y_sm + sem_sm, alpha=0.15)

    plt.title(title)
    plt.xlabel("Training Episode")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[SAVE] {save_path}")
    else:
        plt.show()


def _print_end_window_table(
    *,
    cond_keys: List[str],
    runs_dir: str,
    expected_seeds: Optional[List[int]],
    window: int,
):
    rows = []
    for ck in cond_keys:
        spec = CONDITIONS[ck]
        loaded = _load_condition(spec["tags"], root_dir=runs_dir, expected_seeds=expected_seeds, verbose=False)
        if not loaded.dfs:
            continue
        per_seed = []
        for df in loaded.dfs:
            per_seed.append(_end_window_stats(df, window=window))

        # aggregate across seeds
        keys = sorted(set().union(*[d.keys() for d in per_seed]))
        agg = {"condition": spec["label"], "n": len(per_seed)}
        for k in keys:
            vals = [d.get(k, np.nan) for d in per_seed]
            agg[k] = float(np.nanmean(vals))
        rows.append(agg)

    if not rows:
        print("[WARN] No end-window stats available.")
        return

    # stable column order
    cols = ["condition", "n", "reward", "steps", "mastery_mean", "mastery_min", "completion", "avg_reward_per_topic_slot"]
    cols = [c for c in cols if any(c in r for r in rows)]

    print(f"\n=== End-window summary (last {window} episodes) ===")
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    with pd.option_context("display.max_colwidth", 80, "display.width", 180):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if abs(x) < 10 else f"{x:.3f}"))
    print("=== End summary ===\n")


# -----------------------------------------------------------------------------
# Suites
# -----------------------------------------------------------------------------
def run_suite_paper(args):
    """
    PAPER REIMPLEMENTATION CLAIM (3 plots):

    Plot 1 (reward):  flat vs multi (no ES)
    Plot 2 (steps):   flat vs multi (no ES)
    Plot 3 (avg over agents): single vs multi(no ES) vs mutual ES vs sim-weighted ES
    """
    out = args.save_dir

    _plot_metric(
        title="Paper Re-implementation: Reward per Episode (Flat vs HRL multi no-ES)",
        cond_keys=["flat", "paper_no_es"],
        metric="reward",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "paper_plot1_reward.png") if out else None,
    )

    _plot_metric(
        title="Paper Re-implementation: Steps per Episode (Flat vs HRL multi no-ES)",
        cond_keys=["flat", "paper_no_es"],
        metric="steps",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "paper_plot2_steps.png") if out else None,
    )

    # Fig.7-ish: avg over all agents (topic-slot average) — skip if missing.
    _plot_metric(
        title="Paper Re-implementation: Avg Reward over Agents (single vs multi + ES variants)",
        cond_keys=["paper_single", "paper_no_es", "paper_mutual", "paper_cfa"],
        metric="avg_reward_per_topic_slot",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "paper_plot3_avg_reward_agents.png") if out else None,
    )

    _print_end_window_table(
        cond_keys=["flat", "paper_single", "paper_no_es", "paper_mutual", "paper_cfa"],
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        window=args.end_window,
    )


def run_suite_tutee(args):
    """
    ROBUST & DEFENSIBLE EXTENSION CLAIMS (recommended minimal set):

    A) Isolated tutee value (no ES backbone):
       - HRL multi no-ES vs +Tutee no-ES vs random tutee controls (no-ES)
       Show reward + steps + mastery_mean.

    B) Full system claim (ES backbone):
       - HRL multi sim-weighted ES vs +Tutee+ES vs random tutee controls (+ES)
       Show reward + steps + mastery_mean.

    End-window table prints quantitative summaries.
    """
    out = args.save_dir

    # A1 reward (no ES)
    _plot_metric(
        title="Tutee Value (No ES): Reward per Episode",
        cond_keys=["paper_no_es", "tutee_no_es", "tutee_ctrl_ready_no_es", "tutee_ctrl_all_no_es"],
        metric="reward",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "tutee_A1_reward_no_es.png") if out else None,
    )

    # A2 steps (no ES)
    _plot_metric(
        title="Tutee Value (No ES): Steps per Episode",
        cond_keys=["paper_no_es", "tutee_no_es", "tutee_ctrl_ready_no_es", "tutee_ctrl_all_no_es"],
        metric="steps",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "tutee_A2_steps_no_es.png") if out else None,
    )

    # A3 mastery_mean (no ES) — more defensible than reward
    _plot_metric(
        title="Tutee Value (No ES): Mean Mastery per Episode",
        cond_keys=["paper_no_es", "tutee_no_es", "tutee_ctrl_ready_no_es", "tutee_ctrl_all_no_es"],
        metric="mastery_mean",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "tutee_A3_mastery_no_es.png") if out else None,
    )

    # B1 reward (with ES)
    _plot_metric(
        title="Tutee Value (+ES): Reward per Episode",
        cond_keys=["paper_cfa", "tutee_es", "tutee_ctrl_ready_es", "tutee_ctrl_all_es"],
        metric="reward",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "tutee_B1_reward_es.png") if out else None,
    )

    # B2 steps (with ES)
    _plot_metric(
        title="Tutee Value (+ES): Steps per Episode",
        cond_keys=["paper_cfa_forget", "tutee_es", "tutee_ctrl_ready_es", "tutee_ctrl_all_es"],
        metric="steps",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "tutee_B2_steps_es.png") if out else None,
    )

    # B3 mastery_mean (with ES)
    _plot_metric(
        title="Tutee Value (+ES): Mean Mastery per Episode",
        cond_keys=["paper_cfa_forget", "tutee_es", "tutee_ctrl_ready_es", "tutee_ctrl_all_es"],
        metric="mastery_mean",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "tutee_B3_mastery_es.png") if out else None,
    )

    # C1/C2: Show your tutee extension beats the paper’s proposed ES variants (mutual, sim-weighted)
    _plot_metric(
        title="Tutee vs Paper ES Variants: Reward per Episode",
        cond_keys=["paper_no_es", "paper_mutual", "paper_cfa", "tutee_es"],
        metric="reward",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "tutee_C1_reward_vs_paper_es.png") if out else None,
    )

    _plot_metric(
        title="Tutee vs Paper ES Variants: Mean Mastery per Episode",
        cond_keys=["paper_no_es", "paper_mutual", "paper_cfa", "tutee_es"],
        metric="mastery_mean",
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        smooth_w=args.smooth_w,
        show_seeds=args.show_seeds,
        show_band=args.show_band,
        verbose=args.verbose,
        save_path=os.path.join(out, "tutee_C2_mastery_vs_paper_es.png") if out else None,
    )

    _print_end_window_table(
        cond_keys=[
            "paper_no_es", "tutee_no_es", "tutee_ctrl_ready_no_es", "tutee_ctrl_all_no_es",
            "paper_cfa_forget", "tutee_es", "tutee_ctrl_ready_es", "tutee_ctrl_all_es",
        ],
        runs_dir=args.runs_dir,
        expected_seeds=args.expected_seeds,
        window=args.end_window,
    )


def _parse_seeds(s: str) -> Optional[List[int]]:
    s = (s or "").strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, default="tutee", choices=["paper", "tutee"])
    ap.add_argument("--runs_dir", type=str, default="education_framework/runs")
    ap.add_argument("--save_dir", type=str, default="", help="If set, saves PNGs there instead of showing windows.")
    ap.add_argument("--expected_seeds", type=str, default="0,23,48", help="Comma-separated seeds expected per condition (warn if missing).")
    ap.add_argument("--smooth_w", type=int, default=50)
    ap.add_argument("--end_window", type=int, default=100, help="Last-W episodes used for end-window summary table.")
    ap.add_argument("--show_seeds", action="store_true", default=True, help="Plot faint per-seed curves.")
    ap.add_argument("--no_show_seeds", action="store_true", default=False, help="Disable per-seed curves.")
    ap.add_argument("--show_band", action="store_true", default=True, help="Plot mean ± SEM band (across seeds).")
    ap.add_argument("--no_show_band", action="store_true", default=False, help="Disable SEM band.")
    ap.add_argument("--verbose", action="store_true", default=False, help="Verbose file listing per condition.")
    args = ap.parse_args()

    args.expected_seeds = _parse_seeds(args.expected_seeds)
    args.show_seeds = bool(args.show_seeds and (not args.no_show_seeds))
    args.show_band = bool(args.show_band and (not args.no_show_band))

    args.save_dir = (args.save_dir or "").strip()
    if args.save_dir == "":
        args.save_dir = None

    if args.suite == "paper":
        run_suite_paper(args)
    else:
        run_suite_tutee(args)


if __name__ == "__main__":
    main()