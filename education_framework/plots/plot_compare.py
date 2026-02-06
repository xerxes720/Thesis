import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configure your run directories
# -----------------------------
# Point these to the folders that contain seed subfolders with metrics.csv inside.
# The script will search recursively for "**/seed_*/metrics.csv".

COND_DIRS = {
    "single-agent": "../runs",
    "single-ll": "../runs",
    "multi-agent": "../runs",
    "weighted transfer": "../runs",
    "tutee": "../runs",

}

COND_FILTERS = {
    "single-agent": ["flat_single"],
    "single-ll": ["single_ll"],
    "multi-agent":  ["multi_no_es"],                 # no experience sharing
    "weighted transfer": ["multi_weighted_cka"],     # experience sharing (weighted CKA)
    "tutee": ["tutee_weighted_cka"],                 # tutee + (weighted CKA) in your names
}

# Needed for "average reward over all agents" proxy
NUM_TOPICS = 7  # <- set this to bundle.n_topics in your experiment
# NUM_TOPICS = 8  # or pull from config if you have it

SMOOTH_W = 100  # paper-like smoothing; tweak to 50/200 if you want


def _rolling_mean(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if w <= 1:
        return y
    # keep same length; simple centered-ish effect via trailing mean
    s = pd.Series(y)
    return s.rolling(window=w, min_periods=max(1, w // 10)).mean().to_numpy()


def _find_metrics_csvs(path_or_glob: str):
    # 1) exact file
    if os.path.isfile(path_or_glob) and path_or_glob.endswith(".csv"):
        return [path_or_glob]

    # 2) directory -> search typical layouts
    if os.path.isdir(path_or_glob):
        patterns = [
            os.path.join(path_or_glob, "**", "metrics.csv"),
            os.path.join(path_or_glob, "**", "metrics__*.csv"),
            os.path.join(path_or_glob, "**", "seed_*", "metrics.csv"),
            os.path.join(path_or_glob, "**", "seed_*", "metrics__*.csv"),
            os.path.join(path_or_glob, "**", "*.csv"),
        ]
        out = []
        for pat in patterns:
            out.extend(glob.glob(pat, recursive=True))
        return sorted(set(out))

    # 3) assume it is already a glob pattern
    return sorted(set(glob.glob(path_or_glob, recursive=True)))


def _load_curves_for_condition(root_dir: str, must_contain=None):
    paths = _find_metrics_csvs(root_dir)

    if must_contain:
        must_contain = list(must_contain)

        # def token_match(p: str) -> bool:
        #     base = os.path.basename(p)
        #     # strip extension, split your run naming convention
        #     base = base.replace(".csv", "")
        #     tokens = set(base.split("__"))
        #     return all(req in tokens for req in must_contain)

        paths = [p for p in paths if all(s in p for s in must_contain)]
    dfs = []
    for p in sorted(paths):
        df = pd.read_csv(p)

        # Accept both naming conventions if you had older logs
        if "steps" not in df.columns and "steps_cost" in df.columns:
            df = df.rename(columns={"steps_cost": "steps"})

        required = {"episode", "reward", "steps"}
        if not required.issubset(set(df.columns)):
            continue

        ll_cols = [c for c in df.columns if c.startswith("ll_reward_")]

        keep = ["episode", "reward", "steps"] + ll_cols

        # keep arch if present (needed to detect flat correctly)
        if "arch" in df.columns:
            keep = ["arch"] + keep
        # NEW: keep these if present
        if "tutee_reward_total" in df.columns:
            keep.append("tutee_reward_total")
        if "avg_agent_reward" in df.columns:
            keep.append("avg_agent_reward")

        df = df[keep].copy()
        df = df.sort_values("episode").reset_index(drop=True)
        dfs.append(df)

    return dfs


def _pad_stack(arrs):
    """
    Stack 1D arrays of different lengths into (n, Lmax) with NaNs padding.
    """
    if not arrs:
        return None
    L = max(len(a) for a in arrs)
    out = np.full((len(arrs), L), np.nan, dtype=float)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


def _mean_curve(dfs, col: str, cumulative: bool = False):
    """
    Mean across seeds by episode number using an outer-join on 'episode'.
    Returns x (episode array) and mean_y.
    """
    if not dfs:
        return None, None

    merged = None
    for i, df in enumerate(dfs):
        d = df[["episode", col]].copy()
        y = d[col].to_numpy(dtype=float)
        if cumulative:
            y = np.cumsum(y)
        d[col] = y
        d = d.rename(columns={col: f"{col}_{i}"})

        merged = d if merged is None else merged.merge(d, on="episode", how="outer")

    merged = merged.sort_values("episode").reset_index(drop=True)
    ycols = [c for c in merged.columns if c.startswith(f"{col}_")]
    mean = merged[ycols].to_numpy(dtype=float)
    mean = np.nanmean(mean, axis=1)

    x = merged["episode"].to_numpy(dtype=int)
    return x, mean

def _mean_and_std_curve(dfs, col: str, cumulative: bool = False):
    if not dfs:
        return None, None, None

    merged = None
    for i, df in enumerate(dfs):
        d = df[["episode", col]].copy()
        y = d[col].to_numpy(dtype=float)
        if cumulative:
            y = np.cumsum(y)
        d[col] = y
        d = d.rename(columns={col: f"{col}_{i}"})
        merged = d if merged is None else merged.merge(d, on="episode", how="outer")

    merged = merged.sort_values("episode").reset_index(drop=True)
    ycols = [c for c in merged.columns if c.startswith(f"{col}_")]
    mat = merged[ycols].to_numpy(dtype=float)

    mean = np.nanmean(mat, axis=1)
    std  = np.nanstd(mat, axis=1)

    x = merged["episode"].to_numpy(dtype=int)
    return x, mean, std


# def _representative_curve(dfs, col: str, cumulative: bool = False):
#     """
#     Pick a single 'representative' seed curve (first one found) to label as non-AVG.
#     """
#     if not dfs:
#         return None, None
#     y = dfs[0][col].to_numpy(dtype=float)
#     if cumulative:
#         y = np.cumsum(y)
#     x = np.arange(1, len(y) + 1)
#     return x, y

def _all_seed_curves(dfs, col: str, cumulative: bool = False):
    """
    Yield (x, y) for each seed separately so matplotlib does not connect seeds together.
    Uses df["episode"] as x (important if episodes are missing).
    """
    for df in dfs:
        x = df["episode"].to_numpy(dtype=int)
        y = df[col].to_numpy(dtype=float)
        if cumulative:
            y = np.cumsum(y)
        yield x, y

def plot_fig5_reward_per_episode():
    single_dfs = _load_curves_for_condition(COND_DIRS["single-agent"], COND_FILTERS["single-agent"])
    multi_dfs = _load_curves_for_condition(COND_DIRS["multi-agent"], COND_FILTERS["multi-agent"])
    tutee_dfs = _load_curves_for_condition(COND_DIRS["tutee"], COND_FILTERS["tutee"])

    print("Loaded:", len(single_dfs), len(multi_dfs), len(tutee_dfs))

    plt.figure(figsize=(7.2, 4.2))

    # representative raw (one seed)
    # plot all seeds (thin)
    for x, y in _all_seed_curves(single_dfs, "reward", cumulative=False):
        plt.plot(x, y, alpha=0.15, linewidth=1.0, label=None)

    for x, y in _all_seed_curves(multi_dfs, "reward", cumulative=False):
        plt.plot(x, y, alpha=0.15, linewidth=1.0, label=None)

    for x, y in _all_seed_curves(tutee_dfs, "reward", cumulative=False):
        plt.plot(x, y, alpha=0.15, linewidth=1.0, label=None)

    # mean across seeds + smoothing
    xs_avg, ys_avg = _mean_curve(single_dfs, "reward", cumulative=False)
    xm_avg, ym_avg = _mean_curve(multi_dfs, "reward", cumulative=False)
    xt_avg, yt_avg = _mean_curve(tutee_dfs, "reward", cumulative=False)

    if xs_avg is not None: plt.plot(xs_avg, _rolling_mean(ys_avg, SMOOTH_W), linewidth=2.5, label="Single-agent (Avg)")
    if xm_avg is not None: plt.plot(xm_avg, _rolling_mean(ym_avg, SMOOTH_W), linewidth=2.5, label="Multi-agent (Avg)")
    if xt_avg is not None: plt.plot(xt_avg, _rolling_mean(yt_avg, SMOOTH_W), linewidth=2.5, label="Tutee (Avg)")

    plt.xlabel("Training Episode")
    plt.ylabel("Cumulative Reward Obtained")  # paper wording: cumulative *within episode*
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_fig6_steps_per_episode():
    single_dfs = _load_curves_for_condition(COND_DIRS["single-agent"], COND_FILTERS["single-agent"])
    multi_dfs  = _load_curves_for_condition(COND_DIRS["multi-agent"],  COND_FILTERS["multi-agent"])
    tutee_dfs  = _load_curves_for_condition(COND_DIRS["tutee"],        COND_FILTERS["tutee"])

    plt.figure(figsize=(7.2, 4.2))

    # 1) mean per-episode steps across seeds (NO cumulative here)
    xs, ys = _mean_curve(single_dfs, "steps", cumulative=False)
    xm, ym = _mean_curve(multi_dfs,  "steps", cumulative=False)
    xt, yt = _mean_curve(tutee_dfs,  "steps", cumulative=False)

    # 2) smooth the per-episode signal (this is the right place to smooth)
    if xs is not None: ys_s = _rolling_mean(ys, SMOOTH_W)
    if xm is not None: ym_s = _rolling_mean(ym, SMOOTH_W)
    if xt is not None: yt_s = _rolling_mean(yt, SMOOTH_W)

    # Choose ONE of these two:

    # A) Paper-style "steps per episode" (recommended)
    if xs is not None: plt.plot(xs, ys_s, linewidth=2.5, label="Single-agent (Avg)")
    if xm is not None: plt.plot(xm, ym_s, linewidth=2.5, label="Multi-agent (Avg)")
    if xt is not None: plt.plot(xt, yt_s, linewidth=2.5, label="Tutee (Avg)")
    plt.ylabel("Steps per Episode (to completion)")

    # B) If you truly want cumulative over training, cumsum AFTER smoothing:
    # if xs is not None: plt.plot(xs, np.cumsum(ys_s), linewidth=2.5, label="Single-agent (Avg)")
    # if xm is not None: plt.plot(xm, np.cumsum(ym_s), linewidth=2.5, label="Multi-agent (Avg)")
    # if xt is not None: plt.plot(xt, np.cumsum(yt_s), linewidth=2.5, label="Tutee (Avg)")
    # plt.ylabel("Cumulative Steps (over Training)")


    plt.xlabel("Training Episode")
    plt.tight_layout()
    plt.legend()
    plt.show()





def plot_average_reward_over_all_agents():
    single_dfs = _load_curves_for_condition(COND_DIRS["single-ll"], COND_FILTERS["single-ll"])
    multi_dfs = _load_curves_for_condition(COND_DIRS["multi-agent"], COND_FILTERS["multi-agent"])
    tutee_dfs = _load_curves_for_condition(COND_DIRS["tutee"], COND_FILTERS["tutee"])
    dfs_weighted = _load_curves_for_condition(COND_DIRS["weighted transfer"], COND_FILTERS["weighted transfer"])

    def avg_agent_reward_curve(dfs):
        """
        Comparable signal across ALL conditions (Option A):
          y = total_reward_per_episode / NUM_TOPICS

        We infer NUM_TOPICS from ll_reward_* columns when available,
        otherwise fall back to a configured constant.
        """
        ys = []

        # infer num_topics robustly
        def infer_num_topics(df):
            ll_cols = [c for c in df.columns if c.startswith("ll_reward_")]
            if ll_cols:
                return len(ll_cols)
            # fallback: use your global constant if you have one
            # e.g., NUM_TOPICS = 7
            return NUM_TOPICS

        for df in dfs:
            if "reward" not in df.columns:
                # nothing sensible to plot
                continue

            num_topics = infer_num_topics(df)
            if num_topics <= 0:
                continue

            y = df["reward"].to_numpy(dtype=float) / float(num_topics)
            ys.append(y)

        mat = _pad_stack(ys)
        if mat is None:
            return None, None

        mean = np.nanmean(mat, axis=0)
        x = np.arange(1, len(mean) + 1)
        return x, _rolling_mean(mean, SMOOTH_W)

    plt.figure(figsize=(7.2, 4.2))

    x1, y1 = avg_agent_reward_curve(single_dfs)
    x2, y2 = avg_agent_reward_curve(multi_dfs)
    x3, y3 = avg_agent_reward_curve(dfs_weighted)
    x4, y4 = avg_agent_reward_curve(tutee_dfs)

    if x1 is not None: plt.plot(x1, y1, linewidth=2.0, label="Single-agent")
    if x2 is not None: plt.plot(x2, y2, linewidth=2.0, label="Multi-agent")
    if x3 is not None: plt.plot(x3, y3, linewidth=2.0, label="Weighted Transfer")
    if x4 is not None: plt.plot(x4, y4, linewidth=2.0, label="Tutee")

    plt.xlabel("Training Episode")
    plt.ylabel("Average Reward over all Agents")
    plt.tight_layout()
    plt.legend()
    plt.show()



if __name__ == "__main__":
    plot_fig5_reward_per_episode()
    plot_fig6_steps_per_episode()
    plot_average_reward_over_all_agents()
