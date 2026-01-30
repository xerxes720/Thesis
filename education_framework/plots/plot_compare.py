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
    # HRL single low-level (one shared tutor LL agent)
    "single-agent": "../runs/single_agent.csv",

    # HRL multi-agent without sharing
    "multi-agent": "../runs/multi_agent.csv",

    # HRL multi-agent with weighted transfer (experience sharing = weighted_cka)
    "weighted transfer": "../runs/multi_agent_es.csv",
}

# Needed for "average reward over all agents" proxy
NUM_TOPICS = 7  # <- set this to bundle.n_topics in your experiment
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
            os.path.join(path_or_glob, "**", "seed_*", "metrics.csv"),
        ]
        out = []
        for pat in patterns:
            out.extend(glob.glob(pat, recursive=True))
        return sorted(set(out))

    # 3) assume it is already a glob pattern
    return sorted(set(glob.glob(path_or_glob, recursive=True)))


def _load_curves_for_condition(root_dir: str):
    """
    Returns a list of DataFrames, one per seed, each with columns: episode, reward, steps.
    """
    paths = _find_metrics_csvs(root_dir)
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
    Compute mean curve across seeds (NaN padded). Returns (x, mean_y).
    """
    ys = []
    for df in dfs:
        y = df[col].to_numpy(dtype=float)
        if cumulative:
            y = np.cumsum(y)
        ys.append(y)

    mat = _pad_stack(ys)
    if mat is None:
        return None, None

    mean = np.nanmean(mat, axis=0)
    x = np.arange(1, len(mean) + 1)
    return x, mean


def _representative_curve(dfs, col: str, cumulative: bool = False):
    """
    Pick a single 'representative' seed curve (first one found) to label as non-AVG.
    """
    if not dfs:
        return None, None
    y = dfs[0][col].to_numpy(dtype=float)
    if cumulative:
        y = np.cumsum(y)
    x = np.arange(1, len(y) + 1)
    return x, y


def plot_fig5_reward_per_episode():
    single_dfs = _load_curves_for_condition(COND_DIRS["single-agent"])
    multi_dfs = _load_curves_for_condition(COND_DIRS["multi-agent"])

    plt.figure(figsize=(7.2, 4.2))

    # representative raw (one seed)
    xs, ys = _representative_curve(single_dfs, "reward", cumulative=False)
    xm, ym = _representative_curve(multi_dfs, "reward", cumulative=False)

    # mean across seeds + smoothing
    xs_avg, ys_avg = _mean_curve(single_dfs, "reward", cumulative=False)
    xm_avg, ym_avg = _mean_curve(multi_dfs, "reward", cumulative=False)

    if xs is not None:    plt.plot(xs, ys, alpha=0.25, linewidth=1.0, label="Single-agent")
    if xm is not None:    plt.plot(xm, ym, alpha=0.25, linewidth=1.0, label="Multi-agent")
    if xs_avg is not None: plt.plot(xs_avg, _rolling_mean(ys_avg, SMOOTH_W), linewidth=2.5, label="Single-agent (Avg)")
    if xm_avg is not None: plt.plot(xm_avg, _rolling_mean(ym_avg, SMOOTH_W), linewidth=2.5, label="Multi-agent (Avg)")

    plt.xlabel("Training Episode")
    plt.ylabel("Cumulative Reward Obtained")  # paper wording: cumulative *within episode*
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_fig6_steps_per_episode():
    single_dfs = _load_curves_for_condition(COND_DIRS["single-agent"])
    multi_dfs = _load_curves_for_condition(COND_DIRS["multi-agent"])

    plt.figure(figsize=(7.2, 4.2))

    xs, ys = _representative_curve(single_dfs, "steps", cumulative=False)
    xm, ym = _representative_curve(multi_dfs, "steps", cumulative=False)

    xs_avg, ys_avg = _mean_curve(single_dfs, "steps", cumulative=False)
    xm_avg, ym_avg = _mean_curve(multi_dfs, "steps", cumulative=False)

    if xs is not None:    plt.plot(xs, ys, alpha=0.25, linewidth=1.0, label="Single-agent")
    if xm is not None:    plt.plot(xm, ym, alpha=0.25, linewidth=1.0, label="Multi-agent")
    if xs_avg is not None: plt.plot(xs_avg, _rolling_mean(ys_avg, SMOOTH_W), linewidth=2.5, label="Single-agent (Avg)")
    if xm_avg is not None: plt.plot(xm_avg, _rolling_mean(ym_avg, SMOOTH_W), linewidth=2.5, label="Multi-agent (Avg)")

    plt.xlabel("Training Episode")
    plt.ylabel("Cumulative Steps per Episode")  # paper phrasing, but value is per-episode steps
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_average_reward_over_all_agents():
    dfs_single = _load_curves_for_condition(COND_DIRS["single-agent"])
    dfs_multi = _load_curves_for_condition(COND_DIRS["multi-agent"])
    dfs_weighted = _load_curves_for_condition(COND_DIRS["weighted transfer"])

    def avg_agent_reward_curve(dfs, *, single_agent: bool):
        """
        HRL: mean(ll_reward_*) per episode
        Flat: (no ll_reward_*) -> use episode 'reward' as the single agent’s reward
        """
        ys = []
        for df in dfs:
            ll_cols = [c for c in df.columns if c.startswith("ll_reward_")]

            if single_agent:
                y = df["reward"].to_numpy(dtype=float)
            elif ll_cols:
                y = df[ll_cols].to_numpy(dtype=float)
                y = np.nanmean(y, axis=1)


            else:
                continue

            ys.append(y)

        mat = _pad_stack(ys)
        if mat is None:
            return None, None

        mean = np.nanmean(mat, axis=0)
        x = np.arange(1, len(mean) + 1)
        return x, _rolling_mean(mean, SMOOTH_W)

    plt.figure(figsize=(7.2, 4.2))

    x1, y1 = avg_agent_reward_curve(dfs_single, single_agent=True)
    x2, y2 = avg_agent_reward_curve(dfs_multi, single_agent=False)
    x3, y3 = avg_agent_reward_curve(dfs_weighted, single_agent=False)

    if x1 is not None: plt.plot(x1, y1, linewidth=2.0, label="Single-agent")
    if x2 is not None: plt.plot(x2, y2, linewidth=2.0, label="Multi-agent")
    if x3 is not None: plt.plot(x3, y3, linewidth=2.0, label="Weighted Transfer")

    plt.xlabel("Training Episode")
    plt.ylabel("Average Reward over all Agents")
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_fig5_reward_per_episode()
    plot_fig6_steps_per_episode()
    plot_average_reward_over_all_agents()
