"""
Reproducible pipeline to generate the correlation‑vs‑distance plots in the CACHING experiments

Goal
----
• Obtain the final figure `all_corr_paper_sem.svg` from the list of .pkl files defined below.

What it does
------------
1) `compute_metrics(path)`: loads one experiment .pkl and extracts three sets of
   pairwise correlations vs exact Euclidean distance on the 5×5 grid for:
   - cache–cache  (in reward, exploring vs exploring)
   - cache–retrieval (in reward, exploring vs retrieving)
   - cache–visit (in reward exploring vs out‑of‑reward retrieving)
   It returns raw (distance, correlation) arrays per condition and also mean±std
   per unique distance for convenience.

2) Aggregation across multiple .pkl files + bootstrap SEM:
   - Compute mean correlation at every *exact distance* and its SEM
   - Draw per‑condition dots and fit a simple decaying exponential to the means (dotted line).

Inputs
------
This expects each .pkl to contain the original fields used by the project:
  step_reward, trajectories, goal_location, retrieving, memory_traces, reached, action

"""

# --- Required imports (kept as requested) ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse  # kept for parity with original imports (not used)
import pickle
from collections import defaultdict
import matplotlib.patches as mpatches  # kept for parity with original imports (not used)

from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# -------------------------
# Configuration block
# -------------------------
# Edit the list below to point to your experiment .pkl files.
PATHS = [
    # Example files used in the paper workflow (replace with your own):
    "/local/jon/repositories/mann_spatial_learning-CACHING/experiments/test_data/ppo_homecage_Jon_5_caching_1.pkl",
    "/local/jon/repositories/mann_spatial_learning-CACHING/experiments/test_data/ppo_homecage_Jon_5_caching_2.pkl",
    "/local/jon/repositories/mann_spatial_learning-CACHING/experiments/test_data/ppo_homecage_Jon_5_caching_3.pkl",
    "/local/jon/repositories/mann_spatial_learning-CACHING/experiments/test_data/ppo_homecage_Jon_5_caching_4.pkl",
    "/local/jon/repositories/mann_spatial_learning-CACHING/experiments/test_data/ppo_homecage_Jon_5_caching_5.pkl",

]

OUT_FIG     = 'all_corr_paper_sem.svg'
BOOTSTRAPS = 1000       # Number of bootstrap resamples
SEM_TH     = 0.05       # SEM threshold for drawing the brackets ][
BRACKET_LEN= 0.25       # Horizontal bracket length
SEED       = 0          # RNG seed for reproducibility

LABELS = ['cache-cache', 'cache-retrieval', 'cache-visit']
COLORS = {
    'cache-cache':     '#e67300ff',
    'cache-retrieval': 'purple',
    'cache-visit':     'gray',
}
KEY_MAP = {'cache-cache':'cc','cache-retrieval':'cr','cache-visit':'vv'}

# Fixed 5×5 grid mapping (same layout as the original code)
GRID = np.array([
    [4, 9, 14, 19, 24],
    [3, 8, 13, 18, 23],
    [2, 7, 12, 17, 22],
    [1, 6, 11, 16, 21],
    [0, 5, 10, 15, 20],
])

# -------------------------------------------------
# Helper: normalize vectors group-wise by a boolean mask
# -------------------------------------------------

def _normalize_by_mask(vectors: np.ndarray, mask: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    """Mean-center the subset of `vectors` selected by `mask`.
    If `previous` is provided, start from it; otherwise create a zeros-like array.
    """
    out = np.zeros_like(vectors) if previous is None else previous.copy()
    if mask.any():
        mean_vec = vectors[mask].mean(axis=0)
        out[mask] = vectors[mask] - mean_vec
    return out

# -------------------------------------------------
# 1) Per-file metric extraction
# -------------------------------------------------

def compute_metrics(path: str | Path) -> dict:
    """Extract raw distances and correlations for three conditions from a single .pkl.

    Returns
    -------
    dict with keys:
      dist_mean_cc, corr_mean_cc, std_cc
      dist_mean_cr, corr_mean_cr, std_cr
      dist_mean_vv, corr_mean_vv, std_vv
      dist_raw_cc,  corr_raw_cc,
      dist_raw_cr,  corr_raw_cr,
      dist_raw_vv,  corr_raw_vv
    """
    data = pickle.load(open(path, 'rb'))

    step_reward   = data['step_reward']             # not used directly here
    trajectories  = data['trajectories']            # list of lists of (node, orient)
    goals         = data['goal_location']           # list of lists [[node], [node], ...]
    retrieving    = data['retrieving']              # list of 0/1 (or bool) indicating retrieval phase
    memory_traces = data['memory_traces']           # dict of arrays per episode
    reached       = data['reached']                 # list of 0/1 per step
    actions       = data['action']                  # per step, can contain "reset"

    # ---- Collect full sequences across episodes; cap at 1000 steps (as in source) ----
    rnn_out_full = []
    retrieving_full = []
    goals_full = []
    agent_node_full = []
    in_reward_full = []
    action_full = []

    for eps in range(len(memory_traces)):
        eps_rnn_out = np.squeeze(memory_traces[eps]['rnn_out'])
        traj = [item for sub in trajectories[eps] for item in sub]  # flatten

        # goals[eps] is list of singleton lists: keep the scalar
        goals[eps] = [item[0] for item in goals[eps]]
        going_nodes = goals[eps]
        goals_full.extend(going_nodes)

        # shift reached one step forward (prepend 0), like the original logic
        reached[eps] = [0] + reached[eps][:-1]

        # normalize actions to ints and map "reset"→4
        act = actions[eps]
        act = np.array([elem[0] if isinstance(elem, list) else elem for elem in act])
        act = np.array([4 if a == "reset" else int(a) for a in act])

        # step loop
        retrieval_vec = []
        for i in range(len(retrieving[0])):
            agent_node_full.append(traj[i][0])
            # in‑reward if agent node equals current goal node
            in_reward_full.append(1 if traj[i][0] == going_nodes[i] else 0)
            # retrieving flag as int
            retrieval_vec.append(0 if retrieving[eps][i] is False else 1)

        # turn the first 0 after a 1 into 1 (matches original post‑processing)
        def _transform(vec):
            out = []
            for k in range(len(vec)):
                if k > 0 and vec[k-1] == 1 and vec[k] == 0:
                    out.append(1)
                else:
                    out.append(vec[k])
            return out

        retrieval_vec = _transform(retrieval_vec)

        # collect episode data
        rnn_out_full.extend(eps_rnn_out)
        retrieving_full.extend(retrieval_vec)
        action_full.extend(act)

    # to arrays
    rnn_out_full     = np.array(rnn_out_full)
    agent_node_full  = np.array(agent_node_full)
    in_reward_full   = np.array(in_reward_full)
    retrieving_full  = np.array(retrieving_full)
    action_full      = np.array(action_full)

    # cap to 1000 steps for parity with the source
    n_keep = min(1000, len(agent_node_full))
    idx = np.arange(n_keep)
    rnn_out_full    = rnn_out_full[idx]
    agent_node_full = agent_node_full[idx]
    in_reward_full  = in_reward_full[idx]
    retrieving_full = retrieving_full[idx]
    action_full     = action_full[idx]

    # pairwise Euclidean distance between steps based on the 5×5 grid
    coords = {GRID[i, j]: (i, j) for i in range(5) for j in range(5)}
    n = len(agent_node_full)
    dist_euclid = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = coords[agent_node_full[i]]
        for j in range(i + 1, n):
            xj, yj = coords[agent_node_full[j]]
            d = float(np.hypot(xi - xj, yi - yj))
            dist_euclid[i, j] = dist_euclid[j, i] = d

    # masks for conditions
    steps_in_reward  = (in_reward_full == 1)
    steps_out_reward = ~steps_in_reward
    steps_ret = (retrieving_full == 1)
    steps_exp = (retrieving_full == 0)

    mask_cc = steps_in_reward & (action_full == 3) & steps_exp   # cache–cache
    mask_cr = steps_in_reward & (action_full == 3) & steps_ret   # cache–retrieval
    mask_vv_row = steps_in_reward & (action_full == 3) & steps_exp  # row side
    mask_vv_col = steps_out_reward & steps_ret                      # column side

    # mean‑center rnn_out per group (three passes)
    rnn_out_norm = _normalize_by_mask(rnn_out_full, mask_cc)
    rnn_out_norm = _normalize_by_mask(rnn_out_full, mask_cr, previous=rnn_out_norm)
    rnn_out_norm = _normalize_by_mask(rnn_out_full, steps_out_reward & steps_ret, previous=rnn_out_norm)

    # collect pairwise correlations and distances
    dist_cc, corr_cc = [], []
    dist_cr, corr_cr = [], []
    dist_vv, corr_vv = [], []

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_euclid[i, j]
            if mask_cc[i] and mask_cc[j]:
                r, _ = pearsonr(rnn_out_norm[i], rnn_out_norm[j])
                dist_cc.append(d); corr_cc.append(r)
            if mask_cc[i] and mask_cr[j]:
                r, _ = pearsonr(rnn_out_norm[i], rnn_out_norm[j])
                dist_cr.append(d); corr_cr.append(r)
            if mask_vv_row[i] and mask_vv_col[j]:
                r, _ = pearsonr(rnn_out_norm[i], rnn_out_norm[j])
                dist_vv.append(d); corr_vv.append(r)

    dist_cc = np.asarray(dist_cc); corr_cc = np.asarray(corr_cc)
    dist_cr = np.asarray(dist_cr); corr_cr = np.asarray(corr_cr)
    dist_vv = np.asarray(dist_vv); corr_vv = np.asarray(corr_vv)

    # aggregate mean/std over *unique* exact distances
    def _mean_std_by_unique(dists: np.ndarray, vals: np.ndarray):
        u = np.unique(dists)
        m = np.array([vals[dists == x].mean() for x in u])
        s = np.array([vals[dists == x].std()  for x in u])
        return u, m, s

    d_cc, m_cc, s_cc = _mean_std_by_unique(dist_cc, corr_cc) if len(dist_cc) else (np.array([]), np.array([]), np.array([]))
    d_cr, m_cr, s_cr = _mean_std_by_unique(dist_cr, corr_cr) if len(dist_cr) else (np.array([]), np.array([]), np.array([]))
    d_vv, m_vv, s_vv = _mean_std_by_unique(dist_vv, corr_vv) if len(dist_vv) else (np.array([]), np.array([]), np.array([]))

    return {
        # aggregated curves
        'dist_mean_cc': d_cc, 'corr_mean_cc': m_cc, 'std_cc': s_cc,
        'dist_mean_cr': d_cr, 'corr_mean_cr': m_cr, 'std_cr': s_cr,
        'dist_mean_vv': d_vv, 'corr_mean_vv': m_vv, 'std_vv': s_vv,
        # raw pairs (for bootstrap)
        'dist_raw_cc': dist_cc, 'corr_raw_cc': corr_cc,
        'dist_raw_cr': dist_cr, 'corr_raw_cr': corr_cr,
        'dist_raw_vv': dist_vv, 'corr_raw_vv': corr_vv,
    }

# -------------------------------------------------
# 2) Multi-file bootstrap + final figure
# -------------------------------------------------

def _exp_fun(x, a, b, c):
    return a * np.exp(-b * x) + c


def run_pipeline(paths: list[str | Path]):
    assert len(paths) > 0, "PATHS is empty. Please add your .pkl file paths at the top of the script."

    rng = np.random.default_rng(SEED)

    # 1) compute per-file metrics
    metrics_list = [compute_metrics(p) for p in paths]

    # 2) gather raw arrays per label
    raw_dist = {lbl: [m[f'dist_raw_{KEY_MAP[lbl]}'] for m in metrics_list] for lbl in LABELS}
    raw_corr = {lbl: [m[f'corr_raw_{KEY_MAP[lbl]}'] for m in metrics_list] for lbl in LABELS}

    # 3) unique exact distances per label across all files
    uniq_dists = {lbl: np.sort(np.unique(np.hstack(raw_dist[lbl]))) for lbl in LABELS}

    # 4) bootstrap SEM of the mean correlation at each exact distance
    sem_boot = {lbl: [] for lbl in LABELS}
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(paths), size=len(paths), endpoint=False)
        for lbl in LABELS:
            d_bs = np.hstack([raw_dist[lbl][i] for i in idx])
            c_bs = np.hstack([raw_corr[lbl][i] for i in idx])
            means_at_d = []
            for d in uniq_dists[lbl]:
                mask = (d_bs == d)
                means_at_d.append(c_bs[mask].mean() if mask.any() else np.nan)
            sem_boot[lbl].append(means_at_d)

    sem   = {lbl: np.nanstd(sem_boot[lbl], axis=0)  for lbl in LABELS}
    means = {lbl: np.nanmean(sem_boot[lbl], axis=0) for lbl in LABELS}

    # 5) draw the figure
    fig, ax = plt.subplots(figsize=(4, 4))
    for lbl in LABELS:
        x_vals = uniq_dists[lbl]
        y_vals = means[lbl]
        err    = sem[lbl]
        col    = COLORS[lbl]

        # 5a) dots only
        ax.plot(x_vals, y_vals, marker='o', linestyle='None', color=col, label=lbl)

        # 5b) draw ][ brackets where SEM exceeds threshold
        for x, y, e in zip(x_vals, y_vals, err):
            if np.isfinite(e) and e > SEM_TH:
                ax.plot([x, x], [y - e, y + e], color=col, linewidth=1)
                ax.plot([x - BRACKET_LEN/2, x + BRACKET_LEN/2], [y + e, y + e], color=col, linewidth=1)
                ax.plot([x - BRACKET_LEN/2, x + BRACKET_LEN/2], [y - e, y - e], color=col, linewidth=1)

        # 5c) dotted exponential fit over valid points
        valid = (~np.isnan(y_vals)) & (~np.isinf(y_vals))
        if valid.sum() > 3:
            try:
                p0 = [y_vals[valid][0] - y_vals[valid][-1], 0.5, y_vals[valid][-1]]
                bounds = ([-np.inf, 0, -np.inf], [np.inf, 5, np.inf])
                popt, _ = curve_fit(_exp_fun, x_vals[valid], y_vals[valid], p0=p0, bounds=bounds)
                xs = np.linspace(0, x_vals[valid].max(), 300)
                ax.plot(xs, _exp_fun(xs, *popt), linestyle=':', linewidth=1.5, color=col)
            except Exception:
                pass

    # 6) final formatting
    ax.set_xlabel('distance', fontsize=15)
    ax.set_ylabel('correlation', fontsize=15)
    ax.set_xlim(-0.2, 5.2)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.tick_params(labelsize=13)
    ax.axhline(0, ls='--', lw=0.8, color='gray', alpha=0.7)
    ax.legend(fontsize=13)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(True)
    plt.tight_layout()
    plt.savefig(f"figs/fig_4b_{OUT_FIG}", dpi=300)
    # no plt.show() to keep batch-friendly


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == '__main__':
    run_pipeline(PATHS)
