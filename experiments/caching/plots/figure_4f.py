"""
Reproducible goal-conditioned unit rate maps
========================================================

This script reproduces the **goal-conditioned rate maps** of RNN units for a 5×5
grid environment. It flattens episodes, builds standard behavioral masks, and
plots one 5×5 rate map per goal location (25 subplots), saving one PNG per unit.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for reproducible file output
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# =============================
# USER PARAMETERS
# =============================
INPUT_PATH = "/local/jon/repositories/mann_spatial_learning-CACHING/experiments/test_data/ppo_homecage_Jon_5_caching_1.pkl"
OUTDIR = "figs/fig_4f_units_ratemap_goal_conditioned"

# Choose units:
#   - "all"
UNITS = [8, 45, 60, 170, 197]

MIN_OCC = 3            # minimum visits per cell to compute a value
SMOOTH_SIGMA = 0.8     # Gaussian sigma for spatial smoothing (0 disables)
CMAP = "turbo"         # matplotlib colormap name
EXCLUDE_IN_REWARD = False  # exclude steps where agent is physically at the goal


# =============================
# HELPERS
# =============================

# Fixed 5×5 node layout used to place subplots consistently
_POS_MATRIX = np.array([
    [4,  9, 14, 19, 24],
    [3,  8, 13, 18, 23],
    [2,  7, 12, 17, 22],
    [1,  6, 11, 16, 21],
    [0,  5, 10, 15, 20]
])


def node_to_xy(node: int) -> tuple[int, int]:
    """
    Return (row, col) indices in the 5×5 grid for a given node id.
    """
    idx = np.argwhere(_POS_MATRIX == node)
    if idx.size == 0:
        raise ValueError(f"Node {node} not found in position matrix.")
    return tuple(idx[0])


def compute_rate_map(
    single_unit: np.ndarray,
    agent_nodes: np.ndarray,
    mask: np.ndarray,
    min_occ: int = 5,
    smooth_sigma: float | None = 0.8
) -> np.ndarray:
    """Compute a 5×5 rate map for one unit by averaging activity per node.
    """
    rate_map = np.full((5, 5), np.nan, dtype=float)
    for r in range(5):
        for c in range(5):
            node = _POS_MATRIX[r, c]
            idx = (agent_nodes == node) & mask
            if idx.sum() >= min_occ:
                rate_map[r, c] = single_unit[idx].mean()

    if smooth_sigma is not None and smooth_sigma > 0:
        # Smooth values without leaking into NaNs:
        nan_mask = np.isnan(rate_map)
        filled = np.where(nan_mask, 0.0, rate_map)
        smoothed = gaussian_filter(filled, sigma=smooth_sigma, mode="nearest")
        occ = gaussian_filter(~nan_mask, sigma=smooth_sigma, mode="nearest")
        occ[occ == 0] = np.nan  # avoid division by zero
        rate_map = smoothed / occ
        rate_map[nan_mask] = np.nan
    return rate_map


def plot_goal_conditioned_rate_maps(
    unit_idx: int,
    activations: np.ndarray,
    agent_nodes: np.ndarray,
    goal_nodes: np.ndarray,
    actions: np.ndarray,
    retrieving: np.ndarray,
    in_reward: np.ndarray,
    min_occ: int = 3,
    smooth_sigma: float | None = 0.8,
    cmap: str = "turbo",
    exclude_in_reward: bool = False
) -> plt.Figure:
    """
    Plot 25 maps (one per goal) for a single unit; return the Figure.
    """
    if not isinstance(unit_idx, (int, np.integer)):
        raise TypeError(f"`unit_idx` must be int; got {type(unit_idx)}: {unit_idx!r}")
    if activations.ndim != 2:
        raise ValueError(f"`activations` must be (T, D); got shape {activations.shape}")
    if unit_idx < 0 or unit_idx >= activations.shape[1]:
        raise IndexError(f"unit_idx {unit_idx} out of bounds for D={activations.shape[1]}")

    unit_activity = activations[:, unit_idx]

    # Behavioral mask: steps used to accumulate activity into maps
    beh_mask = (retrieving == 1) & (actions != 4)  # ignore "reset"
    if exclude_in_reward:
        beh_mask &= (in_reward == 0)

    maps = []
    global_min, global_max = np.inf, -np.inf
    for goal in range(25):
        g_mask = beh_mask & (goal_nodes == goal)
        m = compute_rate_map(unit_activity, agent_nodes, g_mask, min_occ, smooth_sigma)
        maps.append(m)
        if np.any(~np.isnan(m)):
            global_min = min(global_min, np.nanmin(m))
            global_max = max(global_max, np.nanmax(m))
    if not np.isfinite(global_min):
        global_min, global_max = 0.0, 1.0

    fig, axes = plt.subplots(5, 5, figsize=(4, 4))
    for g, m in enumerate(maps):
        gi, gj = node_to_xy(g)
        ax = axes[gi, gj]
        ax.imshow(m, cmap=cmap, origin="upper", vmin=global_min, vmax=global_max)
        # highlight the goal location
        ax.scatter(gj, gi, marker="o", facecolors="none", edgecolors="black", s=30, linewidths=2)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return fig


def coerce_actions_to_int(arr) -> np.ndarray:
    """
    Convert raw action tokens into integer codes.
    """
    out = []
    for x in arr:
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        if isinstance(x, str) and x.lower() == "reset":
            out.append(4)
            continue
        try:
            out.append(int(x))
        except Exception:
            out.append(0)
    return np.array(out, dtype=int)


def transform_retrieval(vec) -> np.ndarray:
    """
    Apply 'first zero after one becomes one' rule to retrieval sequence.
    """
    out = []
    for i, v in enumerate(vec):
        if i > 0 and vec[i-1] == 1 and v == 0:
            out.append(1)
        else:
            out.append(int(v))
    return np.array(out, dtype=int)


def load_and_prepare(path: str) -> tuple[np.ndarray, ...]:
    """
    Load pickle and build flattened time series for plotting.
    """
    data = pickle.load(open(path, "rb"))
    trajectories = data["trajectories"]
    goals = data["goal_location"]
    retrieving = data["retrieving"]
    actions = data["action"]
    memory_traces = data["memory_traces"]

    rnn_out_full = []
    agent_node_full, goals_full, action_full, in_reward_full, retrieving_full = [], [], [], [], []

    for eps in range(len(memory_traces)):
        eps_rnn_out = np.squeeze(memory_traces[eps]["rnn_out"])  # (T_eps, D)
        rnn_out_full.extend(eps_rnn_out)

        # Flatten trajectory for this episode into [(node, ori), ...]
        trajectory_sq = [item for sublist in trajectories[eps] for item in sublist]

        # Goal at each step: if nested, take the first element
        eps_goals = [g[0] if isinstance(g, (list, tuple)) else g for g in goals[eps]]

        # Normalize actions to integer codes
        eps_actions = coerce_actions_to_int(actions[eps])

        # Build per-step signals
        retrieval = []
        for i in range(len(retrieving[eps])):
            node_i, _ = trajectory_sq[i]
            goal_i = eps_goals[i]

            agent_node_full.append(node_i)
            goals_full.append(goal_i)
            action_full.append(eps_actions[i])
            in_reward_full.append(int(node_i == goal_i))  # at-goal flag
            retrieval.append(int(retrieving[eps][i]))

        # Apply the "first zero after one becomes one" rule
        retrieving_full.extend(transform_retrieval(retrieval))

    return (
        np.array(rnn_out_full, dtype=float),
        np.array(agent_node_full, dtype=int),
        np.array(goals_full, dtype=int),
        np.array(action_full, dtype=int),
        np.array(retrieving_full, dtype=int),
        np.array(in_reward_full, dtype=int),
    )


def parse_units(units_field, dim: int) -> list[int]:
    """
    Parse UNITS field into a validated list of unit indices.
    """
    if isinstance(units_field, str):
        s = units_field.strip().lower()
        if s == "all":
            return list(range(dim))
        # strip square brackets if present and split by comma
        s = s.strip("[]")
        toks = [t.strip() for t in s.split(",") if t.strip()]
        try:
            ids = [int(t) for t in toks]
        except ValueError:
            raise ValueError(
                f"Could not parse UNITS={units_field!r}. "
                "Use 'all', '0,5,17', '[0,5,17]' or a Python list [0,5,17]."
            )
    elif isinstance(units_field, (list, tuple, np.ndarray)):
        ids = [int(x) for x in units_field]
    else:
        raise TypeError(f"Invalid type for UNITS: {type(units_field)}")

    for i in ids:
        if i < 0 or i >= dim:
            raise IndexError(f"Unit {i} out of range (D={dim}).")
    return ids


# =============================
# MAIN
# =============================

def main() -> None:
    """Entrypoint: load data, parse units, generate and save figures."""
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Loading: {INPUT_PATH}")
    (acts, agent_nodes, goal_nodes, actions, retrieving, in_reward) = load_and_prepare(INPUT_PATH)

    if acts.ndim != 2:
        raise ValueError(f"Expected activations (T, D), got {acts.shape}")
    dim = acts.shape[1]

    unit_ids = parse_units(UNITS, dim)
    print(f"Generating maps for units: {unit_ids} (D={dim})")

    for u in unit_ids:
        fig = plot_goal_conditioned_rate_maps(
            u, acts, agent_nodes, goal_nodes, actions,
            retrieving, in_reward,
            min_occ=MIN_OCC, smooth_sigma=(None if SMOOTH_SIGMA == 0 else SMOOTH_SIGMA),
            cmap=CMAP, exclude_in_reward=EXCLUDE_IN_REWARD
        )
        out_path = os.path.join(OUTDIR, f"neuron_{u}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved maps to: {OUTDIR}")


if __name__ == "__main__":
    main()
