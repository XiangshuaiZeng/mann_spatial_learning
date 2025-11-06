"""
LDA analysis of agent-position and goal-position representations.

This script loads a pickled dataset produced by the CACHING experiments,
extracts neural variables and labels, and plots 3D Linear Discriminant Analysis (LDA)
embeddings colored by:
  1) Agent position (node id)
  2) Goal position (node id)
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# =============================================================
# User settings
# =============================================================
# Edit this to point to your .pkl file
DATA_PATH = Path(
    '/local/jon/repositories/mann_spatial_learning-CACHING/experiments/test_data/ppo_homecage_Jon_5_caching_1.pkl'
)
# Only analyze RNN output
ANALYZE_DATA= 'rnn_out'

# Plotting settings
POINT_SIZE = 6
ALPHA = 0.9
FIGSIZE = (7, 6)
RANDOM_SEED = 42

# =============================================================
# Helper functions
# =============================================================

def flatten_episodes(nested: List[np.ndarray]) -> np.ndarray:
    """Concatenate a list-of-arrays (one per episode) into a single array."""
    if len(nested) == 0:
        return np.array([])
    return np.concatenate([np.squeeze(x) for x in nested], axis=0)


def to_int_actions(raw_actions: List[List[int] | int | str]) -> np.ndarray:
    """Map actions to integers and return a 1D numpy array.

    Expected mapping (as used in the original code):
        0 = forward, 1 = right, 2 = left, 3 = reset/cache-step marker, 4 = reset
    Any string 'reset' becomes 4.
    """
    out: List[int] = []
    for a in raw_actions:
        if isinstance(a, list):
            a = a[0]
        if isinstance(a, str) and a.lower() == 'reset':
            out.append(4)
        else:
            out.append(int(a))
    return np.asarray(out, dtype=int)


def build_sequences(data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build flattened step-wise arrays across all episodes with robust per-episode alignment.

    We truncate each episode to the *minimum* length across trajectory, goals,
    retrieving, and actions to avoid shape mismatches.

    Returns:
        agent_node_full: Node occupied by the agent at each step (flattened)
        goals_full: Goal node at each step (flattened)
        in_reward_full: 1 if agent_node == goal on that step, else 0
        retrieving_full: 1 if retrieving True on that step, else 0
        action_full: integer-coded actions per step
    """
    trajectories = data['trajectories']  # list[episode][step] -> (node, orientation)
    goals = data['goal_location']       # list[episode][step] -> list[[goal]] or int-like
    retrieving = data['retrieving']     # list[episode][step] -> bool
    actions = data['action']            # list[episode][step] -> int/str

    agent_nodes: List[int] = []
    goals_all: List[int] = []
    retrieving_all: List[int] = []
    actions_all: List[int] = []

    for eps in range(len(trajectories)):
        # Flatten (node, orientation) for this episode
        traj_eps = [item for sub in trajectories[eps] for item in sub]
        nodes_eps = [n for (n, _hd) in traj_eps]

        # Goals can be nested lists; coerce to scalar per step
        goals_eps = [g[0] if isinstance(g, list) else g for g in goals[eps]]

        # Retrieving -> {False:0, True:1}
        ret_eps = [1 if r else 0 for r in retrieving[eps]]

        # Actions -> ints
        act_eps = to_int_actions(actions[eps]).tolist()

        # Align lengths within this episode
        min_len = min(len(nodes_eps), len(goals_eps), len(ret_eps), len(act_eps))
        if min_len == 0:
            continue
        if not (len(nodes_eps) == len(goals_eps) == len(ret_eps) == len(act_eps)):
            # Truncate to aligned length
            nodes_eps = nodes_eps[:min_len]
            goals_eps = goals_eps[:min_len]
            ret_eps = ret_eps[:min_len]
            act_eps = act_eps[:min_len]

        agent_nodes.extend(nodes_eps)
        goals_all.extend(goals_eps)
        retrieving_all.extend(ret_eps)
        actions_all.extend(act_eps)

    agent_node_full = np.asarray(agent_nodes, dtype=int)
    goals_full = np.asarray(goals_all, dtype=int)
    retrieving_full = np.asarray(retrieving_all, dtype=int)
    action_full = np.asarray(actions_all, dtype=int)

    if not (len(agent_node_full) == len(goals_full) == len(retrieving_full) == len(action_full)):
        raise ValueError(
            f"Aligned arrays must have equal length. Got lengths: "
            f"agent={len(agent_node_full)}, goals={len(goals_full)}, "
            f"retrieving={len(retrieving_full)}, action={len(action_full)}"
        )

    # In-reward flag: agent on goal node
    in_reward_full = (agent_node_full == goals_full).astype(int)

    return agent_node_full, goals_full, in_reward_full, retrieving_full, action_full

def load_memory_trace(memory_traces: List[Dict], vector: str ) -> np.ndarray:
    """Extract and flatten *rnn_out* across episodes -> shape (T, D)."""
    chunks: List[np.ndarray] = []
    for eps in range(len(memory_traces)):
        arr = np.squeeze(memory_traces[eps][vector])  # (T, D)
        if arr.ndim == 1:
            arr = arr[:, None]
        chunks.append(arr)
    return np.concatenate(chunks, axis=0)


def plot_lda_3d(data: np.ndarray,
                labels: np.ndarray,
                colors: np.ndarray,
                title: str) -> None:
    """3D LDA projection; points colored via precomputed color array."""
    lda = LDA(n_components=3)
    embedding = lda.fit_transform(data, labels)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
               c=colors, s=POINT_SIZE, alpha=ALPHA)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title)
    plt.tight_layout()
    # save results
    safe_title = title.replace(' ', '_').replace('—', '-')
    Path("figs").mkdir(exist_ok=True)
    fig.savefig(f"figs/fig_4c_{safe_title}.png", dpi=300, bbox_inches='tight')
    # show plots
    plt.show()


# =============================================================
# Main
# =============================================================
if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    # Load data
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # Build labels and step selection
    agent_node_full, goals_full, in_reward_full, retrieving_full, action_full = build_sequences(data)

    # ---------------------------------------------------------
    # Build node color palette (same style as the original script)
    # ---------------------------------------------------------
    posiciones = np.array([
        [4, 9, 14, 19, 24],
        [3, 8, 13, 18, 23],
        [2, 7, 12, 17, 22],
        [1, 6, 11, 16, 21],
        [0, 5, 10, 15, 20]
    ])

    def _interp(c1, c2, w):
        a = np.array(mcolors.to_rgb(c1)); b = np.array(mcolors.to_rgb(c2))
        return mcolors.to_hex((1 - w) * a + w * b)

    fixed = {0: '#FF0000', 4: '#0000FF', 12: '#000000', 20: '#FFFF00', 24: '#008000'}
    centers = {0: (4, 0), 4: (0, 0), 12: (2, 2), 20: (4, 4), 24: (0, 4)}

    node_colors: Dict[int, str] = {}
    for i in range(posiciones.shape[0]):
        for j in range(posiciones.shape[1]):
            node = posiciones[i, j]
            if node in fixed:
                node_colors[node] = fixed[node]
                continue
            d0 = np.linalg.norm(np.array([i, j]) - np.array(centers[0]))
            d4 = np.linalg.norm(np.array([i, j]) - np.array(centers[4]))
            d12 = np.linalg.norm(np.array([i, j]) - np.array(centers[12]))
            d20 = np.linalg.norm(np.array([i, j]) - np.array(centers[20]))
            d24 = np.linalg.norm(np.array([i, j]) - np.array(centers[24]))
            tot = d0 + d4 + d12 + d20 + d24
            w0 = (tot - 3*d0) / tot
            w4 = (tot - 3*d4) / tot
            w12 = ((tot - 3*d12) / tot) * 0.1
            w20 = (tot - 3*d20) / tot
            w24 = (tot - 3*d24) / tot
            c = _interp(_interp(_interp(_interp(fixed[0], fixed[4], w4/(w0+w4)),
                                        fixed[12], w12/(w0+w4+w12)),
                                 fixed[20], w20/(w0+w4+w12+w20)),
                        fixed[24], w24/(w0+w4+w12+w20+w24))
            node_colors[node] = c

    # Colors per step
    agent_colors = np.array([node_colors[n] for n in agent_node_full])
    goal_colors = np.array([node_colors[n] for n in goals_full])

    # ---------------------------------------------------------
    # Load neural data: rnn_out in this case
    # ---------------------------------------------------------
    X = load_memory_trace(data['memory_traces'], ANALYZE_DATA)

    # ---------------------------------------------------------
    # LDA cases requested:
    # 1) LDA with labels = GOAL, color by GOAL and by AGENT
    # 2) LDA with labels = AGENT, color by AGENT and by GOAL
    # ---------------------------------------------------------

    # (1) Labels = GOAL
    plot_lda_3d(data=X,
                labels=goals_full,
                colors=goal_colors,
                title=f'LDA on {ANALYZE_DATA} — labels: Goal, colors: Goal')

    plot_lda_3d(data=X,
                labels=goals_full,
                colors=agent_colors,
                title=f'LDA on {ANALYZE_DATA} — labels: Goal, colors: Agent')

    # (2) Labels = AGENT
    plot_lda_3d(data=X,
                labels=agent_node_full,
                colors=agent_colors,
                title=f'LDA on {ANALYZE_DATA} — labels: Agent, colors: Agent')

    plot_lda_3d(data=X,
                labels=agent_node_full,
                colors=goal_colors,
                title=f'LDA on {ANALYZE_DATA} — labels: Agent, colors: Goal')
