#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA-based unit importance for Agent/Goal (A, B, C only) — minimal helpers
=========================================================================

What this script does:
1) Load the experiment pickle and flatten the time series across episodes.
2) Build the "active" mask:
      (retrieving==1 & action!=4) OR (retrieving==0 & in_reward==1 & action==3)
3) Fit two LDAs on the active samples:
      - LDA(agent_node)
      - LDA(goal_node)
4) Compute per-unit importances as ||loadings|| (norm of LDA scalings).
5) Plot:
      (A) bar(importance_agent)
      (B) bar(importance_goal)
      (C) scatter(importance_agent vs importance_goal), save as SVG.
6) Show A, B, C with plt.show().

Notes:
- Comments are in English.
- Parameters are defined below.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import pearsonr


# ========= USER PARAMETERS (edit here) =========
INPUT_PATH = "/local/jon/repositories/mann_spatial_learning-CACHING/experiments/test_data/ppo_homecage_Jon_5_caching_1.pkl"
LIMIT_ACTIVE_TO = 10000                 # cap number of active steps (None disables)
HIGHLIGHT_IDS = [8, 60, 170, 45, 197]   # optional points to annotate in (C)
SAVE_SCATTER_C_PATH = "figs/fig_4e.svg"   # output file for subplot C (SVG)

# Optional axis limits for (C); set to None for auto scaling
X_LIM_C = (0, 27.2)  # e.g., (0, 27.2)
Y_LIM_C = (0, 20.4)  # e.g., (0, 20.4)

LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
SCATTER_SIZE = 50
SCATTER_ALPHA = 0.7
BAR_ALPHA = 0.7
# ==============================================


def main() -> None:
    """Single-pass pipeline: load, flatten, mask, LDA, plot (A,B,C), save C, show."""
    # --- Load pickle ---
    data = pickle.load(open(INPUT_PATH, "rb"))

    trajectories = data["trajectories"]       # list[eps][list of (node, ori)]
    goals        = data["goal_location"]      # list[eps][list of goal node(s)]
    retrieving   = data["retrieving"]         # list[eps][list of bool/int]
    actions      = data["action"]             # list[eps][raw tokens or ints]
    traces       = data["memory_traces"]      # list[eps][dicts with 'rnn_out']

    # --- Flatten time series across episodes ---
    # build arrays (T,D) activations and (T,) labels/masks aligned per step.
    rnn_out_full = []
    agent_node_full = []
    goal_node_full = []
    action_full = []
    in_reward_full = []
    retrieving_full = []

    for eps in range(len(traces)):
        # (T_eps, D)
        eps_rnn_out = np.squeeze(traces[eps]["rnn_out"])
        rnn_out_full.extend(eps_rnn_out)

        # Flatten trajectory: [(node, ori), ...]
        traj = [item for sublist in trajectories[eps] for item in sublist]

        # Goals per step: if nested like [g], take first element; else take scalar
        eps_goals = [g[0] if isinstance(g, (list, tuple)) else g for g in goals[eps]]

        # Actions to ints: 'reset'->4; try int(); else 0
        eps_actions = []
        for a in actions[eps]:
            if isinstance(a, (list, tuple)) and len(a) > 0:
                a = a[0]
            if isinstance(a, str) and a.lower() == "reset":
                eps_actions.append(4)
            else:
                try:
                    eps_actions.append(int(a))
                except Exception:
                    eps_actions.append(0)
        eps_actions = np.asarray(eps_actions, dtype=int)

        # Retrieval to binary then apply "first zero after one -> one"
        raw_ret = np.asarray([1 if retrieving[eps][i] else 0 for i in range(len(retrieving[eps]))], dtype=int)
        if len(raw_ret) > 1:
            z = raw_ret.copy()
            flip_idx = (raw_ret[1:] == 0) & (raw_ret[:-1] == 1)
            z[1:][flip_idx] = 1
            raw_ret = z

        # Per-step aligned signals
        for i in range(len(raw_ret)):
            node_i, _ = traj[i]
            goal_i = eps_goals[i]
            agent_node_full.append(node_i)
            goal_node_full.append(goal_i)
            action_full.append(eps_actions[i])
            in_reward_full.append(int(node_i == goal_i))
            retrieving_full.append(int(raw_ret[i]))

    # numpy arrays
    X_all = np.asarray(rnn_out_full, dtype=float)       # (T, D)
    agent_node_full = np.asarray(agent_node_full, int)
    goal_node_full = np.asarray(goal_node_full, int)
    action_full = np.asarray(action_full, int)
    in_reward_full = np.asarray(in_reward_full, int)
    retrieving_full = np.asarray(retrieving_full, int)

    # --- Select "active" indices ---
    # keep steps where (retrieving==1 & action!=4) OR (retrieving==0 & in_reward==1 & action==3)
    active_idx = np.where(
        ((retrieving_full == 1) & (action_full != 4)) |
        ((retrieving_full == 0) & (in_reward_full == 1) & (action_full == 3))
    )[0]
    if LIMIT_ACTIVE_TO is not None and len(active_idx) > LIMIT_ACTIVE_TO:
        active_idx = active_idx[:LIMIT_ACTIVE_TO]

    # Active subsets
    X = X_all[active_idx]                     # (N, D)
    y_agent = agent_node_full[active_idx]
    y_goal  = goal_node_full[active_idx]

    # --- LDA fits & loadings(||loadings||) ---
    # fit 2 LDA with 2 components; take per-dimension norm of scalings.
    lda_agent = LDA(n_components=2).fit(X, y_agent)
    imp_agent = np.linalg.norm(lda_agent.scalings_, axis=1)

    lda_goal  = LDA(n_components=2).fit(X, y_goal)
    imp_goal  = np.linalg.norm(lda_goal.scalings_, axis=1)

    # --- Figure with A, B, C only ---
    fig = plt.figure(figsize=(20, 5))

    # (A) Agent importance (bar)
    axA = fig.add_subplot(1, 3, 1)
    axA.bar(np.arange(len(imp_agent)), imp_agent, alpha=BAR_ALPHA)
    axA.set_title("Loading LDA (norm) – Agent Node", fontsize=LABEL_FONTSIZE)
    axA.set_xlabel("Unit index", fontsize=LABEL_FONTSIZE)
    axA.set_ylabel("||loadings||", fontsize=LABEL_FONTSIZE)
    axA.tick_params(axis='both', labelsize=TICK_FONTSIZE)

    # (B) Goal importance (bar)
    axB = fig.add_subplot(1, 3, 2)
    axB.bar(np.arange(len(imp_goal)), imp_goal, alpha=BAR_ALPHA, color="orange")
    axB.set_title("Loading LDA (norm) – Goal Node", fontsize=LABEL_FONTSIZE)
    axB.set_xlabel("Unit index", fontsize=LABEL_FONTSIZE)
    axB.set_ylabel("||loadings||", fontsize=LABEL_FONTSIZE)
    axB.tick_params(axis='both', labelsize=TICK_FONTSIZE)

    # (C) Agent vs Goal (scatter)
    r, p = pearsonr(imp_agent, imp_goal)
    axC = fig.add_subplot(1, 3, 3)
    axC.scatter(imp_agent, imp_goal, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
    axC.set_xlabel("LDA loading magnitude – Agent", fontsize=LABEL_FONTSIZE)
    axC.set_ylabel("LDA loading magnitude – Goal", fontsize=LABEL_FONTSIZE)
    axC.set_title(f"Agent vs Goal (r={r:.2f}, p={p:.1e})", fontsize=LABEL_FONTSIZE)
    axC.grid(True, alpha=0.3)
    axC.tick_params(axis='both', labelsize=TICK_FONTSIZE)

    if X_LIM_C is not None: axC.set_xlim(*X_LIM_C)
    if Y_LIM_C is not None: axC.set_ylim(*Y_LIM_C)

    # Optional highlighted points
    for i in HIGHLIGHT_IDS:
        if 0 <= i < len(imp_agent):
            axC.scatter(imp_agent[i], imp_goal[i], c="black", s=SCATTER_SIZE)
            axC.text(imp_agent[i], imp_goal[i], str(i),
                     fontsize=LABEL_FONTSIZE, ha='right', va='bottom')

    fig.savefig(SAVE_SCATTER_C_PATH, format="svg", bbox_inches="tight")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
