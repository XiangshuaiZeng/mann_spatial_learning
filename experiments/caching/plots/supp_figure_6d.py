"""
LDA-based ablation by unit importance (Agent & Goal)
=======================================================================

What this script does:
1) Load the experiment pickle and flatten time series across episodes.
2) For both LDAs:
      - AP: classify agent_node
      - GP: classify goal_node
   Train LDA, compute per-unit importance as ||scalings_||, and evaluate accuracy
   while removing:
      (a) the top-k most important units
      (b) k random units (averaged over n repeats)
   across a sequence of k values (STEPS).
4) Plot a single figure with four curves:
      AP-random, AP-top, GP-random, GP-top

"""

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split


# ========= USER PARAMETERS (edit here) =========
INPUT_PATH = "/local/jon/repositories/mann_spatial_learning-CACHING/experiments/test_data/ppo_homecage_Jon_5_caching_1.pkl"

TEST_SIZE = 0.30          # fraction of samples for test split
RANDOM_STATE = 42         # seed for reproducibility
N_COMPONENTS = 2          # LDA latent dims (will be capped by n_classes-1)
N_REPEATS = 10            # repeats for random ablations (averaged)
LIMIT_ACTIVE_TO = 10000   # cap number of active steps (None disables)

# Ablation schedule; if None, a default 0..D step of 10 will be used per task
STEPS = None
# ==============================================


def main() -> None:
    """Single-pass pipeline: load → flatten → select active → ablation (AP & GP) → plot."""
    # -------------------------
    # Load pickle
    # -------------------------
    data = pickle.load(open(INPUT_PATH, "rb"))

    trajectories = data["trajectories"]      # list[eps][list of (node, ori)]
    goals        = data["goal_location"]     # list[eps][list of goal nodes]
    retrieving   = data["retrieving"]        # list[eps][list of bool/int]
    actions      = data["action"]            # list[eps][raw token or int]
    traces       = data["memory_traces"]     # list[eps][{'rnn_out', ...}]

    # -------------------------
    # Flatten time series
    # -------------------------
    # Build (T, D) activations and aligned (T,) labels/masks.
    rnn_out_full = []
    agent_node_full, goal_node_full = [], []
    action_full, in_reward_full, retrieving_full = [], [], []

    rng = np.random.RandomState(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    for eps in range(len(traces)):
        # (T_eps, D)
        eps_rnn_out = np.squeeze(traces[eps]["rnn_out"])
        rnn_out_full.extend(eps_rnn_out)

        # Flatten trajectory: [(node, ori), ...]
        traj = [item for sublist in trajectories[eps] for item in sublist]

        # Goal per step: if nested like [g], take first element
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
    X_all = np.asarray(rnn_out_full, dtype=float)      # (T, D)
    y_agent_all = np.asarray(agent_node_full, int)
    y_goal_all  = np.asarray(goal_node_full, int)
    action_full = np.asarray(action_full, int)
    in_reward_full = np.asarray(in_reward_full, int)
    retrieving_full = np.asarray(retrieving_full, int)

    # -------------------------
    # Select "active" indices
    # -------------------------
    # Mask: (retrieving==1 & action!=4)  OR  (retrieving==0 & in_reward==1 & action==3)
    # we apply this mask because is the times the agent is retrieving or caching
    active_idx = np.where(
        ((retrieving_full == 1) & (action_full != 4)) #|
        #((retrieving_full == 0) & (in_reward_full == 1) & (action_full == 3))
    )[0]
    if LIMIT_ACTIVE_TO is not None and len(active_idx) > LIMIT_ACTIVE_TO:
        active_idx = active_idx[:LIMIT_ACTIVE_TO]

    X = X_all[active_idx]
    y_agent = y_agent_all[active_idx]
    y_goal  = y_goal_all[active_idx]

    # -------------------------
    # Run ablation for AP and GP
    # -------------------------
    baseline_agent, sizes_agent, random_acc_agent, top_acc_agent = ablation_by_importance_lda(
        X, y_agent,
        n_components=N_COMPONENTS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        steps=STEPS,
        n_repeats=N_REPEATS,
    )

    baseline_goal, sizes_goal, random_acc_goal, top_acc_goal = ablation_by_importance_lda(
        X, y_goal,
        n_components=N_COMPONENTS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        steps=STEPS,
        n_repeats=N_REPEATS,
    )

    # -------------------------
    # Plot (single figure, four curves)
    # -------------------------
    fig, ax = plt.subplots(figsize=(4, 4))

    # AP (Agent Position) curves — blue
    ax.plot(sizes_agent, random_acc_agent, marker='o', linestyle='-',  linewidth=1.5,
            label='AP – random', color='#1f77b4')
    ax.plot(sizes_agent, top_acc_agent,    marker='o', linestyle='--', linewidth=1.5,
            label='AP – top-ranked', color='#1f77b4')

    # GP (Goal Position) curves — red
    ax.plot(sizes_goal, random_acc_goal, marker='s', linestyle='-',  linewidth=1.5,
            label='GP – random',  color='#d62728')
    ax.plot(sizes_goal, top_acc_goal,    marker='s', linestyle='--', linewidth=1.5,
            label='GP – top-ranked',  color='#d62728')

    ax.set_xlabel("Number of units removed", fontsize=18)
    ax.set_ylabel("Classification accuracy", fontsize=18)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=18)
    ax.grid(True)
    ax.legend(frameon=False, loc='lower left', fontsize=14)
    plt.tight_layout()
    fig.savefig("figs/supp_fig_6d", dpi=300, bbox_inches="tight")
    plt.show()


def ablation_by_importance_lda(
    X, y,
    n_components=2,
    test_size=0.3,
    random_state=42,
    steps=None,
    n_repeats=20
):
    """
    Train a multiclass LDA, extract unit 'importance' (row-wise norm of scalings_),
    and evaluate test accuracy while removing k units either at random or from the
    top-k most important ones, for each k in `steps`.

    Returns
    -------
    baseline_acc : float
        Accuracy with all units (no ablation).
    subset_sizes : list[int]
        The k values used for ablation.
    random_acc : list[float]
        Mean accuracy over `n_repeats` random-removal runs, aligned with `subset_sizes`.
    top_acc : list[float]
        Accuracy after removing the top-k most important units, aligned with `subset_sizes`.
    """
    # Default steps: 0, 10, 20, ... up to D
    if steps is None:
        d = X.shape[1]
        steps = list(range(0, d + 1, 10))

    # Split train/test once for fair comparisons
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit LDA with all units
    lda_full = LDA(n_components=min(n_components, np.unique(y_train).size - 1))
    lda_full.fit(X_train, y_train)
    baseline_acc = lda_full.score(X_test, y_test)

    # Compute per-unit importance: ||scalings_|| across components
    weights = lda_full.scalings_              # shape: (D, n_comp)
    importance = np.linalg.norm(weights, axis=1)  # shape: (D,)

    # Sort units by decreasing importance
    sorted_indices = np.argsort(-importance)

    def evaluate_after_removal(remove_idx):
        """Remove the columns at `remove_idx`, refit LDA, return test accuracy."""
        Xtr = np.delete(X_train, remove_idx, axis=1)
        Xte = np.delete(X_test,  remove_idx, axis=1)
        # Cap n_components by (n_classes-1) in the reduced space
        lda = LDA(n_components=min(n_components, np.unique(y_train).size - 1))
        lda.fit(Xtr, y_train)
        return lda.score(Xte, y_test)

    random_acc, top_acc = [], []
    rng = np.random.RandomState(random_state)

    for k in steps:
        if k == 0:
            random_acc.append(baseline_acc)
            top_acc.append(baseline_acc)
            continue

        # Random removal: average over n_repeats
        accs = []
        for _ in range(n_repeats):
            subset = rng.choice(X_train.shape[1], size=k, replace=False)
            accs.append(evaluate_after_removal(subset))
        random_acc.append(float(np.mean(accs)))

        # Top-k removal
        subset_top = sorted_indices[:k]
        top_acc.append(evaluate_after_removal(subset_top))

    return baseline_acc, steps, random_acc, top_acc


if __name__ == "__main__":
    main()
