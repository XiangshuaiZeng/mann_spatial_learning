"""
Figure 7 (F–J) – Continuous State Space Visualizations
======================================================

What this script does
---------------------
This script reproduces the panels of Figure 7 for an agent trained in a **continuous
state space**:
- **(F, G)**: Low-dimensional embeddings (e.g., PCA) of selected memory traces with
  multiple colorings (head orientation, agent node, goal, action, in-reward, retrieving, etc.).
- **(H, I)**: Orientation-conditioned **rate maps** (“wind-rose” layout) for each unit of
  `r_key`, optionally at a higher-resolution grid (e.g., 10×10) re-quantized from (x, z).
- **(J)**: **Unit classification** into Place (PC), Head-Direction-modulated Place (HD-PC),
  Non-Place (Non-PC), or Non-def, based on spatial/rotational similarity and simple
  heuristics. A compact summary (counts + pie chart) is produced.

How to use the on/off switches
------------------------------
You can quickly enable or disable entire analysis blocks by toggling these booleans:
- `plot_PCA`            → controls Figure 7F–G style embeddings.
- `draw_heatmaps`       → controls Figure 7H–I style rate maps (global + wind-rose).
- `unit_classification` → controls Figure 7J style classification and summary plots.

You can also choose which memory trace to embed by editing `variables_active`
(e.g., use `'r_key_active'`, `'rnn_out_active'`, `'fwm_active'`, etc.).

Selecting the time steps to analyze (ACTIVE_IDX_RULE)
-----------------------------------------------------
All computations operate on a **subset of steps** chosen by `ACTIVE_IDX_RULE`.
Edit this rule to decide exactly which frames are “active” and included in the analyses.

Examples (swap into `ACTIVE_IDX_RULE = lambda ...:`):
1) **Figure 7f** (retrieval_phase, exclude resets):
   `(number_rew_full > 1) & (action_full != 4)`

2) **Figure 7g**Restrict to a specific head orientation** (e.g., **North** = 0°, after 22.5°→45° binning):
   `(number_rew_full > 1) & (action_full != 4) & (head_ori_full == 0)`

3) **Figure 7h,i,j**All frames** (for rate maps like 7H, 7I):
   `(number_rew_full >= 0)`

Feel free to combine any conditions on:
`number_rew_full`, `action_full`, `head_ori_full`, `retrieving_full`, `in_reward_full`,
`agent_node_full`, etc. The script will slice all variables to these active indices.

In short
--------
1) Point `path` to your dataset.
2) Set `ACTIVE_IDX_RULE` to the subset you want (e.g., retrieval-only, facing North, etc.).
3) Toggle `plot_PCA`, `draw_heatmaps`, and `unit_classification` to build F–J.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap
import matplotlib.colors as mcolors
import pandas as pd

# ----------------------------
# Config: path & active-index rules
# ----------------------------
path = "/local/jon/repositories/mann_spatial_learning-CONTINUOUS_STATE_SPACE/experiments/test_data/ppo_homecage_redwall_DISCRETETANK_1.pkl"

# Define active-index selection rules
# For figure 7f:    (number_rew_full > 1) & (action_full != 4)
# For figure 7g:    (number_rew_full > 1) & (action_full != 4) & (head_ori_full==0)
# For ploting rate maps like figure 7h,i:  (number_rew_full >= 0)
ACTIVE_IDX_RULE = lambda number_rew_full, action_full, head_ori_full: np.where((number_rew_full > 1) & (action_full != 4))[0]

# in variables_active define which memory traces to plot: for example, "r_key", "rnn_out"....

plot_PCA = True  # controls Figure 7F–G style embeddings.
draw_heatmaps = False #controls Figure 7H–I style rate maps (global + wind-rose).
unit_classification = True #controls Figure 7J style classification and summary plots.

# ----------------------------
# Grid layout (node indices)
# ----------------------------
positions= np.array([
    [4, 9, 14, 19, 24],
    [3, 8, 13, 18, 23],
    [2, 7, 12, 17, 22],
    [1, 6, 11, 16, 21],
    [0, 5, 10, 15, 20]
])

# ----------------------------
# Load data
# ----------------------------
data = pickle.load(open(path, 'rb'))
step_reward   = data['step_reward']
trajectories  = data['trajectories']
goals         = data['goal_location']
retrieving    = data['retrieving']
memory_traces = data['memory_traces']
reached       = data['reached']
actions       = data['action']

# ----------------------------
# Small utilities
# ----------------------------
def coordinates_to_node(x, z, grid_size=5):
    """Map continuous (x,z) into nearest 5x5 node id (clipped to world limits)."""
    world_limits = [[-2., 2.], [-2., 2.]]
    x_min, x_max = world_limits[0]
    z_min, z_max = world_limits[1]
    x = np.clip(x, x_min, x_max)
    z = np.clip(z, z_min, z_max)
    row = int(round((x - x_min) / ((x_max - x_min) / (grid_size - 1))))
    col = int(round((z - z_min) / ((z_max - z_min) / (grid_size - 1))))
    row = np.clip(row, 0, grid_size - 1)
    col = np.clip(col, 0, grid_size - 1)
    return row * grid_size + col

def categorize_angle_deg(angle):
    """Quantize angle to 16 sectors of 22.5°."""
    angle = angle % 360
    category = round(angle / 22.5) * 22.5
    return category % 360

def interpolate_color(color1, color2, factor):
    """Linear interpolation between two hex colors (0<=factor<=1)."""
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    return mcolors.to_hex((1 - factor) * c1 + factor * c2)

def embedding_f(data, dim, method='pca', labels=None):
    """Return (components, embedding) with PCA/ICA/UMAP/LDA; flattens last dim if needed."""
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    components, embedding = None, None
    if method == 'umap':
        reducer = umap.UMAP(n_components=3, n_neighbors=100, min_dist=0.4)
        embedding = reducer.fit_transform(data if labels is None else data)
    elif method == 'pca':
        pca = PCA(n_components=3).fit(data)
        components = pca.components_
        embedding = pca.transform(data)
    elif method == 'ica':
        ica = FastICA(n_components=dim, random_state=0)
        ica.fit(data)
        components = ica.components_
        embedding = ica.transform(data)
    elif method == 'umap_supervised':
        reducer = umap.UMAP(n_components=dim, n_neighbors=200, min_dist=0.6)
        embedding = reducer.fit_transform(data, y=labels) if labels is not None else reducer.fit_transform(data)
    elif method == 'lda':
        if labels is None:
            raise ValueError("LDA requires labels.")
        lda = LDA(n_components=min(dim, len(np.unique(labels)) - 1))
        embedding = lda.fit_transform(data, labels)
        components = lda.scalings_
    return components, embedding

def plot_embedding(embedding, dim, coloring):
    """Scatter plot (2D/3D) of provided embedding with given coloring."""
    if dim == 3:
        fig = plt.figure(figsize=(5, 4))
        axs = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(axs)
        sc = axs.scatter3D(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=coloring, cmap='viridis')
    else:
        fig, axs = plt.subplots(figsize=(7, 7))
        sc = axs.scatter(embedding[:, 0], embedding[:, 1], c=coloring, cmap='viridis')
        axs.scatter(embedding[-1, 0], embedding[-1, 1], s=100)
    return axs, sc

# ----------------------------
# Build full sequences
# ----------------------------
r_value_full, r_key_full, w_value_full, new_w_value_full = [], [], [], []
w_key_full, rnn_cell_full, rnn_out_full, w_beta_full = [], [], [], []
fwm_full, delta_fwm_full = [], []
retrieving_full, goals_full = [], []
agent_node_full, head_ori_full, action_full = [], [], []
in_reward_full, number_rew_full, sequence, step_full = [], [], [], []

for eps in range(len(memory_traces)):
    eps_r_value     = np.squeeze(memory_traces[eps]['r_value'])
    eps_r_key       = np.squeeze(memory_traces[eps]['r_key'])
    eps_w_value     = np.squeeze(memory_traces[eps]['w_value'])
    eps_new_w_value = np.squeeze(memory_traces[eps]['new_w_value'])
    eps_w_key       = np.squeeze(memory_traces[eps]['w_key'])
    eps_rnn_cell    = np.squeeze(memory_traces[eps]['rnn_cell'])
    eps_rnn_out     = np.squeeze(memory_traces[eps]['rnn_out'])
    eps_w_beta      = np.squeeze(memory_traces[eps]['w_beta'])
    eps_fwm_full    = np.squeeze(memory_traces[eps]['fwm'])
    eps_delta_fwm   = np.squeeze(memory_traces[eps]['delta_fwm'])

    trajectory_sq = [it for sub in trajectories[eps] for it in sub]
    goals[eps] = [it[0] for it in goals[eps]]
    going_nodes = goals[eps]

    # shift reached by one (prepend 0) to match original behavior
    reached[eps] = [0] + reached[eps][:-1]

    retrieval = []
    counter = 0

    # normalize actions (string "reset" -> 4; cast to int otherwise)
    actionss = actions[eps]
    actionss = np.array([e[0] if isinstance(e, list) else e for e in actionss])
    actionss = np.array([4 if a == "reset" else int(a) for a in actionss])

    for i in range(len(retrieving[0])):
        step_full.append(i)

        # agent node & head direction
        agent_node = coordinates_to_node(trajectory_sq[i][0], trajectory_sq[i][1])
        agent_node_full.append(agent_node)
        head_ori_full.append(categorize_angle_deg(trajectory_sq[i][2]))

        # reward counters / flags
        number_rew_full.append(reached[eps][i])
        in_reward_full.append(1 if trajectory_sq[i][0] == goals[eps][i - 1] else 0)
        if trajectory_sq[i][0] == goals[eps][i - 1]:
            counter = 0
        sequence.append(counter)
        counter += 1

        # retrieving flag as 0/1
        retrieval.append(0 if retrieving[eps][i] is False else 1)

    # extend per-episode arrays
    r_value_full.extend(eps_r_value);       r_key_full.extend(eps_r_key)
    w_value_full.extend(eps_w_value);       new_w_value_full.extend(eps_new_w_value)
    w_key_full.extend(eps_w_key);           rnn_cell_full.extend(eps_rnn_cell)
    rnn_out_full.extend(eps_rnn_out);       w_beta_full.extend(eps_w_beta)
    fwm_full.extend(eps_fwm_full);          delta_fwm_full.extend(eps_delta_fwm)
    action_full.extend(actionss);           retrieving_full.extend(retrieval)
    goals_full.extend(going_nodes)

# distance-to-last-reward (negative sequence)
negative_sequence = []
last_idx = -1
for i, val in enumerate(in_reward_full):
    v = 1 if i == 0 else val  # force first value as 1 (matches original quirk)
    if v == 1:
        if last_idx != -1:
            for j in range(last_idx + 1, i):
                negative_sequence[j] = -(i - j)
        last_idx = i
        negative_sequence.append(0)
    else:
        negative_sequence.append(None)
if last_idx != -1:
    for j in range(last_idx + 1, len(in_reward_full)):
        negative_sequence[j] = -(j - last_idx)

# to numpy
r_value_full = np.array(r_value_full);      r_key_full = np.array(r_key_full)
w_value_full = np.array(w_value_full);      new_w_value_full = np.array(new_w_value_full)
w_key_full   = np.array(w_key_full);        rnn_cell_full = np.array(rnn_cell_full)
rnn_out_full = np.array(rnn_out_full);      w_beta_full = np.array(w_beta_full)
retrieving_full = np.array(retrieving_full)
fwm_full = np.array(fwm_full);              delta_fwm_full = np.array(delta_fwm_full)

number_rew_full = np.array(number_rew_full)
head_ori_full   = np.array(head_ori_full)
agent_node_full = np.array(agent_node_full)
goals_full      = np.array(goals_full)
action_full     = np.array(action_full)
in_reward_full  = np.array(in_reward_full)
negative_sequence_full = np.array(negative_sequence)
sequence_full   = np.array(sequence)
step_full       = np.array(step_full)

# ----------------------------
# Color maps (unchanged)
# ----------------------------
angle_color_map = {
    0.0:'#ff0000', 22.5:'#ff5f00', 45.0:'#ffbd00', 67.5:'#e2ff00',
    90.0:'#84ff00', 112.5:'#25ff00', 135.0:'#00ff39', 157.5:'#00ff97',
    180.0:'#00fff6', 202.5:'#00aaff', 225.0:'#004bff', 247.5:'#1300ff',
    270.0:'#7200ff', 292.5:'#d000ff', 315.0:'#ff00cf', 337.5:'#ff0071'
}
actions_color_map = {3:'black', 4:'pink', 2:'green', 1:'yellow', 0:'purple'}

fixed_colors = {
    0:'#FF0000', 4:'#0000FF', 12:'#000000', 20:'#FFFF00', 24:'#008000'
}
coords_fixed = {0:(4,0), 4:(0,0), 12:(2,2), 20:(4,4), 24:(0,4)}

# Interpolate colors across the grid (kept identical logic)
node_colors = {}
for i in range(positions.shape[0]):
    for j in range(positions.shape[1]):
        node = positions[i, j]
        if node in fixed_colors:
            node_colors[node] = fixed_colors[node]
            continue
        d0  = np.linalg.norm(np.array([i, j]) - np.array(coords_fixed[0]))
        d4  = np.linalg.norm(np.array([i, j]) - np.array(coords_fixed[4]))
        d12 = np.linalg.norm(np.array([i, j]) - np.array(coords_fixed[12]))
        d20 = np.linalg.norm(np.array([i, j]) - np.array(coords_fixed[20]))
        d24 = np.linalg.norm(np.array([i, j]) - np.array(coords_fixed[24]))
        tot = d0 + d4 + d12 + d20 + d24
        w0  = (tot - d0  * 3) / tot
        w4  = (tot - d4  * 3) / tot
        w12 = (tot - d12 * 3) / tot
        w20 = (tot - d20 * 3) / tot
        w24 = (tot - d24 * 3) / tot
        w12 *= 0.1
        c = interpolate_color(
              interpolate_color(
                interpolate_color(
                  interpolate_color(fixed_colors[0], fixed_colors[4], w4/(w0+w4)),
                  fixed_colors[12], w12/(w0+w4+w12)),
                fixed_colors[20], w20/(w0+w4+w12+w20)),
            fixed_colors[24], w24/(w0+w4+w12+w20+w24))
        node_colors[node] = c
node_colors.update(fixed_colors)

# (Optional) visualize grid colors — unchanged behavior
fig, ax = plt.subplots()
for i in range(positions.shape[0]):
    for j in range(positions.shape[1]):
        node = positions[i, j]
        rect = plt.Rectangle((j, i), 1, 1, facecolor=node_colors[node])
        ax.add_patch(rect)
plt.xlim(0, positions.shape[1]); plt.ylim(0, positions.shape[0])
plt.gca().invert_yaxis(); plt.axis('off')
# plt.show()

# ----------------------------
# Active selection
# ----------------------------
active_indices = ACTIVE_IDX_RULE(number_rew_full, action_full, head_ori_full)
if len(active_indices) > 100000:
    active_indices = active_indices[:100000]

# Slice active views
r_value_active     = r_value_full[active_indices]
r_key_active       = r_key_full[active_indices]
w_value_active     = w_value_full[active_indices]
new_w_value_active = new_w_value_full[active_indices]
w_key_active       = w_key_full[active_indices]
rnn_cell_active    = rnn_cell_full[active_indices]
rnn_out_active     = rnn_out_full[active_indices]
fwm_active         = fwm_full[active_indices]

number_rew_active      = number_rew_full[active_indices]
head_ori_active        = head_ori_full[active_indices]
agent_node_active      = agent_node_full[active_indices]
goals_active           = goals_full[active_indices]
action_active          = action_full[active_indices]
in_reward_active       = in_reward_full[active_indices]
negative_sequence_active = negative_sequence_full[active_indices]
sequence_active        = sequence_full[active_indices]
retrieving_active      = retrieving_full[active_indices]

# Small step table (kept as original structure and length)
step_data = [{
    'Step': s + 1,
    'Agent Node': agent_node_full[s],
    'Goal': goals_full[s],
    'retrieving': retrieving_full[s],
    'In Reward': in_reward_full[s],
    'Numero de reward': number_rew_full[s],
    'Action': action_full[s],
} for s in range(0, 60)]
df_steps = pd.DataFrame(step_data)
print(df_steps.head(60))

# Color encodings (unchanged)
agent_node_coloring = np.array([node_colors[n] for n in agent_node_active])
head_ori_coloring   = np.array([angle_color_map[ori] for ori in head_ori_active])
goals_coloring      = np.array([node_colors[n] for n in goals_active])
action_coloring     = np.array([actions_color_map[a] for a in action_active])

print(r_key_active.shape)
print(fwm_active.shape)

# ----------------------------
# Embeddings & plots
# ----------------------------
variables_active = {
    #'fwm_active': fwm_active,
    #'new_w_value_active': new_w_value_active,
    #'rnn_out_active': rnn_out_active,
    #'rnn_cell_active': rnn_cell_active,
    'r_key_active': r_key_active,
    #'r_value_active': r_value_active,
    #'w_value_active': w_value_active,
    #'w_key_active': w_key_active
}


if plot_PCA:
    for var_name, var_data in variables_active.items():
        # PCA (3D) with different colorings
        _, emb = embedding_f(var_data, dim=3, method='pca', labels=goals_active)

        axs, sc = plot_embedding(emb, dim=3, coloring=head_ori_coloring)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        axs.legend(title=f"{var_name} - Head Orientation", loc='upper right')
        #plt.savefig(f"figs/fig_7f_{var_name}_PCA_head_ori_coloring.png", dpi=300, bbox_inches='tight')

        axs, sc = plot_embedding(emb, dim=3, coloring=agent_node_coloring)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        axs.legend(title=f"{var_name} - Agent Node", loc='upper right')
        #plt.savefig(f"figs/fig_7g_{var_name}_PCA_agent_node_coloring.png", dpi=300, bbox_inches='tight')

        axs, sc = plot_embedding(emb, dim=3, coloring=goals_coloring)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        axs.legend(title=f"{var_name} - Goals", loc='upper right')

        axs, sc = plot_embedding(emb, dim=3, coloring=action_active)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        uniq = np.unique(action_active)
        colors = sc.cmap(sc.norm(uniq))
        handles = [plt.Line2D([0],[0], marker='o', color=c, linestyle='', markersize=10) for c in colors]
        axs.legend(handles, uniq, title=f"{var_name} - Actions", loc='upper right')

        axs, sc = plot_embedding(emb, dim=3, coloring=in_reward_active)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        axs.legend(title=f"{var_name} - in_reward", loc='upper right')

        axs, sc = plot_embedding(emb, dim=3, coloring=retrieving_active)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        uniq = np.unique(retrieving_active)
        colors = sc.cmap(sc.norm(uniq))
        handles = [plt.Line2D([0],[0], marker='o', color=c, linestyle='', markersize=10) for c in colors]
        axs.legend(handles, uniq, title=f"{var_name} - retrieving", loc='upper right')

        axs, sc = plot_embedding(emb, dim=3, coloring=number_rew_active)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        uniq = np.unique(number_rew_active)
        colors = sc.cmap(sc.norm(uniq))
        handles = [plt.Line2D([0],[0], marker='o', color=c, linestyle='', markersize=10) for c in colors]
        axs.legend(handles, uniq, title=f"{var_name} - n of reached", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.show()


# =========================
# draw_heatmaps: wind-rose heatmaps per unit from r_key
# Reuses: r_key_active, agent_node_active, head_ori_active, trajectories, active_indices
# If grid_size != original grid (5×5), nodes are re-quantized from continuous (x,z).
# =========================

if draw_heatmaps:

    # Map any 22.5°-quantized angle to the nearest 45° sector
    def to_main_orient(angle_deg):
        return (int(round((angle_deg % 360) / 45.0)) * 45) % 360

    # Normalize orientations to 8 sectors so they match the orientation panels
    head_ori_8 = np.array([to_main_orient(a) for a in head_ori_active])
    main_orients = [0, 45, 90, 135, 180, 225, 270, 315]

    def _rebuild_active_positions():
        """
        Rebuild continuous positions for the *flattened* timeline and pick the active subset.
        Mirrors the original flattening (episode -> steps) so indices align with *_full arrays.
        """
        pos_x_full, pos_z_full = [], []
        for eps in range(len(memory_traces)):
            # Flatten episode trajectory [(x,z,theta), ...]
            traj_eps = [it for sub in trajectories[eps] for it in sub]
            # Use the same step range used to build *_full in the first part
            # (the original code used len(retrieving[0]) for each episode)
            T_eps = len(retrieving[0])
            for i in range(T_eps):
                x_i, z_i = traj_eps[i][0], traj_eps[i][1]
                pos_x_full.append(x_i)
                pos_z_full.append(z_i)
        pos_x_full = np.array(pos_x_full)
        pos_z_full = np.array(pos_z_full)
        # Pick the already-selected active indices
        return pos_x_full[active_indices], pos_z_full[active_indices]

    def generate_heatmaps_r_key_windrose(r_key_full, agent_node_full, head_ori_full, grid_size=5,
                                       save=False, out_dir=None, cmap=plt.cm.turbo):
        """
        For each unit (dimension) of r_key:
          - Build a global heatmap (left) across all orientations.
          - Build eight orientation-conditioned heatmaps (right) arranged as a wind-rose:
            {0, 45, 90, 135, 180, 225, 270, 315} degrees.
        If grid_size differs from the original node quantization, nodes are recomputed
        from continuous (x,z) positions for the active subset.
        Returns:
          global_maps: list[D] of (G,G) arrays
          orientation_maps: list[D] of dict {angle -> (G,G)} arrays
        """
        T, D = r_key_full.shape
        assert len(head_ori_full) == T, "head_ori_full must match r_key length."
        assert len(agent_node_full) == T, "agent_node_full must match r_key length."

        # Detect original grid from provided node indices (e.g., 0..24 -> 5×5)
        orig_max = int(np.max(agent_node_full))
        orig_grid = int(round(np.sqrt(orig_max + 1)))
        # If target grid_size != original, re-quantize nodes from continuous positions
        if grid_size != orig_grid:
            pos_x_act, pos_z_act = _rebuild_active_positions()
            # Recompute node ids at the requested grid size using the provided helper
            nodes = np.empty(T, dtype=np.int32)
            for t, (x, z) in enumerate(zip(pos_x_act, pos_z_act)):
                nodes[t] = coordinates_to_node(x, z, grid_size=grid_size)
        else:
            nodes = np.asarray(agent_node_full, dtype=np.int32)

        # Accumulate sums and counts for global and per-orientation maps
        global_maps, orientation_maps = [], []
        for d in range(D):
            sums_g = np.zeros(grid_size * grid_size, dtype=np.float64)
            cnts_g = np.zeros_like(sums_g, dtype=np.int64)
            sums_o = {o: np.zeros(grid_size * grid_size, dtype=np.float64) for o in main_orients}
            cnts_o = {o: np.zeros(grid_size * grid_size, dtype=np.int64) for o in main_orients}

            vals = r_key_full[:, d]
            for n, o, v in zip(nodes, head_ori_full, vals):
                sums_g[n] += v; cnts_g[n] += 1
                if o in sums_o:      # guard against any stray angle
                    sums_o[o][n] += v; cnts_o[o][n] += 1

            gmap = np.full(grid_size * grid_size, np.nan, dtype=np.float64)
            m = cnts_g > 0
            gmap[m] = sums_g[m] / cnts_g[m]
            gmap = gmap.reshape(grid_size, grid_size)

            omaps = {}
            for o in main_orients:
                arr = np.full(grid_size * grid_size, np.nan, dtype=np.float64)
                m = cnts_o[o] > 0
                arr[m] = sums_o[o][m] / cnts_o[o][m]
                omaps[o] = arr.reshape(grid_size, grid_size)

            global_maps.append(gmap)
            orientation_maps.append(omaps)

        # Optional: save a combined figure per unit (global + wind-rose layout)
        if save:
            assert out_dir is not None, "Provide out_dir when save=True."
            os.makedirs(out_dir, exist_ok=True)

            map_size = grid_size
            sep = 1
            cross_size = 3 * map_size + 2 * sep
            combined_h = cross_size
            combined_w = map_size + 1 + cross_size  # global | gap | wind-rose

            def pos(row, col):
                r = row * map_size + row * sep
                c = col * map_size + col * sep
                return r, c

            pos_dict = {
                0:   pos(0, 1),  # N
                45:  pos(0, 0),  # NW
                90:  pos(1, 0),  # W
                135: pos(2, 0),  # SW
                180: pos(2, 1),  # S
                225: pos(2, 2),  # SE
                270: pos(1, 2),  # E
                315: pos(0, 2),  # NE
            }

            for d, (gmap, omaps) in enumerate(zip(global_maps, orientation_maps)):
                combined = np.full((combined_h, combined_w), np.nan, dtype=np.float32)

                # Global map (left, vertically centered)
                gtop = (combined_h - map_size) // 2
                combined[gtop:gtop + map_size, 0:map_size] = gmap

                # Wind-rose (right)
                windrose = np.full((cross_size, cross_size), np.nan, dtype=np.float32)
                for o, (r0, c0) in pos_dict.items():
                    windrose[r0:r0 + map_size, c0:c0 + map_size] = omaps[o]
                combined[:, map_size + 1:] = windrose

                # Per-figure normalization to [0,1] (clip negatives)
                arr = np.clip(combined, a_min=0, a_max=None)
                vmax = np.nanmax(arr) if np.isfinite(np.nanmax(arr)) else 1.0
                if vmax > 0:
                    arr = arr / vmax

                plt.figure()
                im = plt.imshow(arr, origin='upper', cmap=cmap, vmin=0, vmax=1)
                plt.colorbar(im)
                plt.xticks([]); plt.yticks([]); plt.axis('off')
                plt.title(f"r_key: Unit {d}  (grid {grid_size}×{grid_size})")
                out_path = os.path.join(out_dir, f"unit_{d}.png")
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved: {out_path}")

        return global_maps, orientation_maps

    # --- Run: you can switch grid_size between 5 and 10; for 10 it re-quantizes from (x,z)
    global_maps_hm, orientation_maps_hm = generate_heatmaps_r_key_windrose(
        r_key_full=r_key_active,
        agent_node_full=agent_node_active,  # used if grid_size == original; ignored otherwise
        head_ori_full=head_ori_8,
        grid_size=10,
        save=True,
        out_dir="figs/fig_7_units_ratemaps_continuous"
    )


# =========================
# unit_classification: classify units (place / HD-modulated / non-place) from r_key
# Reuses: r_key_active, agent_node_active, head_ori_active, trajectories, memory_traces, retrieving, active_indices
# If grid_size != original (e.g., 10), nodes are re-quantized from continuous (x,z).
# =========================

if unit_classification:
    # ----------------------------
    # Helpers
    # ----------------------------
    def to_main_orient(angle_deg):
        """Map angles (multiples of 22.5) to the nearest 45° sector {0,45,...,315}."""
        return (int(round((angle_deg % 360) / 45.0)) * 45) % 360

    # Coerce head orientations to 8-sector main angles (0..315 step 45)
    head_ori_8 = np.array([to_main_orient(a) for a in head_ori_active])
    orient_list = [0, 45, 90, 135, 180, 225, 270, 315]

    def _rebuild_active_positions():
        """
        Rebuild continuous positions for the flattened timeline and select the active subset.
        Mirrors the original flattening (episode -> steps) so indices align with *_full arrays.
        """
        pos_x_full, pos_z_full = [], []
        for eps in range(len(memory_traces)):
            traj_eps = [it for sub in trajectories[eps] for it in sub]
            T_eps = len(retrieving[0])  # same length used in the first build loop
            for i in range(T_eps):
                pos_x_full.append(traj_eps[i][0])
                pos_z_full.append(traj_eps[i][1])
        pos_x_full = np.array(pos_x_full)
        pos_z_full = np.array(pos_z_full)
        return pos_x_full[active_indices], pos_z_full[active_indices]

    def _nodes_for_grid(agent_nodes, grid_size):
        """
        If the provided nodes match the requested grid_size, return them.
        Otherwise, re-quantize from continuous positions (x,z).
        """
        # infer original grid from max node index (e.g., 0..24 -> 5×5)
        orig_max = int(np.max(agent_nodes))
        orig_grid = int(round(np.sqrt(orig_max + 1)))
        if grid_size == orig_grid:
            return np.asarray(agent_nodes, dtype=np.int32)
        # re-quantize from continuous positions
        pos_x_act, pos_z_act = _rebuild_active_positions()
        nodes = np.empty(pos_x_act.shape[0], dtype=np.int32)
        for t, (x, z) in enumerate(zip(pos_x_act, pos_z_act)):
            nodes[t] = coordinates_to_node(x, z, grid_size=grid_size)
        return nodes

    def build_rate_maps_r_key(r_key, agent_nodes, head_oris, grid_size=5):
        """
        For each unit (dimension) in r_key, build:
          - one global (orientation-agnostic) rate map (GxG)
          - eight orientation-conditioned rate maps for angles in orient_list
        Returns:
          global_maps: list[D] of (G,G)
          orientation_maps: list[D] of dict {angle -> (G,G)}
        """
        T, D = r_key.shape
        assert head_oris.shape[0] == T, "Length mismatch."
        # Ensure nodes correspond to the requested grid (re-quantize if needed)
        nodes = _nodes_for_grid(agent_nodes, grid_size)

        global_maps, orientation_maps = [], []
        for d in range(D):
            sums_global = np.zeros(grid_size * grid_size, dtype=np.float64)
            cnts_global = np.zeros_like(sums_global, dtype=np.int64)
            sums_or = {o: np.zeros(grid_size * grid_size, dtype=np.float64) for o in orient_list}
            cnts_or = {o: np.zeros(grid_size * grid_size, dtype=np.int64) for o in orient_list}

            vals = r_key[:, d]
            for n, o, v in zip(nodes, head_oris, vals):
                sums_global[n] += v; cnts_global[n] += 1
                sums_or[o][n] += v; cnts_or[o][n] += 1

            gmap = np.full(grid_size * grid_size, np.nan, dtype=np.float64)
            m = cnts_global > 0
            gmap[m] = sums_global[m] / cnts_global[m]
            gmap = gmap.reshape(grid_size, grid_size)

            omaps = {}
            for o in orient_list:
                arr = np.full(grid_size * grid_size, np.nan, dtype=np.float64)
                m = cnts_or[o] > 0
                arr[m] = sums_or[o][m] / cnts_or[o][m]
                omaps[o] = arr.reshape(grid_size, grid_size)

            global_maps.append(gmap)
            orientation_maps.append(omaps)

        return global_maps, orientation_maps

    # Build rate maps using the already computed active views
    global_maps, orientation_maps = build_rate_maps_r_key(
        r_key=r_key_active,
        agent_nodes=agent_node_active,   # will be ignored for node values if grid_size changes
        head_oris=head_ori_8,
        grid_size=10                     # works now: nodes are re-quantized from (x,z)
    )

    # Flatten orientation maps into a list per unit: [maps at angles in orient_list]
    rate_maps_per_unit = [
        [orientation_maps[d][ang] for ang in orient_list]
        for d in range(len(orientation_maps))
    ]

    # ----------------------------
    # Metrics
    # ----------------------------
    try:
        from scipy.ndimage import rotate
        _can_rotate = True
    except Exception:
        _can_rotate = False
        def rotate(x, angle=0, reshape=False, order=1, mode="nearest"):
            return x  # no-op

    def skaggs_info(rate_map, eps=1e-12):
        """Skaggs spatial information (approx) assuming uniform occupancy."""
        rm = np.nan_to_num(rate_map, nan=0.0)
        r_bar = rm.mean()
        if r_bar < eps:
            return 0.0
        p = 1.0 / rm.size
        return np.nansum(p * rm / (r_bar + eps) * np.log2((rm + eps) / (r_bar + eps)))

    def pairwise_mean_corr(vecs):
        """Mean Pearson correlation over all i<j pairs; ignores NaNs from degenerate vectors."""
        n = len(vecs)
        if n < 2:
            return np.nan
        cs = []
        for i in range(n):
            for j in range(i + 1, n):
                c = np.corrcoef(vecs[i], vecs[j])[0, 1]
                if np.isfinite(c):
                    cs.append(c)
        return float(np.mean(cs)) if len(cs) else np.nan

    def unit_metrics(rate_maps, angles_deg):
        """
        Returns: S_space, S_rot, R_vec, I_spatial, F_max
        - S_space: mean corr without rotation
        - S_rot:   mean corr after counter-rotating each map by its angle
        - R_vec:   circular resultant length weighted by mean rate
        - I_spatial: Skaggs info on the global map (mean across orientations)
        - F_max:   peak of the global map
        """
        vec = [m.flatten() for m in rate_maps]
        S_space = pairwise_mean_corr(vec)

        if _can_rotate:
            rot_maps = [rotate(m, angle=-a, reshape=False, order=1, mode="nearest")
                        for m, a in zip(rate_maps, angles_deg)]
            S_rot = pairwise_mean_corr([m.flatten() for m in rot_maps])
        else:
            S_rot = np.nan

        mean_rate = np.array([np.nanmean(m) for m in rate_maps])
        ang = np.deg2rad(np.array(angles_deg))
        R_vec = np.abs(np.sum(mean_rate * np.exp(1j * ang))) / (np.sum(np.abs(mean_rate)) + 1e-12)

        gmap = np.nanmean(rate_maps, axis=0)
        I_spatial = skaggs_info(gmap)
        F_max = float(np.nanmax(gmap))

        return S_space, S_rot, R_vec, I_spatial, F_max

    # ----------------------------
    # Classification
    # ----------------------------
    TH_F_ABS = 0.25   # minimal absolute peak in the global map
    TH_I     = 0.30   # reserved for future filtering
    TH_Sp    = 0.15   # min unrotated similarity
    TH_Sr    = 0.15   # min rotated similarity
    TH_R     = 0.15   # min vector length

    labels = []
    for d, maps in enumerate(rate_maps_per_unit):
        S_sp, S_rot, R_vec, I_sp, F_max = unit_metrics(maps, orient_list)
        label = "Non-def"
        if all(np.nanmax(m) < TH_F_ABS for m in maps):
            label = "Non-PC"
        elif (np.isfinite(S_sp) and np.isfinite(S_rot)) and (S_sp > S_rot) and (S_sp >= TH_Sp):
            label = "PC"
        elif (np.isfinite(S_rot) and S_rot >= TH_Sr):
            # if R_vec >= TH_R:  # enable for stricter HD criterion
            label = "HD-PC"
        labels.append(label)

    # ----------------------------
    # Summary
    # ----------------------------
    units_place   = [i for i, c in enumerate(labels) if c == "PC"]
    units_hdpc    = [i for i, c in enumerate(labels) if c == "HD-PC"]
    units_nonpc   = [i for i, c in enumerate(labels) if c == "Non-PC"]
    units_nondef  = [i for i, c in enumerate(labels) if c == "Non-def"]

    total = len(labels)
    print("\n==========  FINAL CLASSIFICATION  ==========")
    print(f"Total units: {total}")
    print(f"  • Place Cells              : {len(units_place)}  ({100*len(units_place)/max(1,total):.1f}%)")
    print(f"  • HD-modulated Place Cells : {len(units_hdpc)}  ({100*len(units_hdpc)/max(1,total):.1f}%)")
    print(f"  • Non-Place Cells          : {len(units_nonpc)}  ({100*len(units_nonpc)/max(1,total):.1f}%)")
    print(f"  • Non-def Cells            : {len(units_nondef)} ({100*len(units_nondef)/max(1,total):.1f}%)\n")

    classification = {
        "PC": units_place,
        "HD-PC": units_hdpc,
        "Non-PC": units_nonpc,
        "Non-def": units_nondef,
    }
    print(classification)

    # ----------------------------
    # Pie chart (compact)
    # ----------------------------
    n_pc, n_hdpc = len(units_place), len(units_hdpc)
    n_nonplace   = len(units_nonpc) + len(units_nondef)

    sizes = [n_pc, n_hdpc, n_nonplace]
    labels_multiline = [
        f'Place\nUnits (n={n_pc})',
        f'HD-modulated\nUnits (n={n_hdpc})',
        f'Non-Place\nUnits (n={n_nonplace})'
    ]
    colors = ['#4c72b0', '#55a868', '#c44e52']

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'color': 'black'},
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'gray'}
    )
    plt.setp(autotexts, size=14, weight='bold')

    for i, w in enumerate(wedges):
        ang = 0.5 * (w.theta1 + w.theta2)
        x, y = 1.25 * np.cos(np.deg2rad(ang)), 1.25 * np.sin(np.deg2rad(ang))
        ax.text(x, y, labels_multiline[i], ha='center', va='center', fontsize=12, weight='bold')

    ax.axis('equal')
    plt.tight_layout()
    plt.show()
