"""
Figure 7 (F–J) – Discrete State Space Visualizations
====================================================

This script mirrors the behavior and layout of the provided
`figure_7_PCA_continuous.py`, but adapted to **discrete** state space
(5×5 grid of node IDs with cardinal head orientations).

What you get (same panels as continuous version):
- **(F, G)**: Low-dimensional embeddings (e.g., PCA) of selected memory traces with
  multiple colorings (head orientation, agent node, goal, action, in-reward, retrieving, etc.).
- **(H, I)**: Orientation-conditioned **rate maps** (“wind-rose” layout) for each unit of
  `r_key` at the native grid resolution.
- **(J)**: **Unit classification** into Place (PC), Head-Direction-modulated Place (HD-PC),
  Non-Place (Non-PC), or Non-def, based on spatial/rotational similarity and simple
  heuristics. A compact summary (counts + pie chart) is produced.

How to use the on/off switches (same idea as the continuous script)
------------------------------------------------------------------
- `plot_PCA`            → controls Figure 7F–G style embeddings.
- `draw_heatmaps`       → controls Figure 7H–I style rate maps (global + wind-rose).
- `unit_classification` → controls Figure 7J style classification and summary plots.

Active step selection
---------------------
Edit `ACTIVE_IDX_RULE` to decide which frames are included (e.g., retrieval-only,
exclude resets, restrict to specific head direction, etc.).

In short
--------
1) Point `path` to your **discrete** dataset.
2) Set `ACTIVE_IDX_RULE` to the subset you want (e.g., retrieval-only, facing North).
3) Toggle `plot_PCA`, `draw_heatmaps`, and `unit_classification`.
"""

import os
import numpy as np
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

# ============================
# Config: path & active-index rules
# ============================
# Example: change to your discrete-run pickle path
path = "/local/jon/repositories/mann_spatial_learning-main_OLD/experiments/new_homecage_Jon_redwall_encoding_invisible_ppo_diff_3_cnn_train_run2/data/test/ppo_homecage_Jon_redwall_encoding_invisible_fwm_1_2000_50ep_2000_relu.pkl"

# Figure 7 subsetting (edit as needed)
# e.g. retrieval_phase w/o resets:           (number_rew_full > 1) & (action_full != 4)
# restrict to a head orientation (e.g. 0°):  (number_rew_full > 1) & (action_full != 4) & (head_ori_full == 0)
# all frames (for rate maps):                (number_rew_full >= 0)
ACTIVE_IDX_RULE = lambda number_rew_full, action_full, head_ori_full: np.where((number_rew_full > 1) & (action_full != 4) & (head_ori_full == 0))[0]

# Analysis toggles
plot_PCA = True
draw_heatmaps = False
unit_classification = True

# ============================
# Grid layout (node indices)
# ============================
positions = np.array([
    [4, 9, 14, 19, 24],
    [3, 8, 13, 18, 23],
    [2, 7, 12, 17, 22],
    [1, 6, 11, 16, 21],
    [0, 5, 10, 15, 20]
])
GRID_SIZE = 5  # native discrete grid

# Helper: orientation mapping (dataset typically uses {0, 90, -90, -180})
# We normalize to the 4 cardinal angles {0, 90, 180, 270} for plotting,
# and also provide a function to map to 8-sector (45°) if ever needed.

def normalize_cardinal(ang):
    # map {-180, 180}→180; {-90}→270; {0}→0; {90}→90
    a = int(ang)
    if a == -180: a = 180
    if a == -90:  a = 270
    return a % 360

def to_main_8(ang):
    # to nearest 45° (0,45,...,315)
    return (int(round((ang % 360) / 45.0)) * 45) % 360

# ============================
# Load data
# ============================
with open(path, 'rb') as f:
    data = pickle.load(f)

step_reward   = data['step_reward']
trajectories  = data['trajectories']     # per-episode: list of lists of (node, orientation)
goals         = data['goal_location']    # per-episode sequence of goal node IDs (may be [[node], ...])
retrieving    = data['retrieving']
memory_traces = data['memory_traces']
reached       = data['reached']          # step-wise count of rewards reached
actions       = data['action']           # may contain strings like "reset"

# ============================
# Small utilities
# ============================

def interpolate_color(color1, color2, factor):
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    return mcolors.to_hex((1 - factor) * c1 + factor * c2)


def embedding_f(data, dim, method='pca', labels=None):
    """Return (components, embedding). Flattens last dim if needed."""
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

# ============================
# Build flattened timelines (match the continuous script’s structure)
# ============================
r_value_full, r_key_full, w_value_full, new_w_value_full = [], [], [], []
w_key_full, rnn_cell_full, rnn_out_full, w_beta_full = [], [], [], []
fwm_full, delta_fwm_full = [], []
retrieving_full, goals_full = [], []
agent_node_full, head_ori_full, action_full = [], [], []
in_reward_full, number_rew_full, sequence, step_full = [], [], [], []

has_fwm = all(('fwm' in mt and 'delta_fwm' in mt) for mt in memory_traces)

for eps in range(len(memory_traces)):
    eps_r_value     = np.squeeze(memory_traces[eps]['r_value'])
    eps_r_key       = np.squeeze(memory_traces[eps]['r_key'])
    eps_w_value     = np.squeeze(memory_traces[eps]['w_value'])
    eps_new_w_value = np.squeeze(memory_traces[eps]['new_w_value'])
    eps_w_key       = np.squeeze(memory_traces[eps]['w_key'])
    eps_rnn_cell    = np.squeeze(memory_traces[eps]['rnn_cell'])
    eps_rnn_out     = np.squeeze(memory_traces[eps]['rnn_out'])
    eps_w_beta      = np.squeeze(memory_traces[eps]['w_beta'])

    # Optional FWM
    if has_fwm:
        eps_fwm_full  = np.squeeze(memory_traces[eps]['fwm'])
        eps_delta_fwm = np.squeeze(memory_traces[eps]['delta_fwm'])

    # Flatten episode trajectories: list of (node, orientation)
    traj_eps = [it for sub in trajectories[eps] for it in sub]

    # Goals can be [[node], ...] or [node, ...] → unify to [node, ...]
    goals_eps = [g[0] if isinstance(g, (list, tuple, np.ndarray)) else g for g in goals[eps]]

    # Match original behavior: shift reached by one (prepend 0)
    reached[eps] = [0] + reached[eps][:-1]

    # Normalize actions ("reset"→4; else int)
    actionss = actions[eps]
    actionss = np.array([e[0] if isinstance(e, list) else e for e in actionss])
    actionss = np.array([4 if a == "reset" else int(a) for a in actionss])

    retrieval = []
    counter = 0
    T_eps = len(retrieving[0])  # keep same length convention across episodes
    for i in range(T_eps):
        step_full.append(i)

        node_i, ori_i = traj_eps[i]
        agent_node_full.append(int(node_i))
        head_ori_full.append(normalize_cardinal(ori_i))

        number_rew_full.append(reached[eps][i])
        # in-reward if agent node equals previous goal
        in_reward_full.append(1 if node_i == goals_eps[i - 1] else 0)
        if node_i == goals_eps[i - 1]:
            counter = 0
        sequence.append(counter)
        counter += 1

        retrieval.append(0 if retrieving[eps][i] is False else 1)

    # Extend arrays
    r_value_full.extend(eps_r_value);       r_key_full.extend(eps_r_key)
    w_value_full.extend(eps_w_value);       new_w_value_full.extend(eps_new_w_value)
    w_key_full.extend(eps_w_key);           rnn_cell_full.extend(eps_rnn_cell)
    rnn_out_full.extend(eps_rnn_out);       w_beta_full.extend(eps_w_beta)
    action_full.extend(actionss);           retrieving_full.extend(retrieval)
    goals_full.extend(goals_eps)

    if has_fwm:
        fwm_full.extend(eps_fwm_full)
        delta_fwm_full.extend(eps_delta_fwm)

# Negative sequence (distance to last reward index)
negative_sequence = []
last_idx = -1
for i, val in enumerate(in_reward_full):
    v = 1 if i == 0 else val
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

# To numpy
r_value_full = np.array(r_value_full);      r_key_full = np.array(r_key_full)
w_value_full = np.array(w_value_full);      new_w_value_full = np.array(new_w_value_full)
w_key_full   = np.array(w_key_full);        rnn_cell_full = np.array(rnn_cell_full)
rnn_out_full = np.array(rnn_out_full);      w_beta_full = np.array(w_beta_full)
retrieving_full = np.array(retrieving_full)

if has_fwm:
    fwm_full = np.array(fwm_full);          delta_fwm_full = np.array(delta_fwm_full)
else:
    fwm_full = np.zeros((r_key_full.shape[0], 1), dtype=np.float32)
    delta_fwm_full = np.zeros_like(fwm_full)

number_rew_full = np.array(number_rew_full)
head_ori_full   = np.array(head_ori_full)
agent_node_full = np.array(agent_node_full)
goals_full      = np.array(goals_full)
action_full     = np.array(action_full)
in_reward_full  = np.array(in_reward_full)
negative_sequence_full = np.array(negative_sequence)
sequence_full   = np.array(sequence)
step_full       = np.array(step_full)

# ============================
# Colors (node palette identical style as continuous)
# ============================
fixed_colors = {
    0:'#FF0000', 4:'#0000FF', 12:'#000000', 20:'#FFFF00', 24:'#008000'
}
coords_fixed = {0:(4,0), 4:(0,0), 12:(2,2), 20:(4,4), 24:(0,4)}

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

# Optional: visualize node palette
fig, ax = plt.subplots()
for i in range(positions.shape[0]):
    for j in range(positions.shape[1]):
        node = positions[i, j]
        rect = plt.Rectangle((j, i), 1, 1, facecolor=node_colors[node])
        ax.add_patch(rect)
plt.xlim(0, positions.shape[1]); plt.ylim(0, positions.shape[0])
plt.gca().invert_yaxis(); plt.axis('off')
# plt.show()

# Extra colormaps
actions_color_map = {3:'black', 4:'pink', 2:'green', 1:'yellow', 0:'purple'}
cardinal_color_map = {0:'#00aa00', 90:'#0088ff', 180:'#aa00aa', 270:'#ff8800'}

# ============================
# Active selection & slicing
# ============================
active_indices = ACTIVE_IDX_RULE(number_rew_full, action_full, head_ori_full)
if len(active_indices) > 10000:
    active_indices = active_indices[:10000]

r_value_active     = r_value_full[active_indices]
r_key_active       = r_key_full[active_indices]
w_value_active     = w_value_full[active_indices]
new_w_value_active = new_w_value_full[active_indices]
w_key_active       = w_key_full[active_indices]
rnn_cell_active    = rnn_cell_full[active_indices]
rnn_out_active     = rnn_out_full[active_indices]
fwm_active         = fwm_full[active_indices]

number_rew_active       = number_rew_full[active_indices]
head_ori_active         = head_ori_full[active_indices]
agent_node_active       = agent_node_full[active_indices]
goals_active            = goals_full[active_indices]
action_active           = action_full[active_indices]
in_reward_active        = in_reward_full[active_indices]
negative_sequence_active= negative_sequence_full[active_indices]
sequence_active         = sequence_full[active_indices]
retrieving_active       = retrieving_full[active_indices]

# Quick step table (first 60 rows)
step_data = [{
    'Step': s + 1,
    'Agent Node': agent_node_full[s],
    'Goal': goals_full[s],
    'retrieving': retrieving_full[s],
    'In Reward': in_reward_full[s],
    'Numero de reward': number_rew_full[s],
    'Action': action_full[s],
} for s in range(0, min(60, len(agent_node_full)))]
print(pd.DataFrame(step_data))

# Color encodings
agent_node_coloring = np.array([node_colors[n] for n in agent_node_active])
head_ori_coloring   = np.array([cardinal_color_map[n] for n in head_ori_active])
goals_coloring      = np.array([node_colors[n] for n in goals_active])
action_coloring     = np.array([actions_color_map[a] for a in action_active])

print("r_key_active shape:", r_key_active.shape)
print("fwm_active shape:", fwm_active.shape)

# ============================
# (F–G) Embeddings
# ============================
variables_active = {
    'r_key_active': r_key_active,
    # 'r_value_active': r_value_active,
    # 'w_value_active': w_value_active,
    # 'rnn_out_active': rnn_out_active,
}

if plot_PCA:
    os.makedirs("figs", exist_ok=True)
    for var_name, var_data in variables_active.items():
        _, emb = embedding_f(var_data, dim=3, method='pca', labels=goals_active)

        axs, sc = plot_embedding(emb, dim=3, coloring=head_ori_coloring)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        axs.legend(title=f"{var_name} - Head Orientation", loc='upper right')
        #plt.savefig(f"figs/fig_7a_{var_name}_PCA_head_ori_coloring.png", dpi=300, bbox_inches='tight')

        axs, sc = plot_embedding(emb, dim=3, coloring=agent_node_coloring)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_zticks([])
        axs.legend(title=f"{var_name} - Agent Node", loc='upper right')
        plt.savefig(f"figs/fig_7g_{var_name}_PCA_agent_node_coloring.png", dpi=300, bbox_inches='tight')

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

# ============================
# (H–I) Wind-rose rate maps per unit (discrete grid)
# ============================
if draw_heatmaps:
    # We use 4 main orientations (0, 90, 180, 270) for discrete data
    main_orients = [0, 90, 180, 270]
    head_ori_4 = np.array([normalize_cardinal(a) for a in head_ori_active])

    def generate_heatmaps_r_key_windrose_discrete(r_key, agent_nodes, head_oris, grid_size=GRID_SIZE,
                                                  save=False, out_dir=None, cmap=plt.cm.turbo):
        """
        For each unit d of r_key (T×D):
          - Build a global heatmap (grid_size×grid_size) across all orientations.
          - Build four orientation-conditioned heatmaps laid out as a wind-rose (N, E, S, W).
        Returns lists of maps for further metrics/plots.
        """
        T, D = r_key.shape
        agent_nodes = np.asarray(agent_nodes, dtype=np.int32)
        head_oris   = np.asarray(head_oris, dtype=np.int32)

        global_maps, orientation_maps = [], []
        for d in range(D):
            sums_g = np.zeros(grid_size * grid_size, dtype=np.float64)
            cnts_g = np.zeros_like(sums_g, dtype=np.int64)
            sums_o = {o: np.zeros(grid_size * grid_size, dtype=np.float64) for o in main_orients}
            cnts_o = {o: np.zeros(grid_size * grid_size, dtype=np.int64) for o in main_orients}

            vals = r_key[:, d]
            for n, o, v in zip(agent_nodes, head_oris, vals):
                sums_g[n] += v; cnts_g[n] += 1
                if o in sums_o:
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

            # Layout (N at top-center; E right-center; S bottom-center; W left-center)
            pos_dict = {
                0:   pos(0, 1),  # N
                90:  pos(1, 2),  # E
                180: pos(2, 1),  # S
                270: pos(1, 0),  # W
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

                # Normalize per figure to [0,1]
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

    os.makedirs("figs/fig_7_units_ratemaps_discrete", exist_ok=True)
    global_maps_hm, orientation_maps_hm = generate_heatmaps_r_key_windrose_discrete(
        r_key=r_key_active,
        agent_nodes=agent_node_active,
        head_oris=head_ori_4,
        grid_size=GRID_SIZE,
        save=True,
        out_dir="figs/fig_7_units_ratemaps_discrete"
    )

# ============================
# (J) Unit classification
# ============================
if unit_classification:
    try:
        from scipy.ndimage import rotate
        _can_rotate = True
    except Exception:
        _can_rotate = False
        def rotate(x, angle=0, reshape=False, order=1, mode="nearest"):
            return x

    orient_list = [0, 90, 180, 270]
    rate_maps_per_unit = [
        [orientation_maps_hm[d][ang] for ang in orient_list]
        for d in range(len(orientation_maps_hm))
    ]

    def skaggs_info(rate_map, eps=1e-12):
        rm = np.nan_to_num(rate_map, nan=0.0)
        r_bar = rm.mean()
        if r_bar < eps:
            return 0.0
        p = 1.0 / rm.size
        return np.nansum(p * rm / (r_bar + eps) * np.log2((rm + eps) / (r_bar + eps)))

    def pairwise_mean_corr(vecs):
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

    # Thresholds (slightly relaxed for discrete 4-orientation maps)
    TH_F_ABS = 0.25
    TH_Sp    = 0.15
    TH_Sr    = 0.15
    TH_R     = 0.10

    labels = []
    for d, maps in enumerate(rate_maps_per_unit):
        S_sp, S_rot, R_vec, I_sp, F_max = unit_metrics(maps, orient_list)
        label = "Non-def"
        if all(np.nanmax(m) < TH_F_ABS for m in maps):
            label = "Non-PC"
        elif (np.isfinite(S_sp) and np.isfinite(S_rot)) and (S_sp > S_rot) and (S_sp >= TH_Sp):
            label = "PC"
        elif (np.isfinite(S_rot) and S_rot >= TH_Sr):
            # If you need stricter HD criterion, also check R_vec >= TH_R
            label = "HD-PC"
        labels.append(label)

    units_place   = [i for i, c in enumerate(labels) if c == "PC"]
    units_hdpc    = [i for i, c in enumerate(labels) if c == "HD-PC"]
    units_nonpc   = [i for i, c in enumerate(labels) if c == "Non-PC"]
    units_nondef  = [i for i, c in enumerate(labels) if c == "Non-def"]

    total = len(labels)
    print("\n==========  FINAL CLASSIFICATION (DISCRETE)  ==========")
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

    # Compact pie chart
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

