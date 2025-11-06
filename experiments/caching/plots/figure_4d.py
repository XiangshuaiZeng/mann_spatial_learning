"""
Simple 1D Goal–Agent Model
------------------------------------------------
- 1D world with 5 discrete nodes (1..5)
- Generates random (agent, goal) pairs
- Adds Gaussian jitter to each coordinate
- Colors nodes smoothly from green → yellow
- Labels events as 'Cache' (agent == goal) or 'Visit' (agent != goal)
- Plots three 2D figures:
    1. Scatter colored by Agent
    2. Scatter colored by Goal
    3. Cache vs Visit comparison
- Plots Euclidean similarity vs. agent distance
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -------------------------------------------------------------
# 1) Arguments
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="Simple 1D Goal–Agent Model")
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--outdir', type=Path, default=Path('figs'))
parser.add_argument('--sigma_cache', type=float, default=0.2)
parser.add_argument('--sigma_visit', type=float, default=0.2)
args = parser.parse_args()

np.random.seed(args.seed)
args.outdir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 2) Define 1D world and color palette (interpolated from green→yellow)
# -------------------------------------------------------------
nodes = np.array([1, 2, 3, 4, 5])
color_left = np.array(mcolors.to_rgb('green'))
color_right = np.array(mcolors.to_rgb('yellow'))

palette = {}
for i, n in enumerate(nodes):
    t = i / (len(nodes) - 1)  # linear interpolation 0..1
    c = (1 - t) * color_left + t * color_right
    palette[n] = mcolors.to_hex(c)

# -------------------------------------------------------------
# 3) Generate random (agent, goal) pairs + Gaussian jitter
# -------------------------------------------------------------
T = int(args.steps)
agent = np.random.choice(nodes, size=T)
goal = np.random.choice(nodes, size=T)

labels = np.where(agent == goal, 'Cache', 'Visit')
points = np.empty((T, 2), dtype=float)

for t, (ag, go) in enumerate(zip(agent, goal)):
    sigma = args.sigma_cache if ag == go else args.sigma_visit
    points[t, 0] = ag + np.random.normal(0, sigma)
    points[t, 1] = go + np.random.normal(0, sigma)

colors_agent = [palette[ag] for ag in agent]
colors_goal = [palette[go] for go in goal]

plt.rcParams["svg.fonttype"] = "path"  # make text outlines for portability

# -------------------------------------------------------------
# 4) Scatter plots: by Agent, by Goal, and Cache vs Visit
# -------------------------------------------------------------
# (A) Colored by Agent
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(points[:, 0], points[:, 1], s=40, c=colors_agent, alpha=0.8, edgecolors='none')
ax.set_xlabel('Agent position', fontsize=13)
ax.set_ylabel('Goal position', fontsize=13)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_yticks([1, 2, 3, 4, 5])
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig(args.outdir / 'fig_4d_scatter_agent.svg')
plt.close(fig)

# (B) Colored by Goal
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(points[:, 0], points[:, 1], s=40, c=colors_goal, alpha=0.8, edgecolors='none')
ax.set_xlabel('Agent position', fontsize=13)
ax.set_ylabel('Goal position', fontsize=13)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_yticks([1, 2, 3, 4, 5])
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig(args.outdir / 'fig_4d_scatter_goal.svg')
plt.close(fig)

# (C) Cache vs Visit comparison
idx_cache = np.where(labels == 'Cache')[0]
idx_visit = np.where(labels == 'Visit')[0]

fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(points[idx_cache, 0], points[idx_cache, 1], s=40, c='#e67300', alpha=0.8, edgecolors='none', label='Cache')
ax.scatter(points[idx_visit, 0], points[idx_visit, 1], s=40, c='gray', alpha=0.8, edgecolors='none', label='Visit')
ax.set_xlabel('Agent position', fontsize=13)
ax.set_ylabel('Goal position', fontsize=13)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_yticks([1, 2, 3, 4, 5])
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower left', fontsize=10)
fig.tight_layout()
fig.savefig(args.outdir / 'fig_4d_goal_agent_cache.svg')
plt.close(fig)

# -------------------------------------------------------------
# 5) Similarity vs distance (Cache–Cache vs Cache–Visit)
# -------------------------------------------------------------

def agent_distance(a, b):
    return abs(int(a) - int(b))


def euclid_similarity(v1, v2, max_dist=6.0):
    """Normalized Euclidean similarity in [0, 1]."""
    return 1.0 - np.linalg.norm(np.ravel(v1) - np.ravel(v2)) / max_dist

cc, cv = defaultdict(list), defaultdict(list)

for i in range(T):
    for j in range(i + 1, T):
        d = agent_distance(agent[i], agent[j])
        s = euclid_similarity(points[i], points[j])
        li, lj = labels[i], labels[j]
        if li == 'Cache' and lj == 'Cache':
            cc[d].append(s)
        elif li != lj:  # Cache–Visit pairs
            cv[d].append(s)

# Compute mean and std per distance
D_cc = sorted(cc)
M_cc = [np.mean(cc[d]) for d in D_cc]
S_cc = [np.std(cc[d]) for d in D_cc]

D_cv = sorted(cv)
M_cv = [np.mean(cv[d]) for d in D_cv]
S_cv = [np.std(cv[d]) for d in D_cv]

# Plot correlation curve
fig, ax = plt.subplots(figsize=(3, 3))
ax.errorbar(D_cc, M_cc, yerr=S_cc, fmt='o--', label='cache–cache', capsize=5, color='#e67300')
ax.errorbar(D_cv, M_cv, yerr=S_cv, fmt='o--', label='cache–visit', capsize=5, color='gray')
ax.set_xlabel('Agent distance', fontsize=13)
ax.set_ylabel('Similarity', fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(-0.1, 4.1)
ax.legend(loc='lower left', fontsize=10)
fig.tight_layout()
fig.savefig(args.outdir / 'fig_4d_corr_model.svg')
plt.close(fig)