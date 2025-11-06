'''
This script generates all the figures or subpanels used for the guidance task, where the plots for analyzing geometric
computation and the bat experment are generated.
Please turn any of the "draw*" to "True" if you want to plot the corresponding figure.
'''
import os
import json
import pickle
import numpy as np
from math import *
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import curve_fit
import umap
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import matplotlib as mat
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import matplotlib.cm
import matplotlib.colors as mcolors
import seaborn as sns
#plt.switch_backend('Qt5Agg')
## set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 30}
mat.rc('font', **font)

# function for drawing the env topology
def draw_envs(env_top_path, axs):
    with open(env_top_path, 'rb') as handle:
        data = pickle.load(handle)
    # here each node is a XY coordinate and each edge is a list containing 2 integers, indicating a connectivity
    # between 2 nodes
    nodes = data['nodes']
    edges = data['edges']
    walls = data['walls']
    # draw the walls
    for w in walls:
        xy = w[:2]
        wh = w[2:] - w[:2]
        rect = mat.patches.Rectangle(xy, wh[0], wh[1], edgecolor='k', facecolor='none')
        axs.add_patch(rect)
    # draw the edges
    for e in edges:
        axs.plot(nodes[e][:, 0], nodes[e][:, 1], color='k', zorder=1)
    # draw the nodes
    axs.scatter(nodes[:, 0], nodes[:, 1], facecolors='white', edgecolors='b', s=80, zorder=3)
    return nodes
# function for defining arrow pos and orientations
def arrow_info(traj, nodes):
    # draw the arrows that define the trajectory
    arrow_pos = []
    # the X and Y components for each arrow direction
    U = []
    V = []
    for n in traj:
        # define the start and end of the arrow
        arrow_pos.append(nodes[n[0]])
        orientation = n[1]
        if orientation == 0:
            U.append(1)
            V.append(0)
        elif orientation == 90:
            U.append(0)
            V.append(1)
        elif orientation == -90:
            U.append(0)
            V.append(-1)
        elif orientation == -180:
            U.append(-1)
            V.append(0)

    arrow_pos = np.array(arrow_pos)
    return arrow_pos, U, V
def intersect2D(a, b):
    """
    Find row intersection between 2D numpy arrays, a and b.
    Returns another numpy array with shared rows
    """
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])
def extract_state(state_tuple):
    orientations = np.array([0, 90, -180, -90])
    node = state_tuple[0]
    ori = int(state_tuple[1])
    state_idx = node*4 + np.where(orientations==ori)[0][0]
    return state_idx
# define plotting functions
def plot_1D(values, figsize=(5, 5)):
    fig, axs = plt.subplots(figsize=figsize)
    axs.plot(values, color='blue', zorder=2)
    # indicating goal change and reach
    axs.vlines(np.arange(goal_change_freq, episode_length, goal_change_freq), 0, np.max(values), linewidth=1.8, color='black', zorder=1)
    axs.set_ylabel('value')
    axs.set_xlabel('time step')
    plt.tight_layout()
    return fig, axs
def plot_2D(values,  figsize=(5, 5), axs=None, fig=None):
    if axs is None:
        fig, axs = plt.subplots(figsize=figsize)
    im = axs.imshow(values.T, cmap='viridis', aspect='auto')
    # indicating goal change and reach
    axs.vlines(np.arange(goal_change_freq, episode_length, goal_change_freq), values.shape[1], values.shape[1] + 30, linewidth=1.8,
               color='black')
    axs.set_ylabel('dimension')
    axs.set_xlabel('time step')
    axs.set_ylim([values.shape[1] + 15, 0])
    # axs.set_xlim([20, 100])
    cbar = fig.colorbar(im,ax=axs)
    plt.tight_layout()
    return fig, axs
def normalize(values):
    return (values - np.mean(values)) / np.std(values)
def flat_data(data, N=50):
    flattened_data = []
    for i in range(len(data) - N):
        flattened_data.append(np.mean(data[i:i + N]))
    return np.asarray(flattened_data)
def reduce_mean(x):
    return x - np.mean(x)
def normalize(one_d_array):
    return (one_d_array - np.mean(one_d_array)) / np.std(one_d_array)
# convert state ([node, ori]) to [row, col, ori]
def state_to_pos(state, width):
    row = (state // 4) // width  # row = node_idx // width
    col = (state // 4) % width  # col = node_idx % wodth
    ori = state % 4
    return np.array([row, col, ori])
# function for computing relative angles, given relative row, col and HD
def compute_rel_angle(hd, r_row, r_col):
    heading = np.array([np.cos(hd / 180 * pi), np.sin(hd / 180 * pi)])
    heading = heading / np.linalg.norm(heading)
    vec_edge = np.array([r_col, r_row])
    vec_edge = vec_edge / np.linalg.norm(vec_edge)
    angle = np.arctan2(heading[0] * vec_edge[1] - heading[1] * vec_edge[0],
                       heading[0] * vec_edge[0] + heading[1] * vec_edge[1])
    return angle / pi * 180.0
def interpolate_color(color1, color2, factor):
    """Interpolates between two colors by a given factor (0 <= factor <= 1)"""
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    return mcolors.to_hex((1 - factor) * c1 + factor * c2)
###### color coding #############
colores_nodos = {}
maze_shape = (5, 5)

coords_fijos = {
    0: (4, 0),
    4: (0, 0),
    12: (2, 2),
    20: (4, 4),
    24: (0, 4)
}

colores_fijos = {
    0: '#FF0000',  # rojo
    4: '#0000FF',  # azul
    12: '#000000',  # blanco
    20: '#FFFF00',  # amarillo
    24: '#008000'  # verde
}
posiciones = np.array([
    [4, 9, 14, 19, 24],
    [3, 8, 13, 18, 23],
    [2, 7, 12, 17, 22],
    [1, 6, 11, 16, 21],
    [0, 5, 10, 15, 20]
])
for i in range(maze_shape[0]):
    for j in range(maze_shape[1]):
        nodo = posiciones[i, j]
        if nodo not in colores_fijos:
            # Calcular las distancias relativas al nodo central (12) y a los nodos extremos
            dist_to_0 = np.linalg.norm(np.array([i, j]) - np.array(coords_fijos[0]))
            dist_to_4 = np.linalg.norm(np.array([i, j]) - np.array(coords_fijos[4]))
            dist_to_12 = np.linalg.norm(np.array([i, j]) - np.array(coords_fijos[12]))
            dist_to_20 = np.linalg.norm(np.array([i, j]) - np.array(coords_fijos[20]))
            dist_to_24 = np.linalg.norm(np.array([i, j]) - np.array(coords_fijos[24]))

            # Interpolación basada en las distancias
            total_dist = dist_to_0 + dist_to_4 + dist_to_12 + dist_to_20 + dist_to_24
            weight_0 = (total_dist - dist_to_0 * 3) / total_dist
            weight_4 = (total_dist - dist_to_4 * 3) / total_dist
            weight_12 = (total_dist - dist_to_12 * 3) / total_dist
            weight_20 = (total_dist - dist_to_20 * 3) / total_dist
            weight_24 = (total_dist - dist_to_24 * 3) / total_dist
            weight_12 *= 0.1

            # Color interpolado basado en las ponderaciones
            color_interpolado = interpolate_color(
                interpolate_color(
                    interpolate_color(
                        interpolate_color(colores_fijos[0], colores_fijos[4], weight_4 / (weight_0 + weight_4)),
                        colores_fijos[12], weight_12 / (weight_0 + weight_4 + weight_12)),
                    colores_fijos[20], weight_20 / (weight_0 + weight_4 + weight_12 + weight_20)),
                colores_fijos[24], weight_24 / (weight_0 + weight_4 + weight_12 + weight_20 + weight_24))
            colores_nodos[nodo] = color_interpolado
colores_nodos.update(colores_fijos)


################### HERE START THE REAL PLOTS  ##########################
# we specify the experiment that we want to plot
action_space = 'egocentric' # egocentric | allocentric
activation = 'relu'
dsnon = False
maze_size = 5
save_dir = 'homecage_Jon_%s_%s_kv%s_newFWM' % (maze_size, action_space, activation)
# load the parameters
with open(save_dir + '/args.txt', 'r') as f:
    args = json.load(f)
num_states = 100
episode_length = args['test_max_step']
goal_change_freq = episode_length // 2
sub_eps_bounds = np.arange(0, episode_length + 1, goal_change_freq)
network = args['network']

#### For Fig. 5a and supplementary Fig. 7a ####
## we visualize the writing strength and the FWM matrix norm
draw1 = False
if draw1:
    env = args['env']
    simu = 1
    stage = args['training_episodes'] # 500, 1000, 1500, args['training_episodes']
    episode = 77
    sub_eps_bounds = np.arange(0, episode_length + 1, goal_change_freq)
    # load the data
    data = pickle.load(open(save_dir + '/data/test/ppo_%s_%s_%s_%s.pkl' % (env, network, simu, stage), 'rb'))
    goals = data['goal_location']
    memory_traces = data['memory_traces']
    trajs = data['trajectories']
    # extract when goal is reached and changed
    step_reward = np.squeeze(np.array(data['step_reward'][episode]))
    goal_reaching = np.where(step_reward > 0)[0]+1
    # we revover the Fast weight matrix for each episode
    eps_new_w_value = np.squeeze(memory_traces[episode]['new_w_value'])
    eps_w_value = np.squeeze(memory_traces[episode]['w_value'])
    eps_w_beta = np.squeeze(memory_traces[episode]['w_beta'])
    eps_w_key = np.squeeze(memory_traces[episode]['w_key'])
    Fw = []
    Fw_norm = []
    Fw_step = np.zeros((args['k_size'], args['v_size']))
    for step in range(episode_length):
        ks, vs, bs, new_vs = eps_w_key[step], eps_w_value[step], eps_w_beta[step], eps_new_w_value[step]
        # apply delta rule
        vs_old = np.einsum('ij, j->i', Fw_step, ks)
        new_v = bs * new_vs
        new_v_k = np.einsum("i,j->ij", new_v, ks)
        Fw_step = Fw_step + new_v_k
        # scale F down if norm is > 1
        norm = np.linalg.norm(Fw_step)
        norm = np.maximum(norm - 2, 0) + 1
        Fw_norm.append(norm)
        Fw_step = Fw_step / norm

        Fw.append(np.copy(Fw_step))

    reaching_time = 1
    step = goal_reaching[reaching_time] + 1
    cumu_norm = [np.prod(Fw_norm[x:step]) for x in range(step)]
    contents = eps_w_beta[:step]/np.array(cumu_norm)
    contents = Fw_norm
    fig, axs = plot_1D(contents, (8, 5))
    axs.vlines(goal_reaching, 0, np.max(contents), linewidth=1, color='red', linestyle='dashed', zorder=1)
    #axs.set_xlim([0, step])
    axs.set_title('%s, F_norm, S%se%s' % (env, simu, episode), fontsize=14)

    # specify memory items to visualize
    m_items = ['w_beta']
    for item in m_items:
        contents = np.squeeze(memory_traces[episode][item])
        # plot keys and values, lstm output
        if 'beta' in item:
            # strength
            fig, axs = plot_1D(contents, (8, 5))
            axs.vlines(goal_reaching, 0, np.max(contents), linewidth=1, color='red', linestyle='dashed', zorder=1)
            #axs.set_xlim([0, 200])
        else:
            # keys, values, etc.
            fig, axs = plt.subplots(1, 1, figsize=(8, 5))
            plot_2D(contents, axs=axs, fig=fig)
            axs.vlines(goal_reaching, contents.shape[1], contents.shape[1] + 15, linewidth=1, color='red',
                       linestyle='dashed')
            axs.set_xlim([0, goal_change_freq])
        axs.set_title('%s, %s, S%se%s' % (env, item, simu, episode), fontsize=12)

#### For Fig. 5b ~ 5i ####
## correlate the mem readouts with actions, state or goal
draw2 = False
if draw2:
    stage = args['training_episodes']  # 500, 1000, 1500, args['training_episodes']
    env = args['env']
    simu = 1
    r_value_coloring = 'action'  # action | goal
    # load the data
    data = pickle.load(open(save_dir + '/data/test/ppo_%s_%s_%s_%s.pkl' % (env, network, simu, stage), 'rb'))
    goals = data['goal_location']
    memory_traces = data['memory_traces']
    step_reward = np.squeeze(data['step_reward'])
    trajs = data['trajectories']
    if action_space == 'allocentric': actions = data['actions']

    ### we analyze the stdiance reduction of the memory reading after knowing action ###
    r_value, nor_r_value, r_key = [], [], []
    reach_action, pre_action, pre_w_value = [], [], []
    rnn_out, rnn_cell = [], []
    step_ori, step_node = [], []
    goal_state, step_goal_state = [], []  # the goal location for each step
    goal_w_value, goal_next_w_value = [], []  # those w values one step after being on the goal
    goal_w_key, goal_next_w_key = [], []
    # we store what is actually retrieved at a certain step after goal-reached
    retrieved, retrieved1 = [], []
    # we also store the writing values pre the goal is reached
    all_w_value, if_goal_reaching = [], []
    dot_r_goal_w_key, dot_r_next_goal_w_key = [], []
    for eps in range(len(trajs)):
        goal_reaching = np.where(step_reward[eps] > 0)[0]
        eps_r_beta = np.squeeze(memory_traces[eps]['r_beta'])
        eps_r_value = np.squeeze(memory_traces[eps]['r_value'])
        eps_new_w_value = np.squeeze(memory_traces[eps]['new_w_value'])
        eps_w_value = np.squeeze(memory_traces[eps]['w_value'])
        eps_w_beta = np.squeeze(memory_traces[eps]['w_beta'])
        eps_r_key = np.squeeze(memory_traces[eps]['r_key'])
        eps_w_key = np.squeeze(memory_traces[eps]['w_key'])
        eps_rnn_out = np.squeeze(memory_traces[eps]['rnn_out'])
        eps_rnn_cell = np.squeeze(memory_traces[eps]['rnn_cell'])
        rw_dim = eps_r_value.shape[1]
        # also extract the trajectory
        eps_traj = []
        for tr in trajs[eps][0]:
            eps_traj.extend(tr)
        eps_traj = np.array(eps_traj)

        ## for each subepisode with a fixed goal
        for i in range(len(sub_eps_bounds) - 1):
            # extract the goal reaching for the subepisode
            x = goal_reaching > sub_eps_bounds[i]
            y = goal_reaching < sub_eps_bounds[i + 1]
            sub_reaching = goal_reaching[np.where(x * y)[0]]
            # if a goal is reached more than 5 times, the agent remembers it
            if len(sub_reaching) > 5:
                # we remove the time step when the agent is actually on the goal because it is not relevant much
                indice = []
                for j in range(0, len(sub_reaching) - 1):
                    indice.extend(list(range(sub_reaching[j] + 2, sub_reaching[j + 1] + 1)))

                r_value.extend(eps_r_value[indice])
                r_key.extend(eps_r_key[indice])
                # we record the goal state, and the w_value one step after the goal reached for
                # the first time
                s = sub_reaching[0] + 1
                goal_w_value.append(eps_new_w_value[s])
                goal_next_w_value.append(eps_new_w_value[s + 1])
                # the writing key when the agent is on the goal
                goal_w_key.append(eps_w_key[s])
                goal_next_w_key.append(eps_w_key[s + 1])
                goal_state.append(goals[eps][sub_reaching[0]][0])

                pre_range = range(sub_eps_bounds[i], sub_reaching[0] - 1)
                all_w_value.extend(eps_new_w_value)
                pre_w_value.extend(eps_new_w_value[pre_range])
                # a vector where if the agent is on the goal or not at each step
                on_goal = np.zeros(len(step_reward[eps]))
                temp = goal_reaching + 1
                temp[-1] = min(temp[-1], len(step_reward[eps]) - 1)
                temp[-2] = min(temp[-2], len(step_reward[eps]) - 1)
                on_goal[temp] = 1
                if_goal_reaching.extend(on_goal)
                # during the back-to-goal period

                for step in indice:
                    step_ori.append(int(eps_traj[step][1]) // 90 % 4)
                    step_node.append(int(eps_traj[step][0]))
                    step_goal_state.append(goals[eps][sub_reaching[0]][0])
                    if action_space == 'egocentric':
                        # extract the action that the agent takes
                        mov = eps_traj[step + 1] - eps_traj[step]
                        if (mov[0] != 0 and mov[1] != 0) or (mov[0] == 0 and np.abs(mov[1]) == 180):
                            # the agent must reach the goal and get reset at step+1
                            mov = goal_state[-1] - eps_traj[step]
                            if mov[0] != 0 and mov[1] == 0:
                                # a forward action, represented by 0
                                reach_action.append(0)
                            elif mov[0] == 0 and mov[1] != 0:
                                reach_action.append((mov[1] // 90) % 4)  # either 1 or 3
                        elif mov[0] == 0 and mov[1] == 0:
                            # the agent bumps to the wall, must have moved forward
                            reach_action.append(0)
                        else:
                            if mov[0] != 0: reach_action.append(0)
                            if mov[1] != 0: reach_action.append((mov[1] // 90) % 4)

                # extract the actions in this period
                if action_space == 'allocentric':
                    reach_action.extend(list(np.squeeze(actions[eps])[indice]))


    def plot_embedding(data, dim, coloring=[], embedding=None, method='pca'):
        components = None
        if embedding is None:
            if method == 'umap':
                reducer = umap.UMAP(n_components=dim)
                embedding = reducer.fit_transform(data)
            elif method == 'pca':
                pca = PCA(n_components=dim).fit(data)
                components = pca.components_
                embedding = pca.transform(data)
            elif method == 'lda':
                reducer = LinearDiscriminantAnalysis(n_components=dim).fit(data, coloring)
                embedding = reducer.transform(data)
        if dim == 3:
            fig = plt.figure()
            axs = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(axs)
            sc = axs.scatter3D(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=coloring)
            axs.set_zticks([])
        elif dim == 2:
            fig, axs = plt.subplots(figsize=(7, 7))
            sc = axs.scatter(embedding[:, 0], embedding[:, 1], c=coloring)
        axs.set_xticks([])
        axs.set_yticks([])
        return components, embedding, axs, sc


    reach_action = np.array(reach_action, dtype=np.int32)
    dot_r_goal_w_key = np.array(dot_r_goal_w_key)
    dot_r_next_goal_w_key = np.array(dot_r_next_goal_w_key)
    r_key = np.array(r_key)
    r_value = np.array(r_value)
    step_node = np.array(step_node)
    step_ori = np.array(step_ori)
    node_x = np.array([x // maze_size for x in step_node])
    node_y = np.array([x % maze_size for x in step_node])
    reach_action = np.array(reach_action)

    goal_w_value = np.array(goal_w_value)
    goal_next_w_value = np.array(goal_next_w_value)
    goal_w_key = np.array(goal_w_key)
    goal_next_w_key = np.array(goal_next_w_key)
    if_goal_reaching = np.array(if_goal_reaching)
    goal_ori = np.array([int(x[1]) // 90 % 4 for x in goal_state])
    goal_node = [int(x[0]) for x in goal_state]
    goal_node_x = np.array([x // maze_size for x in goal_node])
    goal_node_y = np.array([x % maze_size for x in goal_node])
    step_goal_ori = np.array([int(x[1]) // 90 % 4 for x in step_goal_state])
    step_goal_node = np.array([int(x[0]) for x in step_goal_state])
    step_goal_node_x = np.array([x // maze_size for x in step_goal_node])
    step_goal_node_y = np.array([x % maze_size for x in step_goal_node])

    orientations = {'east': 0, 'north': 1, 'west': 2, 'south': 3}
    pca_components = {}
    N = min(5000, len(r_value))  # sometimes we only visualize N data points

    # goal w key and value
    node_coloring = np.array([colores_nodos[node] for node in goal_node])

    components, embedding, axs, sc = plot_embedding(goal_w_key, 3, node_coloring)
    axs.set_title('2D pca of goal w keys, colored by goal node coordinate')

    components, embedding, axs, sc = plot_embedding(goal_w_value, 3, node_coloring)
    axs.set_title('2D pca of goal w values, colored by goal node coordinate')

    # dim reduction for the r_value
    method = 'pca'
    if r_value_coloring == 'action':
        possible_actions = list(np.unique(reach_action[:N]))
        if action_space == 'egocentric':
            action_labels = ['forward', 'left turn', 'right turn']
        elif action_space == 'allocentric':
            action_labels = ['E', 'N', 'W', 'S', 'ER', 'NR', 'WR', 'SR']
        components, embedding, axs, sc = plot_embedding(r_value[:N], 2, reach_action[:N], method='pca')
        handles = sc.legend_elements(num=possible_actions)[0]  # extract the handles from the existing scatter plot
        axs.legend(title='Action', handles=handles, labels=action_labels, bbox_to_anchor=(1, 1))
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.title('pca of reading outputs, colored by actions')

        if action_space == 'egocentric':
            # compute row and column difference between goal and agent
            rel_node_x = step_goal_node_x - node_x
            rel_node_y = step_goal_node_y - node_y
            # for each step, determine whether the goal is in the front or the back of the agent
            # and whether the goal is in the right or left of the agent
            node_front, node_right = [], []
            for xx, yy, ori in zip(rel_node_x, rel_node_y, step_ori):
                # positive X difference + face north or negative X difference + face south
                # means goal in front
                if ori == 0:  # east
                    node_front.append(xx)
                    node_right.append(-yy)
                elif ori == 1:  # north
                    node_front.append(yy)
                    node_right.append(xx)
                # positive col difference + face east or negative row difference + face west
                # means goal in front
                elif ori == 2:  # west
                    node_front.append(-xx)
                    node_right.append(yy)
                elif ori == 3:  # south
                    node_front.append(-yy)
                    node_right.append(-xx)
            node_front = np.array(node_front)
            node_right = np.array(node_right)
            components, _, axs, sc = plot_embedding(r_value[:N], 2, node_front[:N], embedding=embedding)
            handles = sc.legend_elements(num=list(range(-4, 5)))[
                0]  # extract the handles from the existing scatter plot
            axs.legend(title='Dis to agent\nfront', handles=handles, labels=list(range(-4, 5)),
                       bbox_to_anchor=(1, 1))
            plt.title('2D pca of reading outputs, colored by distance to the front')
            components, _, axs, sc = plot_embedding(r_value[:N], 2, node_right[:N], embedding=embedding)
            handles = sc.legend_elements(num=list(range(-4, 5)))[
                0]  # extract the handles from the existing scatter plot
            axs.legend(title='Dis to agent\nright', handles=handles, labels=list(range(-4, 5)),
                       bbox_to_anchor=(1, 1))
            plt.title('2D pca of reading outputs, colored by distance to the right')
    elif r_value_coloring == 'goal':
        components, _, axs, sc = plot_embedding(r_value[:N], 3, step_goal_ori[:N], method=method)
        handles = sc.legend_elements(num=[0, 1, 2, 3])[0]  # extract the handles from the existing scatter plot
        axs.legend(title='Orientation', handles=handles, labels=['north', 'east', 'south', 'west'],
                   loc="upper right")
        plt.title('2D %s of reading outputs, colored by goal ori' % method)

        step_goal_node_coloring = np.array([colores_nodos[node] for node in step_goal_node])
        components, _, axs, sc = plot_embedding(r_value[:N], 3, step_goal_node_coloring[:N], method=method)
        plt.title('2D %s of r outputs , colored by goal node' % (method))

        #
        goal_node_coloring = np.array([colores_nodos[node] for node in goal_node])
        # plot a 2D pca for goal w keys
        _, _, axs, sc = plot_embedding(goal_w_key, 3, goal_node_coloring)
        axs.set_title('pca of goal w keys, colored by goal node')
        axs.grid(True)
        # plot a 2D pca for goal w keys
        _, _, axs, sc = plot_embedding(goal_w_value, 3, goal_node_coloring)
        axs.set_title('pca of next goal w keys, colored by goal node')
        axs.grid(True)

    # ##### pca for reading keys #####
    if action_space == 'egocentric':
        # dim reduction for reading keys, colored by current orientation
        components, embedding, axs, sc = plot_embedding(r_key[:N], 3, step_ori[:N], method='pca')  # plus 5 to move to other colors
        handles = sc.legend_elements(num=[0, 1, 2, 3])[0]
        #axs.legend(title='HD', handles=handles, labels=['E', 'N', 'W', 'S'], loc="upper right")

        # extract all reading keys corresponding to north and south
        idxn = np.where(step_ori[:N] == orientations['north'])[0]
        idxs = np.where(step_ori[:N] == orientations['south'])[0]
        idx = np.concatenate((idxn, idxs))
        components, embedding, axs, sc = plot_embedding(r_key[:N][idx], 2, node_y[:N][idx])
        handles = sc.legend_elements(num=[i for i in range(maze_size)])[0]
        axs.legend(title='Coordinate Y', handles=handles,
                   labels=[i for i in range(maze_size)],
                   bbox_to_anchor=(1, 1))
        axs.set_title('2D pca of r keys with facing north and south, colored by Y coordinate')
        # we project the goal w keys in the same pca space as in north and south
        # embedding_1 = np.matmul(goal_w_key, components.T)
        # embedding = np.concatenate((embedding, embedding_1))
        # values = np.concatenate((goal_w_key, r_key[:N][idx]))
        # cmap = mat.colormaps['viridis']
        # colors = cmap(np.linspace(0, 1, maze_size))
        # goal_y_colors = [colors[x] for x in goal_node_y]
        # r_key_colors = ['gray'] * len(idx)
        # components, embedding, axs, sc = plot_embedding(values, 2, r_key_colors + goal_y_colors, embedding=embedding)
        # handles = sc.legend_elements(num=['gray'])[0]
        # axs.legend(title='Goal\nCoordinate Y', handles=handles, labels=['r key'], bbox_to_anchor=(1, 1))
        # axs.set_title('r keys with facing N and S, together with goal w keys')

        ## take out the r key north and south, while node y is fixed
        idxe = np.where((step_ori[:N] == orientations['east']))[0]
        idxw = np.where((step_ori[:N] == orientations['west']))[0]
        idx = np.concatenate((idxe, idxw))
        components, embedding, axs, sc = plot_embedding(r_key[:N][idx], 2, node_x[:N][idx])
        handles = sc.legend_elements(num=[i for i in range(maze_size)])[0]
        axs.legend(title='Coordinate X', handles=handles, labels=[i for i in range(maze_size)],
                   bbox_to_anchor=(1, 1))
        axs.set_title('2D pca of r keys with facing east and west, colored by X coordinate')

        # 2D pca for only one direction of r keys, color by current node
        # node_coloring = np.array([colores_nodos[node] for node in step_goal_node])
        # components, embedding, axs, sc = plot_embedding(r_key[:N][idxn], 3, node_coloring[:N][idxn])
        # axs.set_title('2D pca of r keys with facing north, colored by node coordinate')

    elif action_space == 'allocentric':
        components, embedding, axs, sc = plot_embedding(r_key[:N], 2, node_y[:N])
        handles = sc.legend_elements(num=[i for i in range(maze_size)])[0]
        axs.legend(title='Coordinate Y', handles=handles, labels=[i for i in range(maze_size)],
                   bbox_to_anchor=(1, 1))
        axs.set_title('2D pca of r keys colored by Y coordinate')

        components, embedding, axs, sc = plot_embedding(r_key[:N], 2, node_x[:N])
        handles = sc.legend_elements(num=[i for i in range(maze_size)])[0]
        axs.legend(title='Coordinate X', handles=handles, labels=[i for i in range(maze_size)],
                   bbox_to_anchor=(1, 1))
        axs.set_title('2D pca of r keys colored by X coordinate')


#### For supplementary Fig.7b, 7c, 7f ####
# working out the geometric theory part I
draw3_1 = False
if draw3_1:
    stage = args['training_episodes']  # 500, 1000, 1500, args['training_episodes']
    env = args['env']
    simu = 2
    # load the data
    data = pickle.load(open(save_dir + '/data/test/ppo_%s_%s_%s_%s.pkl' % (env, network, simu, stage), 'rb'))
    goals = data['goal_location']
    memory_traces = data['memory_traces']
    step_reward = np.squeeze(data['step_reward'])
    trajs = data['trajectories']
    try:
        F_norm = data['F_norm']
        with_f_norm = True
    except KeyError:
        with_f_norm = False
        F_norm = []

    ### we analyze the stdiance reduction of the memory reading after knowing action ###
    r_value, r_key = [], []
    w_value = []
    reach_action = []
    step_ori, step_node = [], []
    goal_reach_index = []  # record the # of steps after the goal is reached
    goal_state, step_goal_state = [], []  # the goal location for each step
    goal_step = 2
    goal_w_value = [[] for _ in range(goal_step)]  # those w values N step after being on the goal
    goal_w_key = [[] for _ in range(goal_step)]
    goal_action = [[] for _ in range(goal_step)]
    # we store what is actually retrieved at a certain step after goal-reached
    retrieved, retrieved1 = [], []
    dot_r_goal_w_key, dot_r_next_goal_w_key = [], []
    for eps in range(len(trajs)):
        goal_reaching = np.where(step_reward[eps] > 0)[0]
        eps_r_value = np.squeeze(memory_traces[eps]['r_value'])
        eps_new_w_value = np.squeeze(memory_traces[eps]['new_w_value'])
        eps_w_value = np.squeeze(memory_traces[eps]['w_value'])
        eps_w_beta = np.squeeze(memory_traces[eps]['w_beta'])
        eps_r_key = np.squeeze(memory_traces[eps]['r_key'])
        eps_w_key = np.squeeze(memory_traces[eps]['w_key'])
        eps_r_beta = np.squeeze(memory_traces[eps]['r_beta'])
        rw_dim = eps_r_value.shape[1]
        # also extract the trajectory
        eps_traj = []
        for tr in trajs[eps][0]:
            eps_traj.extend(tr)
        eps_traj = np.array(eps_traj)
        if not with_f_norm:
            # we revover the Fast weight matrix for each episode
            Fw = []
            Fw_norm = []
            Fw_step = np.zeros((args['k_size'], args['v_size']))
            for step in range(episode_length):
                ks, vs, bs, new_vs = eps_w_key[step], eps_w_value[step], eps_w_beta[step], eps_new_w_value[step]
                # apply delta rule
                vs_old = np.einsum('ij, j->i', Fw_step, ks)
                new_v = bs * new_vs
                new_v_k = np.einsum("i,j->ij", new_v, ks)
                Fw_step = Fw_step + new_v_k
                # scale F down if norm is > 1
                norm = np.linalg.norm(Fw_step)
                norm = np.maximum(norm - 1, 0) + 1
                Fw_step = Fw_step / norm
                Fw_norm.append(norm)
                Fw.append(np.copy(Fw_step))
            F_norm.append(Fw_norm)
        elif with_f_norm:
            Fw_norm = F_norm[eps]

        Fw_norm = np.array(Fw_norm)
        ## for each subepisode with a fixed goal
        for i in range(len(sub_eps_bounds) - 1):
            # extract the goal reaching for the subepisode
            x = goal_reaching > sub_eps_bounds[i]
            y = goal_reaching < sub_eps_bounds[i + 1]
            sub_reaching = goal_reaching[np.where(x * y)[0]]
            # if a goal is reached more than 5 times, the agent remembers it
            if len(sub_reaching) > 2:
                # we remove the time step when the agent is actually on the goal because it is not relevant much
                indice = []
                for j in range(0, len(sub_reaching)-1):
                    indice.extend(list(range(sub_reaching[j] + 2, sub_reaching[j + 1] + 1)))
                if len(indice) < goal_step: continue
                r_value.extend(eps_r_value[indice])
                r_key.extend(eps_r_key[indice]*np.tile([eps_r_beta[indice]], (args['k_size'], 1)).T)
                w_value.extend(eps_new_w_value[indice])
                goal_reach_index.extend([xx+1 for xx in range(len(indice))])
                # we record the goal state,
                goal_state.append(goals[eps][sub_reaching[0]][0])
                # and the w_value several steps after the goal reached for the first time
                s = sub_reaching[0] + 1
                ######## based on the memory writing and reading rule ########
                # write-in contents at step s
                weights = np.matmul(eps_r_key[indice], np.array([eps_w_key[s]]).T)
                # scaling factor from normalization
                temp = eps_new_w_value[s] * eps_w_beta[s]  / Fw_norm[s + 1]
                scale = [np.prod(Fw_norm[indice[0]:idx]) for idx in indice]
                result = np.matmul(weights, np.array([temp])) / np.tile([scale], (args['v_size'], 1)).T

                # write-in contents at step s+1
                weights1 = np.matmul(eps_r_key[indice], np.array([eps_w_key[s + 1]]).T)
                temp = eps_new_w_value[s + 1] * eps_w_beta[s + 1] / Fw_norm[s + 1]
                result1 = np.matmul(weights1, np.array([temp])) / np.tile([scale], (args['v_size'], 1)).T

                # we also take into account the FWM before reaching the goal for the first time
                base = np.outer(eps_new_w_value[s], eps_w_key[s]) * eps_w_beta[s]/ Fw_norm[s]/ Fw_norm[s+1]
                base1 = np.outer(eps_new_w_value[s+1], eps_w_key[s+1]) * eps_w_beta[s+1] / Fw_norm[s+1]
                Fw_norm[s + 1] = 1

                ddd = 0
                for x in indice:
                    temp = (base) / np.prod(Fw_norm[(s + 1):x + 1])
                    temp1 = (base+base1) / np.prod(Fw_norm[(s + 1):x + 1])
                    if x == s+1:
                        pass
                    else:
                        ddd = (ddd + np.outer(eps_new_w_value[x], eps_w_key[x]) * eps_w_beta[x]) / Fw_norm[x]
                        temp += ddd
                        temp1 += ddd
                    retrieved.append(np.matmul(temp, eps_r_key[x]))
                    retrieved1.append(np.matmul(temp1, eps_r_key[x]))
                    dot_r_goal_w_key.append(np.dot(eps_r_key[x], eps_w_key[s]) * eps_w_beta[s]/ Fw_norm[s]/ Fw_norm[s+1]
                                            - np.dot(eps_r_key[x], eps_w_key[s+1]) * eps_w_beta[s+1] / Fw_norm[s+1])
                    dot_r_next_goal_w_key.append(np.dot(eps_r_key[x], eps_w_key[x]) * eps_w_beta[x] / Fw_norm[x])
                    # extract actions
                    step_ori.append(int(eps_traj[x][1]) // 90 % 4)
                    step_node.append(int(eps_traj[x][0]))
                    step_goal_state.append(goals[eps][sub_reaching[0]][0])

                    # extract the action that the agent takes
                    mov = eps_traj[x + 1] - eps_traj[x]
                    if (mov[0] != 0 and mov[1] != 0) or (mov[0] == 0 and np.abs(mov[1]) == 180):
                        # the agent must reach the goal and get reset at step+1
                        mov = goal_state[-1] - eps_traj[x]
                        if mov[0] != 0 and mov[1] == 0:
                            # a forward action, represented by 0
                            action = 0
                        elif mov[0] == 0 and mov[1] != 0:
                            action = (mov[1] // 90) % 4  # either 1 or 3
                    elif mov[0] == 0 and mov[1] == 0:
                        # the agent bumps to the wall, must have moved forward
                        action = 0
                    else:
                        if mov[0] != 0: action = 0
                        if mov[1] != 0: action = (mov[1] // 90) % 4
                    reach_action.append(action)
                    for ii in range(goal_step):
                        if x == s+ii:
                            goal_action[ii].append(action)

                for ii in range(goal_step):
                    goal_w_value[ii].append(eps_new_w_value[s+ii]*eps_w_beta[s+ii]/Fw_norm[s+ii])
                    goal_w_key[ii].append(eps_w_key[s+ii])


    if not with_f_norm:
        data['F_norm'] = F_norm
        pickle.dump(data, open(save_dir + '/data/test/ppo_%s_%s_%s_%s.pkl' % (env, network, simu, stage), 'wb'))

    def plot_embedding(data, dim, coloring=[], embedding=None, method='pca'):
        reducer = None
        if embedding is None:
            if method == 'umap':
                reducer = umap.UMAP(n_components=dim)
                embedding = reducer.fit_transform(data)
            elif method == 'pca':
                reducer = PCA(n_components=dim).fit(data)
                components = reducer.components_
                embedding = reducer.transform(data)
            elif method == 'lda':
                reducer = LinearDiscriminantAnalysis(n_components=dim).fit(data, coloring)
                embedding = reducer.transform(data)
        if dim == 3:
            fig = plt.figure()
            axs = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(axs)
            sc = axs.scatter3D(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=coloring)
            #axs.set_zticks([])
        elif dim == 2:
            fig, axs = plt.subplots(figsize=(7, 7))
            sc = axs.scatter(embedding[:, 0], embedding[:, 1], c=coloring)
        #axs.set_xticks([])
        #axs.set_yticks([])
        return reducer, embedding, axs, sc

    reach_action = np.array(reach_action, dtype=np.int32)
    r_key = np.array(r_key)
    r_value = np.array(r_value)
    step_node = np.array(step_node)
    step_ori = np.array(step_ori)
    node_x = np.array([x // maze_size for x in step_node])
    node_y = np.array([x % maze_size for x in step_node])
    dot_r_goal_w_key = np.array(dot_r_goal_w_key)
    dot_r_next_goal_w_key = np.array(dot_r_next_goal_w_key)

    goal_w_value = np.array(goal_w_value)
    goal_w_key = np.array(goal_w_key)

    goal_reach_index = np.array(goal_reach_index)
    goal_ori = np.array([int(x[1]) // 90 % 4 for x in goal_state])
    goal_node = [int(x[0]) for x in goal_state]
    goal_node_x = np.array([x // maze_size for x in goal_node])
    goal_node_y = np.array([x % maze_size for x in goal_node])
    step_goal_ori = np.array([int(x[1]) // 90 % 4 for x in step_goal_state])
    step_goal_node = np.array([int(x[0]) for x in step_goal_state])
    step_goal_node_x = np.array([x // maze_size for x in step_goal_node])
    step_goal_node_y = np.array([x % maze_size for x in step_goal_node])

    # compute row and column difference between goal and agent
    rel_node_x = step_goal_node_x - node_x
    rel_node_y = step_goal_node_y - node_y
    # for each step, determine whether the goal is in the front or the back of the agent
    # and whether the goal is in the right or left of the agent
    node_front, node_right = [], []
    for xx, yy, ori in zip(rel_node_x, rel_node_y, step_ori):
        # positive X difference + face north or negative X difference + face south
        # means goal in front
        if ori == 0:  # east
            node_front.append(xx)
            node_right.append(-yy)
        elif ori == 1:  # north
            node_front.append(yy)
            node_right.append(xx)
        # positive col difference + face east or negative row difference + face west
        # means goal in front
        elif ori == 2:  # west
            node_front.append(-xx)
            node_right.append(yy)
        elif ori == 3:  # south
            node_front.append(-yy)
            node_right.append(-xx)
    node_front = np.array(node_front)
    node_right = np.array(node_right)

    orientations = {'east': 0, 'north': 1, 'west': 2, 'south': 3}
    N = min(5000, len(r_value))  # sometimes we only visualize N data points

    # dim reduction for the r_value
    dim = 2
    method = 'lda'
    reducer, r_value_embedding, axs, sc = plot_embedding(r_value[:N], dim, reach_action[:N], method=method)
    handles = sc.legend_elements(num=[0, 1, 3])[0]  # extract the handles from the existing scatter plot
    axs.legend(title='Action', handles=handles, labels=['forward', 'left turn', 'right turn'], bbox_to_anchor=(1, 1))
    plt.title('pca of reading outputs, colored by actions')

    _, _, axs, sc = plot_embedding(r_value[:N], dim, node_right[:N], method=method, embedding=r_value_embedding)
    handles = sc.legend_elements(num=list(np.arange(-maze_size+1, maze_size)))[0]  # extract the handles from the existing scatter plot
    axs.legend(title='Dis to\n agent right', handles=handles, labels=list(np.arange(-4, 5)), bbox_to_anchor=(1, 1))

    _, _, axs, sc = plot_embedding(r_value[:N], dim, node_front[:N], method=method, embedding=r_value_embedding)
    handles = sc.legend_elements(num=list(np.arange(-maze_size+1, maze_size)))[0]  # extract the handles from the existing scatter plot
    axs.legend(title='Dis to\n agent front', handles=handles, labels=list(np.arange(-4, 5)), bbox_to_anchor=(1, 1))

    # we visualize the w values at step g and afterwards
    value_embedding, identities = [], []
    id_colors = ['green', 'lightgreen']

    value_embedding.extend(reducer.transform(goal_w_value[0]))
    identities.extend([id_colors[0]]*len(goal_w_value[0]))
    value_embedding.extend(reducer.transform(goal_w_value[1]))
    identities.extend([id_colors[1]] * len(goal_w_value[1]))

    value_embedding = np.array(value_embedding)
    identities = np.array(identities)
    # add also the r_values
    # value_embedding = np.concatenate((value_embedding, r_value_embedding))
    # identities = np.concatenate((identities, np.zeros(len(r_value_embedding))))
    _, _, axs, sc = plot_embedding(goal_w_value, dim, identities, embedding=value_embedding, method=method)
    handles = sc.legend_elements(num=id_colors)[0]  # extract the handles from the existing scatter plot
    axs.legend(title='Identity', handles=handles, labels=['for step g', 'for step g+1'], bbox_to_anchor=(1, 1), fontsize=28)

    idx1 = np.where(goal_reach_index == 2)[0]
    _, _, axs, sc = plot_embedding(goal_w_value, dim, node_right[idx1], embedding=reducer.transform(goal_w_value[0]), method=method)
    handles = sc.legend_elements(num=list(np.arange(-maze_size+1, maze_size)))[0]  # extract the handles from the existing scatter plot
    axs.legend(title='Goal node\n right', handles=handles, labels=list(np.arange(-4, 5)), bbox_to_anchor=(1, 1))
    _, _, axs, sc = plot_embedding(goal_w_value, dim, node_right[idx1],
                                   embedding=reducer.transform(goal_w_value[1]), method=method)
    handles = sc.legend_elements(num=list(np.arange(-maze_size+1, maze_size)))[0]  # extract the handles from the existing scatter plot
    axs.legend(title='Goal node\n right', handles=handles, labels=list(np.arange(-4, 5)), bbox_to_anchor=(1, 1))
    _, _, axs, sc = plot_embedding(goal_w_value, dim, node_right[idx1], embedding=reducer.transform(0.5*goal_w_value[0]+goal_w_value[1]), method=method)
    handles = sc.legend_elements(num=list(np.arange(-maze_size+1, maze_size)))[0]  # extract the handles from the existing scatter plot
    axs.legend(title='Goal node\n right', handles=handles, labels=list(np.arange(-4, 5)), bbox_to_anchor=(1, 1))

    # 2d pca of retrieving results based on the writing two steps after goal reached
    rr_actions = np.copy(reach_action[:N])
    embedding = reducer.transform(retrieved1[:N])
    components, _, axs, sc = plot_embedding(retrieved1[:N], dim, rr_actions, embedding=embedding, method=method)
    handles = sc.legend_elements(num=[0, 1, 3])[0]
    axs.legend(handles=handles, labels=['forward', 'left turn', 'right turn'], loc="upper right", fontsize=36)
    axs.set_title('2D pca of two step retrieved, colored by actions')
    # # take those idx where the values in the last retrieved are negative
    idx1 = np.where((dot_r_goal_w_key > 0)[:N])[0]
    # take those idx where the values in this retrieved are positive
    idx2 = np.where((dot_r_goal_w_key <= 0)[:N])[0]
    # take those idx where the values in this retrieved are negative
    colors = np.zeros((len(embedding), 4))
    cmap = mat.colormaps['viridis']
    cmap = cmap(np.linspace(0, 1, maze_size))
    colors[idx1] = cmap[0]
    colors[idx2] = cmap[3]
    _, _, axs, sc = plot_embedding(retrieved1[:N], dim, colors, embedding = embedding, method=method)
    handles = sc.legend_elements(num=[cmap[0], cmap[3]])[0]
    axs.legend( handles=handles, fontsize=36,
                labels=['P > 0', 'P \u2264 0'],
                loc="best")

    # extract all indice corresponding to north and south
    method = 'pca'
    idxn = np.where(step_ori[:N] == orientations['north'])[0]
    idxs = np.where(step_ori[:N] == orientations['south'])[0]
    idx = np.concatenate((idxn, idxs))
    # we project the goal w keys in the same pca space as in north and south
    for jj in range(2):
        embedding_w_key = reducer.transform(goal_w_key[jj])
        embedding = np.concatenate((embedding_r_key, embedding_w_key))
        values = np.concatenate((goal_w_key[jj], r_key[:N][idx]))
        cmap = mat.colormaps['viridis']
        colors = cmap(np.linspace(0, 1, maze_size))
        goal_y_colors = [colors[x] for x in goal_node_y]
        r_key_colors = ['gray'] * len(idx)
        _, embedding, axs, sc = plot_embedding(values, 2, r_key_colors + goal_y_colors, embedding=embedding, method=method)
        handles = sc.legend_elements(num=['gray'])[0]
        axs.legend(title='Goal\nCoordinate Y', handles=handles, labels=['r key'], bbox_to_anchor=(1, 1))
        axs.set_title('r keys with facing N and S, together with g+%s w keys'%jj)

        # now we express the entire r and w keys in the coordinate of its pca components + residuels
        r_key_pca = np.matmul(embedding_r_key, components)
        r_key_residuel = r_key[:N][idx] - r_key_pca
        w_key_pca = np.matmul(embedding_w_key, components)
        w_key_residuel = goal_w_key[jj] - w_key_pca

        r_keys = [r_key[:N][idx], r_key_pca, r_key_residuel]
        w_keys = [goal_w_key[jj], w_key_pca, w_key_residuel]
        colors = ['b', 'orange', 'k']
        legends = ['complete', 'pca', 'residuel']

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for l in range(3):
            # we then show that for each orientation, the dot product of r_key
            # and goal_w_key reflects their node difference.
            # compute an average goal_w_key vector for each goal node
            avg_goal_w_key = np.zeros((maze_size, maze_size, args['k_size']))
            for x in range(maze_size):
                for y in range(maze_size):
                    idx_ = np.where((goal_node_x == x) * (goal_node_y == y))
                    avg_goal_w_key[x, y] = np.mean(w_keys[l][idx_], axis=0)
            col = 0
            for ori in [1, 3]:
                # for each orientation, compute an average r_key for each node
                avg_r_key = np.zeros((maze_size, maze_size, args['k_size']))
                for x in range(maze_size):
                    for y in range(maze_size):
                        idx_ = np.where((node_x[:N][idx] == x) * (node_y[:N][idx] == y) * (step_ori[:N][idx] == ori))
                        avg_r_key[x, y] = np.mean(r_keys[l][idx_], axis=0)
                # now we fix the node y to be 2
                current_node_y = 2
                dot = np.matmul(avg_goal_w_key, avg_r_key[:, current_node_y, :].T)  # (goal x, goal y, node x)
                # we average over the x coordinates for goal and node
                mu = dot.mean(axis=(0, 2))
                std = dot.std(axis=(0, 2))

                axs[col].errorbar(range(maze_size), mu, std, marker='^', color=colors[l], label=legends[l])
                axs[col].set_xlabel('Y coordinate of goal node')
                if col == 0:
                    axs[col].set_ylabel('value of dot product')
                axs[col].set_title(' HD %s' % ori)
                #if col == 1:  axs[col].legend(bbox_to_anchor=(1.05, 1))
                axs[col].grid()
                plt.tight_layout()
                col += 1

### For supplementary Fig. 7g, 7h, 7i, 7j ####
# working out the geometric theory part II
draw3_2 = False
if draw3_2:
    stage = 900 # 500, 1000, 1500, args['training_episodes']
    env = args['env']
    rescale = 1 # adjust the ratio between w_value at g and g+1, as well as these for w_key; just for convinience of explanation
    # load the data
    simu = 2
    data = pickle.load(open(save_dir + '/data/test/ppo_%s_%s_%s_%s.pkl' % (env, network, simu, stage), 'rb'))

    goals = data['goal_location']
    memory_traces = data['memory_traces']
    step_reward = np.squeeze(data['step_reward'])
    trajs = data['trajectories']
    try:
        F_norm = data['F_norm']
        with_f_norm = True
    except KeyError:
        with_f_norm = False
        F_norm = []

    ### we analyze the stdiance reduction of the memory reading after knowing action ###
    r_value, r_key = [], []
    r_key_separate = []
    w_value = []
    reach_action = []
    step_ori, step_node = [], []
    goal_reach_index = []  # record the # of steps after the goal is reached
    goal_state, step_goal_state = [], []  # the goal location for each step
    goal_step = 2
    goal_w_value = [[] for _ in range(goal_step)]  # those w values N step after being on the goal
    goal_w_key = [[] for _ in range(goal_step)]
    goal_action = [[] for _ in range(goal_step)]
    # we store what is actually retrieved at a certain step after goal-reached
    retrieved, retrieved1 = [], []
    dot_r_goal_w_key, dot_r_next_goal_w_key = [], []
    for eps in range(len(trajs)):
        goal_reaching = np.where(step_reward[eps] > 0)[0]
        eps_r_value = np.squeeze(memory_traces[eps]['r_value'])
        eps_new_w_value = np.squeeze(memory_traces[eps]['new_w_value'])
        eps_w_value = np.squeeze(memory_traces[eps]['w_value'])
        eps_w_beta = np.squeeze(memory_traces[eps]['w_beta'])
        eps_r_key = np.squeeze(memory_traces[eps]['r_key'])
        eps_w_key = np.squeeze(memory_traces[eps]['w_key'])
        eps_r_beta = np.squeeze(memory_traces[eps]['r_beta'])
        rw_dim = eps_r_value.shape[1]
        # also extract the trajectory
        eps_traj = []
        for tr in trajs[eps][0]:
            eps_traj.extend(tr)
        eps_traj = np.array(eps_traj)
        if not with_f_norm:
            # we revover the Fast weight matrix for each episode
            Fw = []
            Fw_norm = []
            Fw_step = np.zeros((args['k_size'], args['v_size']))
            for step in range(episode_length):
                ks, vs, bs, new_vs = eps_w_key[step], eps_w_value[step], eps_w_beta[step], eps_new_w_value[step]
                # apply delta rule
                vs_old = np.einsum('ij, j->i', Fw_step, ks)
                new_v = bs * new_vs
                new_v_k = np.einsum("i,j->ij", new_v, ks)
                Fw_step = Fw_step + new_v_k
                # scale F down if norm is > 1
                norm = np.linalg.norm(Fw_step)
                norm = np.maximum(norm - 2, 0) + 1
                Fw_step = Fw_step / norm
                Fw_norm.append(norm)
                Fw.append(np.copy(Fw_step))
            F_norm.append(Fw_norm)
        elif with_f_norm:
            Fw_norm = F_norm[eps]

        Fw_norm = np.array(Fw_norm)
        ## for each subepisode with a fixed goal
        for i in range(len(sub_eps_bounds) - 1):
            # extract the goal reaching for the subepisode
            x = goal_reaching > sub_eps_bounds[i]
            y = goal_reaching < sub_eps_bounds[i + 1]
            sub_reaching = goal_reaching[np.where(x * y)[0]]
            # if a goal is reached more than 5 times, the agent remembers it
            if len(sub_reaching) > 6:
                # we remove the time step when the agent is actually on the goal because it is not relevant much
                indice = []
                for j in range(0, 2 - 1):
                    indice.extend(list(range(sub_reaching[j] + 2, sub_reaching[j + 1] + 1)))
                if len(indice) < goal_step: continue
                r_value.extend(eps_r_value[indice])
                r_key.extend(eps_r_key[indice])
                r_key_separate.append(eps_r_key[indice])
                w_value.extend(eps_new_w_value[indice])
                goal_reach_index.extend([xx + 1 for xx in range(len(indice))])
                # we record the goal state,
                goal_state.append(goals[eps][sub_reaching[0]][0])
                # and the w_value several steps after the goal reached for the first time
                s = sub_reaching[0] + 1
                ######## based on the memory writing and reading rule ########
                # write-in contents at step s
                weights = np.matmul(eps_r_key[indice], np.array([eps_w_key[s]]).T)
                # scaling factor from normalization
                temp = eps_new_w_value[s] * eps_w_beta[s] / Fw_norm[s + 1]
                scale = [np.prod(Fw_norm[indice[0]:idx]) for idx in indice]
                result = np.matmul(weights, np.array([temp])) / np.tile([scale], (args['v_size'], 1)).T

                # write-in contents at step s+1
                weights1 = np.matmul(eps_r_key[indice], np.array([eps_w_key[s + 1]]).T)
                temp = eps_new_w_value[s + 1] * eps_w_beta[s + 1] / Fw_norm[s + 1]
                result1 = np.matmul(weights1, np.array([temp])) / np.tile([scale], (args['v_size'], 1)).T

                # we also take into account the FWM before reaching the goal for the first time
                base = np.outer(eps_new_w_value[s], eps_w_key[s]) * eps_w_beta[s] / Fw_norm[s] / Fw_norm[s + 1]
                base1 = np.outer(eps_new_w_value[s + 1], eps_w_key[s + 1]) * eps_w_beta[s + 1] / Fw_norm[s + 1]
                # Fw_norm[s + 1] = 1

                ddd = 0
                for x in indice:
                    temp = (base) / np.prod(Fw_norm[(s):x + 1])
                    temp1 = (base + base1) / np.prod(Fw_norm[(s + 1):x + 1])
                    if x == s + 1:
                        pass
                    else:
                        ddd = (ddd + np.outer(eps_new_w_value[x], eps_w_key[x]) * eps_w_beta[x]) / Fw_norm[x]
                        # temp += ddd
                        # temp1 += ddd
                    retrieved.append(np.matmul(temp, eps_r_key[x]))
                    retrieved1.append(np.matmul(temp1, eps_r_key[x]))
                    dot_r_goal_w_key.append(np.dot(eps_r_key[x], eps_w_key[s])
                                            - np.dot(eps_r_key[x], eps_w_key[s + 1]))
                    dot_r_next_goal_w_key.append(np.dot(eps_r_key[x], eps_w_key[x]))
                    # extract actions
                    step_ori.append(int(eps_traj[x][1]) // 90 % 4)
                    step_node.append(int(eps_traj[x][0]))
                    step_goal_state.append(goals[eps][sub_reaching[0]][0])

                    # extract the action that the agent takes
                    mov = eps_traj[x + 1] - eps_traj[x]
                    if (mov[0] != 0 and mov[1] != 0) or (mov[0] == 0 and np.abs(mov[1]) == 180):
                        # the agent must reach the goal and get reset at step+1
                        mov = goal_state[-1] - eps_traj[x]
                        if mov[0] != 0 and mov[1] == 0:
                            # a forward action, represented by 0
                            action = 0
                        elif mov[0] == 0 and mov[1] != 0:
                            action = (mov[1] // 90) % 4  # either 1 or 3
                    elif mov[0] == 0 and mov[1] == 0:
                        # the agent bumps to the wall, must have moved forward
                        action = 0
                    else:
                        if mov[0] != 0: action = 0
                        if mov[1] != 0: action = (mov[1] // 90) % 4
                    reach_action.append(action)
                    for ii in range(goal_step):
                        if x == s + ii:
                            goal_action[ii].append(action)

                goal_w_value[0].append(eps_new_w_value[s] * eps_w_beta[s] / Fw_norm[s] / Fw_norm[s + 1])
                goal_w_key[0].append(eps_w_key[s]/1)
                scale = 4
                goal_w_value[1].append(eps_new_w_value[s + 1] * eps_w_beta[s + 1] / Fw_norm[s + 1])
                goal_w_key[1].append(eps_w_key[s + 1] / scale/1)

    # if not with_f_norm:
    #     data['F_norm'] = F_norm
    #     pickle.dump(data, open(path, 'wb'))


    def plot_embedding(data, dim, coloring=[], embedding=None, method='pca'):
        reducer = None
        if embedding is None:
            if method == 'umap':
                reducer = umap.UMAP(n_components=dim)
                embedding = reducer.fit_transform(data)
            elif method == 'pca':
                reducer = PCA(n_components=dim).fit(data)
                components = reducer.components_
                embedding = reducer.transform(data)
            elif method == 'lda':
                reducer = LinearDiscriminantAnalysis(n_components=dim).fit(data, coloring)
                embedding = reducer.transform(data)
        if dim == 3:
            fig = plt.figure()
            axs = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(axs)
            sc = axs.scatter3D(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=coloring)
            # axs.set_zticks([])
        elif dim == 2:
            fig, axs = plt.subplots(figsize=(7, 7))
            sc = axs.scatter(embedding[:, 0], embedding[:, 1], c=coloring)
        # axs.set_xticks([])
        # axs.set_yticks([])
        return reducer, embedding, axs, sc


    reach_action = np.array(reach_action, dtype=np.int32)
    r_key = np.array(r_key)
    r_value = np.array(r_value)
    #r_key_separate = np.array(r_key_separate)
    step_node = np.array(step_node)
    step_ori = np.array(step_ori)
    node_x = np.array([x // maze_size for x in step_node])
    node_y = np.array([x % maze_size for x in step_node])
    dot_r_goal_w_key = np.array(dot_r_goal_w_key)
    dot_r_next_goal_w_key = np.array(dot_r_next_goal_w_key)

    goal_w_value = np.array(goal_w_value)
    goal_w_key = np.array(goal_w_key)

    goal_reach_index = np.array(goal_reach_index)
    goal_node = [int(x[0]) for x in goal_state]
    # goal_node = goal_state
    goal_node_x = np.array([x // maze_size for x in goal_node])
    goal_node_y = np.array([x % maze_size for x in goal_node])
    step_goal_node = np.array([int(x[0]) for x in step_goal_state])
    # step_goal_node = step_goal_state
    step_goal_node_x = np.array([x // maze_size for x in step_goal_node])
    step_goal_node_y = np.array([x % maze_size for x in step_goal_node])

    # compute row and column difference between goal and agent
    rel_node_x = step_goal_node_x - node_x
    rel_node_y = step_goal_node_y - node_y
    # for each step, determine whether the goal is in the front or the back of the agent
    # and whether the goal is in the right or left of the agent
    node_front, node_right = [], []
    for xx, yy, ori in zip(rel_node_x, rel_node_y, step_ori):
        # positive X difference + face north or negative X difference + face south
        # means goal in front
        if ori == 0:  # east
            node_front.append(xx)
            node_right.append(-yy)
        elif ori == 1:  # north
            node_front.append(yy)
            node_right.append(xx)
        # positive col difference + face east or negative row difference + face west
        # means goal in front
        elif ori == 2:  # west
            node_front.append(-xx)
            node_right.append(yy)
        elif ori == 3:  # south
            node_front.append(-yy)
            node_right.append(-xx)
    node_front = np.array(node_front)
    node_right = np.array(node_right)

    orientations = {'east': 0, 'north': 1, 'west': 2, 'south': 3}
    N = min(5000, len(r_value))  # sometimes we only visualize N data points

    # dim reduction for the r_value
    dim = 2
    method = 'lda'
    reducer, r_value_embedding, axs, sc = plot_embedding(r_value[:N], dim, reach_action[:N], method=method)
    handles = sc.legend_elements(num=[0, 1, 3])[0]  # extract the handles from the existing scatter plot
    # axs.legend(title='Action', handles=handles, labels=['forward', 'left turn', 'right turn'], bbox_to_anchor=(1, 1))
    plt.title('pca of reading outputs, colored by actions')

    if method == 'lda':
        components = reducer.coef_.T
    elif method == 'pca':
        components = reducer.components_.T
    projections = np.matmul(r_value[:N], components)
    fig, axs = plt.subplots()
    # axs = Axes3D(fig, auto_add_to_figure=False)
    # fig.add_axes(axs)
    sc = axs.scatter(projections[:, 0], projections[:, 1], c=reach_action[:N])

    _, _, axs, sc = plot_embedding(r_value[:N], dim, node_right[:N], method=method, embedding=r_value_embedding)
    handles = sc.legend_elements(num=list(np.arange(-4, 5)))[0]  # extract the handles from the existing scatter plot
    axs.legend(title='Dis to\n agent right', handles=handles, labels=list(np.arange(-4, 5)), bbox_to_anchor=(1, 1))

    _, _, axs, sc = plot_embedding(r_value[:N], dim, node_front[:N], method=method, embedding=r_value_embedding)
    handles = sc.legend_elements(num=list(np.arange(-4, 5)))[0]  # extract the handles from the existing scatter plot
    axs.legend(title='Dis to\n agent front', handles=handles, labels=list(np.arange(-4, 5)), bbox_to_anchor=(1, 1))

    # we visualize the w values at step g and afterwards
    identities = []
    id_colors = ['green', 'lightgreen']

    # gw1_embedding = reducer.transform(goal_w_value[0])
    # gw2_embedding = reducer.transform(goal_w_value[1])
    gw1_embedding = np.matmul(goal_w_value[0], components)[:, [0, 1]]
    gw2_embedding = np.matmul(goal_w_value[1], components)[:, [0, 1]]

    identities.extend([id_colors[0]] * len(goal_w_value[0]))
    identities.extend([id_colors[1]] * len(goal_w_value[1]))
    value_embedding = np.concatenate((gw1_embedding, gw2_embedding), axis=0)
    identities = np.array(identities)
    fff = np.concatenate((goal_w_value[0], goal_w_value[1]), axis=0)
    _, _, axs, sc = plot_embedding(goal_w_value, dim, identities, embedding=value_embedding, method=method)
    # axs.set_xlim([-2.1, 2.1])
    # axs.set_ylim([-1.5, 0.8])
    # axs.set_xticks([-2, -1, 0, 1, 2])
    # axs.set_xticklabels([-2, -1, 0, 1, 2])
    handles = sc.legend_elements(num=id_colors)[0]  # extract the handles from the existing scatter plot
    # axs.legend(title='Identity', handles=handles, labels=['for step g', 'for step g+1'], bbox_to_anchor=(1, 1), fontsize=28)

    ## we plot the histogram of the sum of the x coordinates of the two projections of w values
    sum_x = (gw1_embedding + gw2_embedding)[:, 0]
    sum_x[np.where(sum_x > 3)[0]] = 0
    sum_y = (gw1_embedding + gw2_embedding)[:, 1]
    sum_y[np.where(sum_y < -8)[0]] = 0
    fig, axs = plt.subplots()
    import seaborn as sns

    sns.histplot(data=sum_y, stat='probability', edgecolor='k', facecolor='white', label='sum along $2^{nd}$ PC',
                 bins=10)
    sns.histplot(data=sum_x, stat='probability', color='k', label='sum along $1^{st}$ PC', bins=10)
    # axs.legend()
    axs.set_xlabel('value')
    axs.set_ylabel('density')

    # construct based on the equation
    y_g, y_g_next = gw1_embedding[:, 1], gw2_embedding[:, 1]
    idx1 = np.where(goal_reach_index == 1)[0]
    g_coloring = node_right[idx1]
    # g_coloring = np.where(node_right[idx1] >= 0, 2, 1)
    # g_coloring[0] = 0

    dot_g, dot_next_g, sum_g, coloring = [], [], [], []
    mm = []
    for eps in range(len(y_g_next)):
        mm.extend(np.matmul(goal_w_key[1][eps], r_key_separate[eps].T))
        dot_g.extend(np.matmul(goal_w_key[0][eps], r_key_separate[eps].T) * y_g[eps] * 0.3)
        dot_next_g.extend(np.matmul(goal_w_key[1][eps], r_key_separate[eps].T) * y_g_next[eps])
        sum_g.extend(
            np.matmul(goal_w_key[0][eps], r_key_separate[eps].T) * y_g[eps] * 0.3 +
            np.matmul(goal_w_key[1][eps], r_key_separate[eps].T) * y_g_next[eps]
        )
        coloring.extend([g_coloring[eps]] * len(r_key_separate[eps]))

    random_x = np.random.randn(len(y_g_next)) * 0.1
    fig, axs = plt.subplots()
    s_size = 10
    axs.scatter(random_x, y_g_next, c=g_coloring, cmap='viridis', s=s_size)
    random_x_1 = np.random.randn(len(dot_next_g)) * 0.1 + 0.8
    axs.scatter(random_x_1, dot_g, c=coloring, cmap='viridis', s=s_size)
    axs.scatter(random_x_1 + 0.8, dot_next_g, c=coloring, cmap='viridis', s=s_size)
    sc = axs.scatter(random_x_1 + 0.8 * 2, sum_g, c=coloring, cmap='viridis', s=s_size)
    handles = sc.legend_elements(num=np.arange(-4, 5, 1))[0]  # extract the handles from the existing scatter plot
    axs.set_xticks([0, 0.8, 1.6, 2.4])
    axs.set_xticklabels(['$y_{g+1}$', '$C_1$', '$C_2$', '$C_1+C_2$'], fontsize=30)
    axs.set_yticks([])
    # axs.legend(title='Dis to\n agent right', handles=handles, labels=list(np.arange(-4, 5)), bbox_to_anchor=(1, 1))

    # 2d pca of retrieving results based on the writing two steps after goal reached
    rr_actions = np.copy(reach_action[:N])
    embedding = reducer.transform(retrieved1[:N])
    components, _, axs, sc = plot_embedding(retrieved1[:N], dim, rr_actions, embedding=embedding, method=method)
    handles = sc.legend_elements(num=[0, 1, 3])[0]
    # axs.legend(handles=handles, labels=['forward', 'left turn', 'right turn'], loc="upper right", fontsize=36)
    # axs.set_title('2D pca of two step retrieved, colored by actions')
    # # take those idx where the values in the last retrieved are negative
    idx1 = np.where((dot_r_goal_w_key > 0)[:N])[0]
    # take those idx where the values in this retrieved are positive
    idx2 = np.where((dot_r_goal_w_key <= 0)[:N])[0]
    # take those idx where the values in this retrieved are negative
    colors = np.zeros((len(embedding), 4))
    cmap = mat.colormaps['viridis']
    cmap = cmap(np.linspace(0, 1, maze_size))
    colors[idx1] = cmap[0]
    colors[idx2] = cmap[3]
    _, _, axs, sc = plot_embedding(retrieved1[:N], dim, colors, embedding=embedding, method=method)
    handles = sc.legend_elements(num=[cmap[0], cmap[3]])[0]
    axs.legend(handles=handles, fontsize=36,
               labels=['P > 0', 'P \u2264 0'],
               loc="best")
    # axs.set_title('2D pca of two step retrieved, colored by dot product')

    # extract all reading keys corresponding to north and south
    method = 'pca'
    idxn = np.where(step_ori[:N] == orientations['north'])[0]
    idxs = np.where(step_ori[:N] == orientations['south'])[0]
    idx = np.concatenate((idxn, idxs))
    reducer, embedding_r_key, axs, sc = plot_embedding(r_key[:N][idx], 2, node_y[:N][idx], method=method)
    components = reducer.components_  # with pca we extract the pc
    handles = sc.legend_elements(num=[i for i in range(maze_size)])[0]
    axs.legend(title='Coordinate Y', handles=handles, labels=[i for i in range(maze_size)],
               bbox_to_anchor=(1, 1))
    axs.set_title('2D pca of r keys with facing north and south, colored by Y coordinate')

    # we project the goal w keys in the same pca space as in north and south
    embedding_w_key = reducer.transform(goal_w_key[0])
    embedding = np.concatenate((embedding_r_key, embedding_w_key))
    values = np.concatenate((goal_w_key[0], r_key[:N][idx]))
    cmap = mat.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, maze_size))
    goal_y_colors = [colors[x] for x in goal_node_y]
    r_key_colors = ['gray'] * len(idx)
    _, embedding, axs, sc = plot_embedding(values, 2, r_key_colors + goal_y_colors, embedding=embedding, method=method)
    handles = sc.legend_elements(num=['gray'])[0]
    axs.legend(title='Goal\nCoordinate Y', handles=handles, labels=['r key'], bbox_to_anchor=(1, 1))
    axs.set_title('r keys with facing N and S, together with goal w keys')

    # now we express the entire r and w keys in the coordinate of its pca components + residuels
    r_key_ = r_key[:N][idx]
    r_key_pca = np.matmul(embedding_r_key, components)
    w_keys = goal_w_key
    w_keys_pca = [np.matmul(reducer.transform(goal_w_key[0]), components),
                  np.matmul(reducer.transform(goal_w_key[1]), components)]
    w_values_embedding = [gw1_embedding, gw2_embedding]
    # compute the average of w_key, w_value at g and g+1
    avg_g_k, avg_g_k_pca = np.zeros((2, 5, 5, args['k_size'])), np.zeros((2, 5, 5, args['k_size']))
    avg_g_w_value_x = np.zeros((2, 5, 5))
    avg_g_w_value_y = np.zeros((2, 5, 5))
    for x in range(5):
        for y in range(5):
            idx_ = np.where((goal_node_x == x) * (goal_node_y == y))
            for g in range(2):
                avg_g_k[g, x, y] = np.mean(w_keys[g][idx_], axis=0)
                avg_g_k_pca[g, x, y] = np.mean(w_keys_pca[g][idx_], axis=0)
                avg_g_w_value_x[g, x, y] = np.mean(w_values_embedding[g][idx_, 0])
                avg_g_w_value_y[g, x, y] = np.mean(w_values_embedding[g][idx_, 1])

    colors = ['b', 'orange', 'k', 'orange']
    legends = ['$k_g \cdot q_t$      (1)', '$k_g^{pca} \cdot q_t^{pca} $', '$k_{g+1} \cdot q_t$  (3)', '(1) - (3)']
    lines = ['solid'] * 3 + ['dashed']

    # now we fix the node y to be 2
    current_node_y = 2
    current_node_x = 2
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 5.5))
    for col, ori in enumerate([1, 3]):
        avg_q = np.zeros((5, 5, args['k_size']))
        avg_q_pca = np.zeros((5, 5, args['k_size']))
        for x in range(5):
            for y in range(5):
                idx_ = np.where((node_x[:N][idx] == x) * (node_y[:N][idx] == y) * (step_ori[:N][idx] == ori))
                avg_q[x, y] = np.mean(r_key_[idx_], axis=0)
                avg_q_pca[x, y] = np.mean(r_key_pca[idx_], axis=0)

        dot_original = np.matmul(avg_g_k, avg_q[:, current_node_y, :].T)  # (2, goal x, goal y)
        dot_pca = np.matmul(avg_g_k_pca, avg_q_pca[:, current_node_y, :].T)  # (2, goal x, goal y)
        # we average the dot products along HD (now Y)
        axs[col].errorbar(range(5), dot_original[0].mean(axis=(0, 2)), dot_original[0].std(axis=(0, 2)), marker='^',
                          color='b',linewidth=1.0,
                          label='$k_g \cdot q_t$')

        axs[col].errorbar(range(5), dot_pca[0].mean(axis=(0, 2)), dot_pca[0].std(axis=(0, 2)), marker='^',
                          color='orange',linewidth=1.0,
                          label='$k^{pca}_{g} \cdot q^{pca}_t$')

        axs[col].errorbar(range(5), dot_original[1].mean(axis=(0, 2)), dot_original[1].std(axis=(0, 2)), marker='^',
                          color='k',linewidth=1.0,
                          label='$k_{g+1} \cdot q_t$')

        diff = dot_original[0] - dot_original[1]
        axs[col].errorbar(range(5), diff.mean(axis=(0, 2)), diff.std(axis=(0, 2)), marker='^', color='orange',linewidth=1.0,
                          linestyle='dashed')
        axs[col].grid(True)
        axs[col].set_ylim([-4, 7.3])
        axs[col].set_xticks([0, 1,2,3,4])
        # dot_sum_1 = dot_original[0] * avg_g_w_value_x[0] + dot_original[1] * avg_g_w_value_x[1]
        # dot_sum_2 = dot_original[0] * avg_g_w_value_y[0] + dot_original[1] * avg_g_w_value_y[1]
        # axs[1].errorbar(range(5), dot_sum_2.mean(axis=1), dot_sum_2.std(axis=1), marker='^', color='orange',
        #                 label='$sum along 2nd pc$')
        #
        # axs[0].grid(True)
        # axs[1].grid(True)
        # axs[l][col].imshow(r_gw_dot[l].T, vmin=5, vmax=15)
        # axs[l][col].set_ylabel(legends[l])
        # axs[l][col].set_axis_off()
        # axs[0][col].set_title(' HD %s' % ori)
        # if col == 1:  axs[col].legend(bbox_to_anchor=(1.05, 1))
        # plt.legend(bbox_to_anchor=(1.1,1.05))
        plt.tight_layout()

### For supplementary Fig. 9 ####
draw4 = False
# examine place (node) cells in the reading keys, goal cells in the writing keys and action cells in the readouts
if draw4:
    stage = 2000
    env = args['env']
    simu = 1
    # load the data
    data = pickle.load(open(save_dir + '/data/test/ppo_%s_fwm_%s_%s.pkl' % (env, simu, stage), 'rb'))
    goals = data['goal_location']
    memory_traces = data['memory_traces']
    step_reward = np.squeeze(data['step_reward'])
    trajs = data['trajectories']
    orientations = ['east', 'north', 'west', 'south'] # index 0, 1, 2 ,3
    ############################################################
    mem_items = ['r_key', 'r_value', 'w_key', 'w_value', 'rnn_out', 'rnn_cell']
    target = 'r_key'
    time_window = 'back2goal' # on_goal | back2goal
    vector_dim = args['k_size'] if not 'rnn' in target else args['hidden']
    # we collect the memory vector at each location and orientation
    unit_map = np.zeros((4, maze_size, maze_size, vector_dim)) # so 4 orientations, 5 by 5 grids
    s_occupancy = np.zeros((4, maze_size, maze_size)) # record the visiting frequency for each state
    if target == 'r_value': # r_value represent ego-centric goal location
        agent_nodes = [0, 6, 12, 18, 24] # fix the agent's location
        g_unit_map, g_s_occupancy = {}, {}
        for a_node in agent_nodes:
            g_unit_map[a_node] = np.zeros((4, maze_size, maze_size, vector_dim))
            g_s_occupancy[a_node] = np.zeros((4, maze_size, maze_size))
    # unit map in cordinate of how forward (behind) and how left (right) of the goal to the agent
    ecg_map = np.zeros((2*maze_size-1, 2*maze_size-1, vector_dim))
    ecg_map_occupancy = np.zeros((2*maze_size-1, 2*maze_size-1))
    # tuning curves for egocentric direction and distance of the goal
    ec_dis_curve = np.zeros((2*maze_size-1, vector_dim)) # with maze size 5, possible distance 1 to 8
    ec_dis_occupancy = np.zeros((2 * maze_size - 1, ))
    ### iterate all possible relative angles, given the HD of the agent
    HDs = [0, 90, -180, -90]
    rel_angles = []
    for r_row in range(-maze_size+1, maze_size):  # relative row
        for r_col in range(-maze_size+1, maze_size): # relative col
            if r_row == 0 and r_col == 0: continue
            for hd in HDs:
                rel_angles.append(compute_rel_angle(hd, r_row, r_col))
    rel_angles = np.unique(np.round(rel_angles, 1))
    # build an map between angle and its index
    rel_angles_idx = {}
    for i, ang in enumerate(rel_angles): rel_angles_idx[ang] = i
    ec_angle_curve = np.zeros((len(rel_angles), vector_dim))
    ec_angle_occupancy = np.zeros((len(rel_angles), ))
    #########################################################
    for eps in range(len(goals)):
        mem_item = np.squeeze(memory_traces[eps][target])
        if activation == 'tanh': mem_item = mem_item + 1
        # extract the trajectory
        eps_traj = []
        for tr in trajs[eps][0]:
            eps_traj.extend(tr)
        eps_node, eps_ori = [], []
        for x in eps_traj:
            eps_node.append(x[0])
            eps_ori.append((x[1]//90)%4)
        eps_traj = np.array(eps_traj)
        # extract time steps where the goal is found for the first time
        goal_reaching = np.where(step_reward[eps] > 0)[0]
        for i in range(len(sub_eps_bounds) - 1):
            # extract the goal reaching for the subepisode
            x = goal_reaching > sub_eps_bounds[i]
            y = goal_reaching < sub_eps_bounds[i + 1]
            sub_reaching = goal_reaching[np.where(x * y)[0]]
            # if a goal is reached more than 5 times, the agent remembers it
            if len(sub_reaching) > 4:
                # 1 step after the first time receiving the reward, the agent stands on the goal
                step_g = sub_reaching[0] + 1
                # if we examine writing value, then only step_g matters
                if time_window == 'on_goal':
                    indice = [step_g]
                else:
                    # we remove the time step when the agent is actually on the goal because it is not relevant much
                    indice = []
                    for j in range(len(sub_reaching) - 1):
                        indice.extend(list(range(sub_reaching[j] + 2, sub_reaching[j + 1] + 1)))
                    indice.extend(list(range(sub_reaching[-1] + 2, sub_eps_bounds[i + 1])))
                # take out the r keys of going back to goals
                for step in indice:
                    a_node = eps_node[step]
                    row = a_node % maze_size
                    col = a_node // maze_size
                    unit_map[eps_ori[step]][row][col] += mem_item[step]
                    s_occupancy[eps_ori[step]][row][col] += 1
                    if target == 'r_value':
                        g_node = np.squeeze(goals[eps][sub_reaching[0]])
                        g_row = g_node % maze_size
                        g_col = g_node // maze_size
                        if a_node in agent_nodes:
                            # for this agent node and orientation
                            g_unit_map[a_node][eps_ori[step]][g_row][g_col] += mem_item[step]
                            g_s_occupancy[a_node][eps_ori[step]][g_row][g_col] += 1
                        # if goal is on the front, right of the agent based on HD and relative position, then
                        # both row diff and col diff are positive
                        if eps_ori[step]==1: # north
                            forward_dis = (g_row - row)
                            right_dis = (g_col - col)
                        elif eps_ori[step] == 2: # west
                            right_dis = (g_row - row)
                            forward_dis = -(g_col - col)
                        elif eps_ori[step] == 3: # south
                            forward_dis = -(g_row - row)
                            right_dis = -(g_col - col)
                        elif eps_ori[step] == 0: # east
                            right_dis = -(g_row - row)
                            forward_dis = (g_col - col)
                        ecg_map[forward_dis+maze_size-1][right_dis+maze_size-1] += mem_item[step]
                        ecg_map_occupancy[forward_dis+maze_size-1][right_dis+maze_size-1] += 1
                        # build tuing curves
                        manh_dis = np.abs(g_row - row) + np.abs(g_col - col)
                        ec_dis_curve[manh_dis] += mem_item[step]
                        ec_dis_occupancy[manh_dis] += 1
                        if manh_dis == 0: continue # no relative angle here
                        angle = compute_rel_angle(eps_ori[step]*90, g_row - row, g_col - col)
                        angle = np.round(angle, 1)
                        ec_angle_curve[rel_angles_idx[angle]] += mem_item[step]
                        ec_angle_occupancy[rel_angles_idx[angle]] += 1

    # compute the average activities at each state
    s_occupancy = np.expand_dims(s_occupancy+1e-4, axis=-1)
    unit_map /= np.tile(s_occupancy, (1, vector_dim))
    # normlize the unit activity map
    unit_map /= (np.max(np.abs(unit_map))+1e-4)
    # compute the max std of the activity map for each unit
    shape = unit_map.shape
    # we only consider the std around the 2d grid
    unit_map_std = np.std(np.reshape(unit_map, (shape[0], shape[1]*shape[2], shape[3])), axis=1)
    unit_map_std = np.max(unit_map_std) # take the max for comparing
    # also plot the mean of four head directions
    unit_map_m = np.mean(unit_map, axis=0)
    unit_map_m /= (np.max(np.abs(unit_map_m)) + 1e-4)
    if target == 'r_value':
        g_unit_map_std, g_unit_map_m = {}, {}
        for a_node in agent_nodes:
            # compute the average activities at each state
            g_s_occupancy[a_node] = np.expand_dims(g_s_occupancy[a_node] + 1e-4, axis=-1)
            g_unit_map[a_node] /= np.tile(g_s_occupancy[a_node], (1, vector_dim))
            # normlize the unit activity map
            g_unit_map[a_node] /= (np.max(np.abs(g_unit_map[a_node])) + 1e-4)
            # compute the max std of the activity map for each unit
            # we only consider the std around the 2d grid
            g_unit_map_std[a_node] = np.std(np.reshape(g_unit_map[a_node], (shape[0], shape[1] * shape[2], shape[3])), axis=1)
            g_unit_map_std[a_node] = np.max(g_unit_map_std[a_node])  # take the max for comparing
            # also plot the mean of four head directions
            g_unit_map_m[a_node] = np.mean(g_unit_map[a_node], axis=0)
            g_unit_map_m[a_node] /= (np.max(np.abs(g_unit_map_m[a_node])) + 1e-4)
        # normalize the true ecg location unit map
        ecg_map_occupancy = np.expand_dims(ecg_map_occupancy + 1e-4, axis=-1)
        ecg_map /= np.tile(ecg_map_occupancy, (1, vector_dim))
        ecg_map /= (np.max(np.abs(ecg_map)) + 1e-4)
        shape = ecg_map.shape
        # we only consider the std around the 2d grid
        ecg_map_std = np.std(np.reshape(ecg_map, (shape[0] * shape[1], shape[2])), axis=0)
        ecg_map_std = np.max(ecg_map_std)  # take the max for comparing
        # normalize tuning curves
        ec_dis_occupancy = np.expand_dims(ec_dis_occupancy + 1e-4, axis=-1)
        ec_dis_curve /= np.tile(ec_dis_occupancy, (1, vector_dim))
        ec_dis_curve /= (np.max(np.abs(ec_dis_curve)) + 1e-4)
        # filter out small values
        ec_dis_curve = np.where(ec_dis_curve<0.01, 0, ec_dis_curve)

        ec_angle_occupancy = np.expand_dims(ec_angle_occupancy + 1e-4, axis=-1)
        ec_angle_curve /= np.tile(ec_angle_occupancy, (1, vector_dim))
        ec_angle_curve /= (np.max(np.abs(ec_angle_curve)) + 1e-4)
        # filter out small values
        ec_angle_curve = np.where(ec_angle_curve < 0.01, 0, ec_angle_curve)

    # folder to store images
    image_folder = save_dir + '/results/activity_map/simu%s_%s_%s/varied' % (simu, target, time_window)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if target == 'r_value':
        for a_node in agent_nodes:
            image_folder_g = save_dir + '/results/activity_map/simu%s_%s_%s/agent %s/varied' % (simu, target, time_window, a_node)
            if not os.path.exists(image_folder_g):
                os.makedirs(image_folder_g)
        image_folder_ecg = save_dir + '/results/activity_map/simu%s_%s_%s/ecg/varied' % (simu, target, time_window)
        if not os.path.exists(image_folder_ecg):
            os.makedirs(image_folder_ecg)
        image_folder_eg_dis = save_dir + '/results/activity_map/simu%s_%s_%s/eg_dis_curve' % (simu, target, time_window)
        if not os.path.exists(image_folder_eg_dis):
            os.makedirs(image_folder_eg_dis)
        image_folder_eg_angle = save_dir + '/results/activity_map/simu%s_%s_%s/eg_angle_curve' % (simu, target, time_window)
        if not os.path.exists(image_folder_eg_angle):
            os.makedirs(image_folder_eg_angle)
    # iterate over units
    for unit in range(vector_dim):
        fig = plt.figure(figsize=(12, 10))
        axE = fig.add_axes([0.6, 0.4, 0.2, 0.2])
        axN = fig.add_axes([0.4, 0.6, 0.2, 0.2])
        axW = fig.add_axes([0.2, 0.4, 0.2, 0.2])
        axS = fig.add_axes([0.4, 0.2, 0.2, 0.2])
        axes = [axE, axN, axW, axS]
        for i, ax in enumerate(axes):
            im = ax.imshow(unit_map[i, :, :, unit], cmap='jet', vmin=0, vmax=1,
                           origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
        cb_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_ticks(np.arange(0, 1.1, 0.25))
        # store the image
        if np.prod(np.std(unit_map[:,:,:,unit], axis=(1, 2)) < 0.2*unit_map_std) and \
                np.prod(np.max(unit_map[:,:,:,unit], axis=(1, 2)) < 0.2):
            plt.savefig(image_folder+'/../unit%s'% unit)
        else:
            plt.savefig(image_folder + '/unit%s' % unit)
        plt.close(fig)
        # and activity map relative to the given goal locations
        if target == 'r_value':
            # plot in coordinate of HD, dis2g
            fig, axs = plt.subplots(figsize=(8,6))
            im = axs.imshow(ecg_map[:,:,unit], cmap='jet', vmin=0, vmax=1, origin='lower')
            im_ratio = ecg_map_occupancy.shape[0]/ecg_map_occupancy.shape[1]
            cbar = fig.colorbar(im, ax=axs, fraction=0.047*im_ratio)
            cbar.set_ticks(np.arange(0, 1.1, 0.5))
            axs.set_xlabel('Right goal distance')
            axs.set_xticks(np.arange(0, 2*maze_size-1, 1))
            axs.set_xticklabels(np.arange(-maze_size+1, maze_size, 1))
            axs.set_ylabel('Forward goal distance')
            axs.set_yticks(np.arange(0, 2*maze_size-1, 1))
            axs.set_yticklabels(np.arange(-maze_size+1, maze_size, 1))
            axs.set_title('%s Unit %s, activity map' % (target, unit))
            # store the image
            plt.savefig(image_folder_ecg + '/../unit%s' % unit)
            plt.close(fig)
            ### for each given fixed agent's location
            for a_node in agent_nodes:
                a_col = a_node // maze_size
                a_row = a_node % maze_size
                fig = plt.figure(figsize=(12, 10))
                axE = fig.add_axes([0.6, 0.4, 0.2, 0.2])
                axN = fig.add_axes([0.4, 0.6, 0.2, 0.2])
                axW = fig.add_axes([0.2, 0.4, 0.2, 0.2])
                axS = fig.add_axes([0.4, 0.2, 0.2, 0.2])
                axes = [axE, axN, axW, axS]
                for i, ax in enumerate(axes):
                    im = ax.imshow(g_unit_map[a_node][i, :, :, unit], cmap='jet', vmin=0, vmax=1,
                                   origin='lower')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.plot(a_col, a_row, marker='x', markersize=10, color='r', mew=6)
                cb_ax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
                cbar = fig.colorbar(im, cax=cb_ax)
                cbar.set_ticks(np.arange(0, 1.1, 0.25))
                # store the image
                image_folder_g = save_dir + '/results/activity_map/simu%s_%s_%s/agent %s/varied' % (
                    simu, target, time_window, a_node)
                if  np.prod(np.std(g_unit_map[a_node][:, :, :, unit], axis=(1, 2)) < 0.2 * g_unit_map_std[a_node]) and \
                        np.prod(np.max(g_unit_map[a_node][:, :, :, unit], axis=(1, 2)) < 0.2):
                    plt.savefig(image_folder_g + '/../unit%s' % unit)
                else:
                    plt.savefig(image_folder_g + '/unit%s' % unit)
                plt.close(fig)

            ## plot and store tuning curves
            fig, axs = plt.subplots(figsize=(8, 6))
            # align with the x coordinate from the experimental paper
            axs.plot(np.arange(-2*maze_size+2, 1), ec_dis_curve[:, unit][-1::-1], color='k')
            axs.set_xlabel('Distance to goal')
            axs.set_ylabel('Firing rate')
            axs.set_title('%s Unit %s, tuning curve' % (target, unit))
            plt.savefig(image_folder_eg_dis + '/unit%s' % unit)
            plt.close(fig)
            #####################
            fig, axs = plt.subplots(figsize=(8, 6))
            axs.plot(rel_angles, ec_angle_curve[:, unit], color='b')
            axs.set_xlabel('Goal direction')
            axs.set_xticks([-180, 0, 180])
            axs.set_xticklabels([-180, 0, 180])
            axs.set_ylabel('Firing rate')
            axs.set_title('%s Unit %s, tuning curve' % (target, unit))
            plt.savefig(image_folder_eg_angle + '/unit%s' % unit)
            plt.close(fig)

### For Fig. 6 ####
draw5 = False
# compare with the bat experiment, find tuning curves
if draw5:
    ## set the font size of the plots
    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 35}
    mat.rc('font', **font)
    stage = args['training_episodes']  # 500, 1000, 1500, args['training_episodes']
    env = args['env']
    simus = [1,2,3,4,5,6]
    im_store = False
    # we store all units which have a valid tuning curve
    tuning_goal_angle = {}
    tuning_goal_dis = {}
    def fit_gaussian(x, y):
        def gaus(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
        n = len(x)  # the number of data
        mean = sum(x * y) / n
        sigma = sum(y * (x - mean) ** 2) / n
        try:
            popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])  # p0=[1, mean, sigma]
            popt = np.round(popt, 2)
        except RuntimeError:
            popt, pcov = None, None
        return popt, pcov

    for simu in simus:
        tuning_goal_angle['simu_%s'%simu] = {}
        tuning_goal_dis['simu_%s' % simu] = {}
        # load the data
        data = pickle.load(open(save_dir + '/data/test/ppo_%s_%s_%s_%s.pkl' % (env, network, simu, stage), 'rb'))
        goals = data['goal_location']
        memory_traces = data['memory_traces']
        step_reward = np.squeeze(data['step_reward'])
        trajs = data['trajectories']
        orientations = ['east', 'north', 'west', 'south']  # index 0, 1, 2 ,3
        ############################################################
        target = 'r_value'
        time_window = 'back2goal'  # on_goal | back2goal
        vector_dim = args['k_size'] if not 'rnn' in target else args['hidden']
        # tuning curves for egocentric direction and distance of the goal
        ec_dis_curve = np.zeros((2 * maze_size - 2, vector_dim))  # with maze size 5, possible distance 1 to 8
        ec_dis_occupancy = np.zeros((2 * maze_size - 2,))
        ### iterate all possible relative angles, given the HD of the agent
        HDs = [0, 90, -180, -90]
        rel_angles = []
        for r_row in range(-maze_size + 1, maze_size):  # relative row
            for r_col in range(-maze_size + 1, maze_size):  # relative col
                if r_row == 0 and r_col == 0: continue
                for hd in HDs:
                    rel_angles.append(compute_rel_angle(hd, r_row, r_col))
        rel_angles = np.unique(np.round(rel_angles, 1))
        # build an map between angle and its index
        rel_angles_idx = {}
        for i, ang in enumerate(rel_angles): rel_angles_idx[ang] = i
        ec_angle_curve = np.zeros((len(rel_angles), vector_dim))
        ec_angle_occupancy = np.zeros((len(rel_angles),))
        #########################################################
        for eps in range(len(goals)):
            mem_item = np.squeeze(memory_traces[eps][target])
            #if activation == 'tanh': mem_item = (mem_item + 1)/2
            # extract the trajectory
            eps_traj = []
            for tr in trajs[eps][0]:
                eps_traj.extend(tr)
            eps_node, eps_ori = [], []
            for x in eps_traj:
                eps_node.append(x[0])
                eps_ori.append((x[1] // 90) % 4)
            eps_traj = np.array(eps_traj)
            # extract time steps where the goal is found for the first time
            goal_reaching = np.where(step_reward[eps] > 0)[0]
            for i in range(len(sub_eps_bounds) - 1):
                # extract the goal reaching for the subepisode
                x = goal_reaching > sub_eps_bounds[i]
                y = goal_reaching < sub_eps_bounds[i + 1]
                sub_reaching = goal_reaching[np.where(x * y)[0]]
                # if a goal is reached more than 5 times, the agent remembers it
                if len(sub_reaching) > 4:
                    # 1 step after the first time receiving the reward, the agent stands on the goal
                    step_g = sub_reaching[0] + 1
                    # if we examine writing value, then only step_g matters
                    if time_window == 'on_goal':
                        indice = [step_g]
                    else:
                        # we remove the time step when the agent is actually on the goal because it is not relevant much
                        indice = []
                        for j in range(len(sub_reaching) - 1):
                            indice.extend(list(range(sub_reaching[j] + 2, sub_reaching[j + 1] + 1)))
                        indice.extend(list(range(sub_reaching[-1] + 2, sub_eps_bounds[i + 1])))
                    # take out the r keys of going back to goals
                    for step in indice:
                        a_node = eps_node[step]
                        row = a_node % maze_size
                        col = a_node // maze_size
                        g_node = goals[eps][sub_reaching[0]][0][0]
                        g_row = g_node % maze_size
                        g_col = g_node // maze_size
                        # build tuing curves
                        manh_dis = np.abs(g_row - row) + np.abs(g_col - col)
                        ec_dis_curve[manh_dis-1] += mem_item[step] # distance 1 has index 0
                        ec_dis_occupancy[manh_dis-1] += 1
                        if manh_dis == 0: continue  # no relative angle here
                        angle = compute_rel_angle(eps_ori[step] * 90, g_row - row, g_col - col)
                        angle = np.round(angle, 1)
                        ec_angle_curve[rel_angles_idx[angle]] += mem_item[step]
                        ec_angle_occupancy[rel_angles_idx[angle]] += 1
        # normalize tuning curves, goal angle
        ec_dis_occupancy = np.expand_dims(ec_dis_occupancy + 1e-4, axis=-1)
        ec_dis_curve /= np.tile(ec_dis_occupancy, (1, vector_dim))
        ec_dis_curve /= (np.max(np.abs(ec_dis_curve)) + 1e-4)
        # filter out small values
        ec_dis_curve = np.where(ec_dis_curve < 0.01, 0, ec_dis_curve)
        # goal distance
        ec_angle_occupancy = np.expand_dims(ec_angle_occupancy + 1e-4, axis=-1)
        ec_angle_curve /= np.tile(ec_angle_occupancy, (1, vector_dim))
        ec_angle_curve /= (np.max(np.abs(ec_angle_curve)) + 1e-4)
        # filter out small values
        ec_angle_curve = np.where(ec_angle_curve < 0.01, 0, ec_angle_curve)
        # folder to store images
        image_folder_eg_dis = save_dir + '/results/tuning_curve/simu%s_%s_%s/eg_dis_curve/fitted' % (simu, target, time_window)
        if not os.path.exists(image_folder_eg_dis):
            os.makedirs(image_folder_eg_dis)
        image_folder_eg_angle = save_dir + '/results/tuning_curve/simu%s_%s_%s/eg_angle_curve/fitted' % (
            simu, target, time_window)
        if not os.path.exists(image_folder_eg_angle):
            os.makedirs(image_folder_eg_angle)
        # iterate over units
        for unit in range(vector_dim):
            ##### for goal angle
            fig, axs = plt.subplots(figsize=(8, 6))
            axs.plot(rel_angles, ec_angle_curve[:, unit], color='b', linewidth=6)
            axs.set_xlabel('Goal direction')
            axs.set_xticks([-180, 0, 180])
            axs.set_xticklabels([-180, 0, 180])
            axs.set_yticks([0, np.round(max(ec_angle_curve[:, unit]),2)])
            axs.set_yticklabels([0, np.round(max(ec_angle_curve[:, unit]),2)])
            axs.spines[['right', 'top']].set_visible(False)
            axs.spines[['bottom', 'left']].set_linewidth(3)
            axs.set_ylabel('Firing rate')
            ## fit guassian to the tuning curves
            # in case the center of gaussian is around 180 degrees, we also examine a rotated version of the data such that
            # angle 180, -180 is at the center
            num_angle = len(rel_angles)
            new_idx = [(num_angle//2 + i) - ((num_angle//2 + i)//num_angle)*num_angle for i in range(num_angle)]
            new_rel_angles = (rel_angles[new_idx] + 360) - ((rel_angles[new_idx] + 360)//360)*360
            new_ec_angle_curve = ec_angle_curve[new_idx]
            # start fitting
            param, cov = fit_gaussian(rel_angles, ec_angle_curve[:, unit]) # param: [magnitude, mean, variance] of the Gaussian
            if param is None:
                param, cov = fit_gaussian(new_rel_angles, new_ec_angle_curve[:, unit])
            #axs.set_title('%s Unit %s, param: %s' % (target, unit, param))
            plt.tight_layout()
            if param is None:
                if im_store: plt.savefig(image_folder_eg_angle + '/../unit%s' % unit)
            else:
                if param[0] > 0.01 and abs(param[1]) < 360 and abs(param[2]) > 0.01:
                    tuning_goal_angle['simu_%s'%simu][unit] = param[1]
                    if im_store: plt.savefig(image_folder_eg_angle + '/unit%s' % unit)
                else:
                    if im_store: plt.savefig(image_folder_eg_angle + '/../unit%s' % unit)
            plt.close(fig)

            ###### for goal distances
            fig, axs = plt.subplots(figsize=(8, 6))
            # align with the x coordinate from the experimental paper
            axs.plot(np.arange(-2 * maze_size + 2, 0), ec_dis_curve[:, unit][-1::-1], color='k',linewidth=5)
            axs.set_xlabel('Distance to goal')
            axs.set_ylabel('Firing rate')
            axs.set_xticks([-2 * maze_size + 2, -1])
            axs.set_xticklabels([-2 * maze_size + 2, -1])
            axs.set_yticks([0, np.round(max(ec_dis_curve[:, unit]),2)])
            axs.set_yticklabels([0, np.round(max(ec_dis_curve[:, unit]),2)])
            axs.spines[['right', 'top']].set_visible(False)
            axs.spines[['bottom', 'left']].set_linewidth(3)
            ## fit guassian curves the tuning curves
            param, cov = fit_gaussian(np.arange(-2 * maze_size + 2, 0),
                                      ec_dis_curve[:, unit][-1::-1])  # param: [magnitude, mean, variance] of the Gaussian
            #axs.set_title('%s Unit %s, param: %s' % (target, unit, param))
            plt.tight_layout()
            if param is None:
                if im_store: plt.savefig(image_folder_eg_dis + '/../unit%s' % unit)
            else:
                if param[0] > 0.05 and abs(param[1]) < 10 and abs(param[2]) > 0.005:
                    tuning_goal_dis['simu_%s'%simu][unit] = param[1]
                    if im_store: plt.savefig(image_folder_eg_dis + '/unit%s' % unit)
                else:
                    if im_store: plt.savefig(image_folder_eg_dis + '/../unit%s' % unit)
            plt.close(fig)
    ## summarize the tuning curves for goal angles
    preferred_direction = []
    for simu in simus:
        preferred_direction.extend(list(tuning_goal_angle['simu_%s'%simu].values()))
    preferred_direction = np.array(preferred_direction)
    preferred_direction = np.where(preferred_direction>180, 360-preferred_direction, preferred_direction)
    preferred_direction = np.where(preferred_direction<-180, 360+preferred_direction, preferred_direction)
    fid, axs = plt.subplots()
    angle_bins = np.linspace(-180, 180, 10)
    _, bins, _ = axs.hist(preferred_direction, bins = angle_bins, density=True, color='b', edgecolor='black', linewidth=1.5)
    axs.set_ylabel('percentage')
    axs.set_xlabel('preferred direction (\xb0)')
    axs.set_xticks([np.min(bins), 0, np.max(bins)])
    axs.set_xticklabels([np.floor(np.min(bins)), 0, np.ceil(np.max(bins))])
    axs.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    ## summarize the tuning curves for goal dis
    preferred_dis = []
    for simu in simus:
        preferred_dis.extend(list(tuning_goal_dis['simu_%s' % simu].values()))
    preferred_dis = np.array(preferred_dis)
    preferred_dis = np.where(preferred_dis>0, -preferred_dis, preferred_dis)
    hist, bins = np.histogram(preferred_dis, density=True)  # compute density function
    cumu = np.cumsum(hist)
    cumu = [0] + list(cumu)
    fig, axs = plt.subplots()
    axs.plot(bins, cumu, color='k', linewidth=2.5)
    #axs.hist(preferred_dis, density=True)
    axs.set_ylabel('cumulative density')
    axs.set_xlabel('distance to goal')
    axs.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    # compute the average percentage of goal direction, goal distance units and their overlap
    g_angle_perc, g_dis_perc, g_dis_angle_per = 0, 0, 0
    for simu in simus:
        angle_units = list(tuning_goal_angle['simu_%s'%simu].keys())
        dis_units = list(tuning_goal_dis['simu_%s'%simu].keys())
        g_angle_perc += len(angle_units) / args['k_size']
        g_dis_perc += len(dis_units) / args['k_size']
        g_dis_angle_per += len(set(angle_units).intersection(set(dis_units))) / args['k_size']
    g_angle_perc /= len(simus)
    g_dis_perc /= len(simus)
    g_dis_angle_per /= len(simus)
    print(g_angle_perc, g_dis_perc, g_dis_angle_per) # 0.7078125 0.4375 0.371875


plt.show()