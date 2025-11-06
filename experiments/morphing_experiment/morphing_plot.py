'''
This script generates all the figures or subpanels used for the maze morphing task.
Please turn any of the "draw*" to "True" if you want to plot the corresponding figure.
'''


# basic imports
import os
import json
import pickle
import numpy as np
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import matplotlib.cm

from utils import *
from scipy.optimize import curve_fit
import pandas as pd
from plottable import Table, ColDef

## set the font size of the plots
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 34}
mat.rc('font', **font)

# we specify the experiment that we want to plot
scheme = 'double'  # double | single
ds_nonlinearity = False
act_space = 'egocentric'
save_dir = '%s_%s_ds%s_kvrelu_newFWM' % (scheme, act_space, ds_nonlinearity)
# load the parameters
with open(save_dir + '/args.txt', 'r') as f:
    args = json.load(f)
maze_size = 5
num_states = 4 * maze_size ** 2
stage = args['training_episodes']
episode_length = args['test_max_step']
goal_change_freq = episode_length // 2
sub_eps_bounds = np.arange(0, episode_length + 1, goal_change_freq)
network = args['network']
# determine the minimum value for the colorbar
if args['k_activation'] == 'tanh':
    vmin = -1
elif args['k_activation'] == 'relu':
    vmin = 0
test_cages = ['homecage_circle', 'homecage_4_4', 'homecage_3_5', 'homecage_2_6',
              'homecage_1_7', 'homecage_square']
cage_labels = np.array(['circle', '4:4', '3:5', '2:6', '1:7', 'square'])

orientations = {'east': 0, 'north': 1, 'west': 2, 'south': 3}

# Definición de los colores de los nodos exteriores, interiores y el central
def interpolate_color(color1, color2, factor):
    """Interpolates between two colors by a given factor (0 <= factor <= 1)"""
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    return mcolors.to_hex((1 - factor) * c1 + factor * c2)
colores_nodos = {}
maze_shape = (5, 5)
# Coordenadas de los nodos fijos
coords_fijos = {
    0: (4, 0),
    4: (0, 0),
    12: (2, 2),
    20: (4, 4),
    24: (0, 4)
}
# Definimos los colores iniciales
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
# Interpolamos los colores para los nodos intermedios
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

#### For Fig. 3a ####
# we calculate place vector code correlation
draw1 = False
if draw1:
    simus = [1,2,3,4,5]
    stage = args['training_episodes']
    mem_items = ['r_key', 'r_value', 'w_key', 'w_value', 'rnn_out', 'rnn_cell']
    target = 'r_key'
    time_window = 'back2goal'  # on_goal | back2goal
    vector_dim = args['k_size'] if not 'rnn' in target else args['hidden']
    # we compare the similarity between the population code of each env to maze of the two ends
    square_sim, circle_sim = np.zeros(len(test_cages)), np.zeros(len(test_cages))
    for simu in simus:
        # load the data
        data = pickle.load(
            open(
                save_dir + '/data/test/%s_%s_%s_%s_%s.pkl' % (args['network'], args['process_im'], scheme, simu, stage),
                'rb'))
        goals = np.array(data['goal_location'])
        memory_traces = data['memory_traces']
        step_reward = np.array(data['step_reward'])
        trajs = data['trajectories']
        ############################################################
        # we collect the memory vector at each location and orientation
        unit_map = np.zeros((len(test_cages), 4, 5, 5, vector_dim))  # so 4 orientations, 5 by 5 grids
        s_occupancy = np.zeros((len(test_cages), 4, 5, 5))  # record the visiting frequency for each state
        #########################################################
        for eps in range(len(goals)):
            for env_idx in range(len(test_cages)):
                mem_item = np.squeeze(memory_traces[eps][target])[:, env_idx]
                # extract the trajectory
                eps_traj = []
                for tr in trajs[eps][env_idx]:
                    eps_traj.extend(tr)
                eps_node, eps_ori = [], []
                for x in eps_traj:
                    eps_node.append(x[0])
                    eps_ori.append((x[1] // 90) % 4)
                eps_traj = np.array(eps_traj)
                # extract time steps where the goal is found for the first time
                goal_reaching = np.where(step_reward[eps][:, env_idx] > 0)[0]
                for i in range(len(sub_eps_bounds) - 1):
                    # extract the goal reaching for the subepisode
                    x = goal_reaching > sub_eps_bounds[i]
                    y = goal_reaching < sub_eps_bounds[i + 1]
                    sub_reaching = goal_reaching[np.where(x * y)[0]]
                    # if a goal is reached more than 5 times, the agent remembers it
                    if len(sub_reaching) > 4:
                        # 1 step after the first time receiving the reward, the agent stands on the goal
                        step_g = sub_reaching[0] + 1
                        # if we examine writing value/key, then only step_g matters
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
                            row = eps_node[step] % 5
                            col = eps_node[step] // 5
                            unit_map[env_idx][eps_ori[step]][row][col] += mem_item[step]
                            s_occupancy[env_idx][eps_ori[step]][row][col] += 1
        # compute the average activities at each state
        s_occupancy = np.expand_dims(s_occupancy + 1e-4, axis=-1)
        unit_map /= np.tile(s_occupancy, (1, vector_dim))
        active_idx = []
        for l in range(len(test_cages)):
            # pick up the units which are active in at least one env
            f, ids, pd, pc = modulated(unit_map[l])
            active_idx.extend(ids)
        active_idx = list(set(active_idx))

        unit_map = unit_map[:, :, :, :, active_idx]
        # reshape the map to shape [env_idx, state_idx, population code]
        unit_map = np.reshape(unit_map, (len(test_cages), 4 * 5 * 5, len(active_idx)))
        # reduce the mean population vector over state for each env
        unit_map -= np.tile(np.expand_dims(np.mean(unit_map, axis=1), axis=1), (1, num_states, 1))
        for l in range(len(test_cages)):
            s_temp = np.sum(unit_map[-1] * unit_map[l], axis=-1)
            s_temp /= (np.linalg.norm(unit_map[-1], axis=-1) + 1e-4) * (np.linalg.norm(unit_map[l], axis=-1) + 1e-4)
            c_temp = np.sum(unit_map[0] * unit_map[l], axis=-1)
            c_temp /= (np.linalg.norm(unit_map[0], axis=-1) + 1e-4) * (np.linalg.norm(unit_map[l], axis=-1) + 1e-4)
            square_sim[l] += s_temp.mean()
            circle_sim[l] += c_temp.mean()

    square_sim /= len(simus)
    circle_sim /= len(simus)

    fig, axs = plt.subplots()
    x = np.linspace(0, 1, 6)
    axs.plot(x, circle_sim, color='r', marker='x', label='compared to circle')
    axs.plot(x, square_sim, color='b', marker='o', label='compared to square')
    axs.set_ylim([0.15, 1.1])
    # axs.legend()
    axs.set_xticks(x)
    axs.set_xticklabels(np.array(cage_labels))
    axs.set_xlabel('Maze')
    axs.set_ylabel('PV correlation')
    axs.set_title('Training scheme: %s' % scheme)

#### For Fig. 3b ####
## correlate the mem readouts with actions, state or goal
draw2 = False
if draw2:
    simu = 1
    stage = args['training_episodes']
    envs_idx = [0,1,2,3,4,5]
    def list_list(l):
        return [[] for _ in range(l)]

    # load the data
    data = pickle.load(
        open(save_dir + '/data/test/%s_%s_%s_%s_%s.pkl' % (args['network'], args['process_im'], scheme, simu, stage),
             'rb'))
    goals = data['goal_location']
    memory_traces = data['memory_traces']
    step_reward = np.squeeze(data['step_reward'])
    trajs = data['trajectories']
    ### we analyze the stdiance reduction of the memory reading after knowing action ###
    r_value, r_key = list_list(len(envs_idx)), list_list(len(envs_idx))
    rnn_out, rnn_cell = list_list(len(envs_idx)), list_list(len(envs_idx))
    reach_action = list_list(len(envs_idx))
    step_ori, step_node = list_list(len(envs_idx)), list_list(len(envs_idx))
    goal_state, step_goal_state = list_list(len(envs_idx)), list_list(len(envs_idx))
    goal_w_value, goal_w_key, goal_next_w_key = list_list(len(envs_idx)), list_list(len(envs_idx)), list_list(len(envs_idx))

    for eps in range(len(trajs)):
        for l, e_idx in enumerate(envs_idx):
            goal_reaching = np.where(step_reward[eps][:, e_idx] > 0)[0]
            eps_r_beta = np.array(memory_traces[eps]['r_beta'])[:, e_idx]
            eps_r_value = np.array(memory_traces[eps]['r_value'])[:, e_idx]
            eps_new_w_value = np.array(memory_traces[eps]['new_w_value'])[:, e_idx]
            eps_w_value = np.array(memory_traces[eps]['w_value'])[:, e_idx]
            eps_w_beta = np.array(memory_traces[eps]['w_beta'])[:, e_idx]
            eps_r_key = np.array(memory_traces[eps]['r_key'])[:, e_idx]
            eps_w_key = np.array(memory_traces[eps]['w_key'])[:, e_idx]
            # eps_rnn_out = np.squeeze(memory_traces[eps]['rnn_out'])[:, e_idx]
            # eps_rnn_cell = np.squeeze(memory_traces[eps]['rnn_cell'])[:, e_idx]

            rw_dim = eps_r_value.shape[1]
            # also extract the trajectory
            eps_traj = []
            for tr in trajs[eps][e_idx]:
                eps_traj.extend(tr)
            eps_traj = np.array(eps_traj)
            ## for each subepisode with a fixed goal
            for i in range(len(sub_eps_bounds) - 1):
                # extract the goal reaching for the subepisode
                x = goal_reaching > sub_eps_bounds[i]
                y = goal_reaching < sub_eps_bounds[i + 1]
                sub_reaching = goal_reaching[np.where(x * y)[0]]
                # if a goal is reached more than 5 times, the agent remembers it
                if len(sub_reaching) > 4:
                    # we remove the time step when the agent is actually on the goal because it is not relevant much
                    indice = []
                    for j in range(2 - 1):
                        indice.extend(list(range(sub_reaching[j] + 2, sub_reaching[j + 1] + 1)))
                    indice.extend(list(range(sub_reaching[-1] + 2, sub_eps_bounds[i + 1])))

                    r_value[l].extend(eps_r_value[indice])
                    r_key[l].extend(eps_r_key[indice])
                    # rnn_out[l].extend(eps_rnn_out[indice])
                    # rnn_cell[l].extend(eps_rnn_cell[indice])
                    # we record the goal state, and the w_value one step after the goal reached for
                    # the first time
                    s = sub_reaching[0] + 1
                    goal_w_value[l].append(eps_new_w_value[s])
                    # the writing key when the agent is on the goal
                    goal_w_key[l].append(eps_w_key[s])
                    goal_next_w_key[l].append(eps_w_key[s+1])
                    goal_state[l].append(goals[eps][sub_reaching[0]][e_idx])
                    # during the back-to-goal period
                    for step in indice:
                        step_ori[l].append(int(eps_traj[step][1]) // 90 % 4)
                        step_node[l].append(int(eps_traj[step][0]))
                        step_goal_state[l].append(goals[eps][sub_reaching[0]][e_idx])
                        # extract the action that the agent takes
                        mov = eps_traj[step + 1] - eps_traj[step]
                        if (mov[0] != 0 and mov[1] != 0) or (mov[0] == 0 and np.abs(mov[1]) == 180):
                            # the agent must reach the goal and get reset at step+1
                            mov = goal_state[l][-1] - eps_traj[step]

                            if mov[0] != 0 and mov[1] == 0:
                                # a forward action, represented by 0
                                reach_action[l].append(0)
                            elif mov[0] == 0 and mov[1] != 0:
                                reach_action[l].append((mov[1] // 90) % 4)  # either 1 or 3
                        elif mov[0] == 0 and mov[1] == 0:
                            # the agent bumps to the wall, must have moved forward
                            reach_action[l].append(0)
                        else:
                            if mov[0] != 0: reach_action[l].append(0)
                            if mov[1] != 0: reach_action[l].append((mov[1] // 90) % 4)

    def cat_data(data, N):
        d = []
        for item in data:
            d.extend(item[:N])
        return np.array(d)

    # for each of the environments, we take out N data points and perform pca
    N = 2000
    step_env_idx = []
    for l in range(len(envs_idx)): step_env_idx += [l] * N
    r_key = cat_data(r_key, N)
    r_value = cat_data(r_value, N)
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
            axs.set_zticks([])
        elif dim == 2:
            fig, axs = plt.subplots(figsize=(7, 7))
            sc = axs.scatter(embedding[:, 0], embedding[:, 1], c=coloring)
        axs.set_xticks([])
        axs.set_yticks([])
        return reducer, embedding, axs, sc
    method = 'pca'
    # dim reduction for reading keys, colored by environment idx
    reducer, embedding, axs, sc = plot_embedding(r_key, 3, step_env_idx, method=method)
    handles = sc.legend_elements(num=range(len(envs_idx)))[0]  # extract the handles from the existing scatter plot
    axs.legend(title='maze', handles=handles, labels=list(cage_labels[envs_idx]), loc="upper right")

#### For Fig. 3c, 3d, 3e and supplementary Fig. 5 ####
# visualize single unit activity map
draw3 = False
if draw3:
    if_store = False  # whether to store activity map
    schemes = ['single', 'double']  # double | single
    simus = [1,2,3,4,5,6]
    envs_idx = [0,1,2,3,4,5]  # envs where unit firing map will be plotted
    comp = [0, 5]  # compare between circle and square
    spatial_units = np.zeros((len(schemes), len(simus),
                              3))  # the three dimension: [# spatial units in env1, # spatial units in env2, # overlaps]
    rate_remapping = np.zeros(len(envs_idx))  # store the avg rate change
    for m, scheme in enumerate(schemes):
        save_dir = '%s_%s_ds%s_kvrelu_newFWM' % (scheme, act_space, ds_nonlinearity)
        fig1, axs1 = plt.subplots()
        sigmoid_pass = np.zeros(len(simus))
        num_active_units = 0
        firing_rate_std = []  # std of fr of each unit among all envs
        for n, simu in enumerate(simus):
            # we store the idx of every hd place unit and its center
            hd_p_units = {}
            for id in envs_idx:
                hd_p_units[id] = {}
            # load the data
            data = pickle.load(open(save_dir + '/data/test/fwm_cnn_%s_%s_%s.pkl' % (scheme, simu, stage), 'rb'))
            memory_traces = data['memory_traces']
            step_reward = np.squeeze(data['step_reward'])
            trajs = data['trajectories']
            orientations = {'east': (1, 2), 'north': (0, 1), 'west': (1, 0), 'south': (2, 1)}  # index 0, 1, 2 ,3
            ############################################################
            mem_items = ['r_key', 'r_value', 'w_key', 'w_value']
            target = 'r_key'
            time_window = 'back2goal'  # on_goal | back2goal
            vector_dim = args['k_size'] if not 'rnn' in target else args['hidden']
            for env_idx in envs_idx:
                # we collect the memory vector at each location and orientation
                unit_map = np.zeros((4, maze_size, maze_size, vector_dim))  # so 4 orientations, 5 by 5 grids
                s_occupancy = np.zeros((4, maze_size, maze_size))  # record the visiting frequency for each state
                #########################################################
                for eps in range(len(trajs)):
                    mem_item = np.squeeze(memory_traces[eps][target])[:, env_idx]
                    # extract the trajectory
                    eps_traj = []
                    for tr in trajs[eps][env_idx]:
                        eps_traj.extend(tr)
                    eps_node, eps_ori = [], []
                    for x in eps_traj:
                        eps_node.append(x[0])
                        eps_ori.append((x[1] // 90) % 4)
                    eps_traj = np.array(eps_traj)
                    # extract time steps where the goal is found for the first time
                    goal_reaching = np.where(step_reward[eps][:, env_idx] > 0)[0]
                    for i in range(len(sub_eps_bounds) - 1):
                        # extract the goal reaching for the subepisode
                        x = goal_reaching > sub_eps_bounds[i]
                        y = goal_reaching < sub_eps_bounds[i + 1]
                        sub_reaching = goal_reaching[np.where(x * y)[0]]
                        # if a goal is reached more than 5 times, the agent remembers it
                        if len(sub_reaching) > 5:
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
                # compute the average activities at each state
                s_occupancy = np.expand_dims(s_occupancy + 1e-4, axis=-1)
                unit_map /= np.tile(s_occupancy, (1, vector_dim))
                # normlize the unit activity map
                unit_map /= (np.max(np.abs(unit_map)) + 1e-4)
                # compute the max std of the activity map for each unit
                shape = unit_map.shape
                # analyze the distributions for place cells and view modulated cells
                f, ids, pd, pc = modulated(unit_map)
                hd_p_units[env_idx]['ids'] = ids
                hd_p_units[env_idx]['centers'] = pc
                hd_p_units[env_idx]['unit_map'] = unit_map

                if if_store:
                    # folder to store images
                    image_folder = save_dir + '/results/activity_map/simu%s_%s_%s/%s/modulated' % (
                        simu, target, time_window, test_cages[env_idx])
                    if not os.path.exists(image_folder):
                        os.makedirs(image_folder)
                    # iterate over units
                    for unit in range(vector_dim):
                        fig = plt.figure(figsize=(12, 10))
                        axE = fig.add_axes([0.6, 0.4, 0.2, 0.2])
                        axN = fig.add_axes([0.4, 0.6, 0.2, 0.2])
                        axW = fig.add_axes([0.2, 0.4, 0.2, 0.2])
                        axS = fig.add_axes([0.4, 0.2, 0.2, 0.2])
                        axes = [axE, axN, axW, axS]
                        for i, ax in enumerate(axes):
                            im = ax.imshow(unit_map[i, :, :, unit], cmap='jet', vmin=vmin, vmax=1,
                                           origin='lower')
                            ax.set_xticks([])
                            ax.set_yticks([])
                        cb_ax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
                        cbar = fig.colorbar(im, cax=cb_ax)
                        cbar.set_ticks(np.arange(vmin, 1.1, 0.25))
                        # store the image
                        if unit not in ids:
                            plt.savefig(image_folder + '/../unit%s' % unit)
                        else:
                            plt.savefig(image_folder + '/unit%s' % unit)
                        plt.close(fig)

            # compare between the place fields between two mazes
            # check units overlap
            same_units = set(hd_p_units[comp[0]]['ids']).intersection(set(hd_p_units[comp[1]]['ids']))
            # print(scheme, simu)
            # print('overlapped', same_units)
            # print(cage_labels[comp[0]], set(hd_p_units[comp[0]]['ids']) - same_units)
            # print(cage_labels[comp[1]], set(hd_p_units[comp[1]]['ids']) - same_units)
            same_units = list(same_units)
            spatial_units[m][n][0] = len(hd_p_units[comp[0]]['ids'])
            spatial_units[m][n][1] = len(hd_p_units[comp[1]]['ids'])
            spatial_units[m][n][2] = len(same_units)

            # with the overlapped units in both envs, check the rate change
            delta = np.abs((hd_p_units[comp[0]]['unit_map'][:, :, :, same_units] - hd_p_units[comp[1]]['unit_map'][:, :,
                                                                                   :, same_units]))
            delta /= (hd_p_units[comp[0]]['unit_map'][:, :, :, same_units] + 1e-4)
            rate_remapping[m] += delta.mean()

            # examine the rate change of the place units from the circle maze to the square maze
            active_units = list(set(hd_p_units[0]['ids']).union(
                set(hd_p_units[5]['ids'])))  # we take all units that have some spatial response
            normalized_fr = np.zeros((len(envs_idx), len(active_units)))
            for env_idx in envs_idx:
                normalized_fr[env_idx] = hd_p_units[env_idx]['unit_map'][:, :, :, active_units].mean(axis=(0, 1, 2))
            normalized_fr = normalized_fr.T  # dim: [num_units, num_envs]
            normalized_fr /= np.tile([np.max(normalized_fr, axis=1)], (len(envs_idx), 1)).T


            # visualize
            def sigmoid(x, x0, k):
                y = 1 / (1 + np.exp(k * (x - x0)))
                return (y)


            xdata = np.arange(1, len(envs_idx) + 1)

            for u in range(len(active_units)):
                ydata = normalized_fr[u]
                try:
                    popt, pcov = curve_fit(sigmoid, xdata, ydata)
                    if np.max(pcov) < 10 and abs(popt[1]) > 0.4 and abs(ydata[0] - ydata[-1]) > 0.2:
                        sigmoid_pass[n] += 1
                        if popt[1] < 0:
                            axs1.plot(xdata, ydata, color='b', linewidth=1.5)
                        elif popt[1] >= 0:
                            axs1.plot(xdata, ydata[-1::-1], color='b', linewidth=1.5)
                except RuntimeError:
                    pass

            axs1.set_ylabel('Normalized fring rate')
            axs1.set_xlabel('Transition point')
            axs1.set_title(scheme)
            plt.tight_layout()
            num_active_units += len(active_units)
            firing_rate_std.extend(normalized_fr.std(axis=1))

        ## we also plot the histgram for the std of nor firing rate
        fig, axs2 = plt.subplots()
        axs2.hist(firing_rate_std, bins=20, density=True, edgecolor='black', color='blue')
        axs2.set_xlabel('firing rate std')
        axs2.set_ylabel('density (%)')
        axs2.set_title('%s' % scheme)
        plt.tight_layout()

        print(scheme, np.mean(sigmoid_pass), num_active_units / len(simus))

    print(spatial_units.mean(axis=1))
    # now we plot a bar plot
    spatial_units = np.mean(spatial_units, axis=1) / vector_dim  # avg over simulations
    w = 0.5  # bar width
    x = np.arange(2)
    for i in range(len(schemes)):
        fig2, axs2 = plt.subplots()
        axs2.bar(x, np.ones(len(comp)), color='b', width=w, label='silent units')  # bars for all units
        axs2.bar(x, spatial_units[i][:2], color='g', width=w, label='HD-place units')
        axs2.bar(x, [spatial_units[i][2], spatial_units[i][2]], color='k', width=w, label='overlaps')
        axs2.set_xticks(x)
        axs2.set_xticklabels(cage_labels[comp])
        axs2.set_xlim([-0.5, 1.5])
        if i == 0:
            axs2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
       # axs2.set_title('%s training scheme' % scheme)
    plt.tight_layout()

#### For Supplementary Fig. 4c ####
# we first visualize the learning curves
draw4 = False
if draw4:
    fig1, axs1 = plt.subplots()  # for lc
    schemes = ['single', 'double']
    colors = ['cornflowerblue', 'orange']
    for i, scheme in enumerate(schemes):
        save_dir = '%s_%s_ds%s_kvrelu_newFWM' % (scheme, act_space, ds_nonlinearity)
        avg_goal_reaching = []
        for simu in range(1, 8 + 1):
            data = pickle.load(
                open(save_dir + '/data/training/%s_%s_%s_%s.pkl' % (args['network'], args['process_im'], scheme, simu),
                     'rb'))
            rewards = np.array(data['step_reward'])
            # extract the number of goal reaching for each episode
            goal_reaching = np.mean(rewards > 0, axis=2)  # avg over environments
            goal_reaching = np.sum(goal_reaching, axis=1)  # sum over time steps
            avg_goal_reaching.append(goal_reaching)

        avg_goal_reaching = np.array(avg_goal_reaching)
        mu = flat_data(avg_goal_reaching.mean(axis=0))
        std = flat_data(avg_goal_reaching.std(axis=0))
        # plot hit rate per episode
        axs1.plot(range(1, len(mu) + 1), mu, linewidth=2, label=scheme, color=colors[i])
        axs1.fill_between(range(1, len(mu) + 1), mu - std, mu + std, alpha=0.2, color=colors[i])
    axs1.grid(True)
    axs1.legend()
    axs1.set_title('Learning curve')
    axs1.set_xlabel('trial')
    axs1.set_ylabel('Average # of goal reaching')
    plt.tight_layout()

#### For supplementary Fig. 4a, 4b ####
# we look at the performance during the test
draw5 = False
if draw5:
    N = 3  # we examine the # steps in the first N trial where the goal is found
    stage = args['training_episodes']
    simus = [1, 2, 3, 4,5,6]
    schemes = ['single', 'double']
    colors = ['cornflowerblue', 'orange']
    envs_idx = [0, 1, 2, 3, 4, 5]
    threshold = 30
    goal_change = np.arange(0 + goal_change_freq, episode_length + goal_change_freq, goal_change_freq)
    x_labels = ['trial %s' % x for x in range(1, N + 1)]
    fig, axs = plt.subplots()  # for plotting avg # of reach the goal
    for i, scheme in enumerate(schemes):
        save_dir = '%s_%s_ds%s_kvrelu_newFWM' % (scheme, act_space, ds_nonlinearity)
        goal_remembering = np.zeros((len(envs_idx), episode_length // goal_change_freq, N))
        remembered_goals = [[] for _ in range(len(envs_idx))]  # a goal that is found and remebered
        non_remembered_goals = [[] for _ in range(len(envs_idx))]  # a goal that is found but not remembered
        unfound_goals = [[] for _ in range(len(envs_idx))]  # a goal that is not found
        total_goals = [0 for _ in range(len(envs_idx))]  # all sampled goals
        m = np.zeros((len(envs_idx),
                      episode_length // goal_change_freq))  # recored the number of goals which are found at least N times
        for simu in range(1, 4 + 1):
            data = pickle.load(
                open(save_dir + '/data/test/fwm_cnn_%s_%s_%s.pkl' % (scheme, simu, stage), 'rb'))
            step_reward = np.array(data['step_reward'])
            goals = np.array(data['goal_location'])
            # now we divide each episode into several mini-episode
            mini_epi = [0] + list(goal_change)
            # for each episode j
            for j, item in enumerate(step_reward):
                for e_idx in envs_idx:
                    total_goals[e_idx] += episode_length // goal_change_freq
                    # for each mini-episode k
                    for k in range(1, len(mini_epi)):
                        goal_reaching = np.zeros(N)
                        goal = goals[j][mini_epi[k - 1]][e_idx]
                        found_goal = np.where(item[mini_epi[k - 1]:mini_epi[k], e_idx] > 0)[0]
                        # unfound goal
                        if len(found_goal) == 0:
                            unfound_goals[e_idx].append(goal)
                        elif len(found_goal) == 1 or len(found_goal) == 2:
                            non_remembered_goals[e_idx].append(goal)
                        elif len(found_goal) >= N:
                            m[e_idx][k - 1] += 1
                            goal_reaching[0] = found_goal[0]
                            for n in range(1, N):
                                goal_reaching[n] = found_goal[n] - found_goal[n - 1]
                            goal_remembering[e_idx][k - 1] += goal_reaching[:N]
                            # if the second and third times of finding the goal is less than a threshold,
                            # then we say this goal is remembered. The threshold is determined by the size and
                            # structure of the env.
                            if goal_reaching[1] < threshold and goal_reaching[2] < threshold:
                                remembered_goals[e_idx].append(goal)
                            else:
                                non_remembered_goals[e_idx].append(goal)
        # take average
        goal_remembering /= np.tile(np.expand_dims(m, axis=-1), (1, N))
        goal_remembering = goal_remembering.mean(axis=1)
        # plot bar chart
        w = 0.12  # width of the bar
        xs = np.arange(1, len(envs_idx) + 1)
        patterns = ['', '/', '-']
        # for each trial
        reach_labels = ['1st reach', '2nd reach', '3rd reach']
        for reach in range(N):
            axs.bar(xs + reach * w + i * 3 * w, height=goal_remembering[:, reach], edgecolor='black', color=colors[i],
                    width=w, label=reach_labels[reach], hatch=patterns[reach])
        axs.set_ylabel('# steps to find the goal')
        axs.set_xticks(xs + 3 * w)
        axs.set_xticklabels(cage_labels)
        axs.legend(fontsize=31, bbox_to_anchor=(1.05, 0.5))
        axs.set_title('%s' % scheme)

        goals_table = np.zeros((len(envs_idx), 3))
        for e_idx in envs_idx:
            goals_table[e_idx][0] = len(remembered_goals[e_idx]) / total_goals[e_idx]
            goals_table[e_idx][1] = len(non_remembered_goals[e_idx]) / total_goals[e_idx]
            goals_table[e_idx][2] = len(unfound_goals[e_idx]) / total_goals[e_idx]
        goals_table = pd.DataFrame(goals_table.T, columns=cage_labels,
                                   index=['remembered', 'not\n remembered', 'unfound'])
        goals_table = goals_table.round(2)
        fig1, axs1 = plt.subplots(figsize=(12, 6))
        Table(goals_table, textprops={'fontsize': 25, 'ha': 'center'},
              column_definitions=[ColDef(name='index', width=2.5)])
        axs1.set_title(scheme)
        plt.tight_layout()

#### For supplementary Fig. 4d ####
draw6 = False
if draw6:
    schemes = ['single', 'double']
    colors = ['blue', 'orange']
    colormaps = ['Blues', 'Oranges']
    simus = [1,2,3,4,5,6,7,8]
    stage = args['training_episodes']
    envs_idx = [0, 1, 2, 3, 4, 5]
    for i, scheme in enumerate(schemes):
        save_dir = '%s_%s_ds%s_kvrelu_newFWM' % (scheme, act_space, ds_nonlinearity)
        visited_nodes = [np.zeros((maze_size, maze_size)) for _ in range(len(envs_idx))]
        for simu in simus:
            # load the data
            data = pickle.load(
                open(save_dir + '/data/test/%s_%s_%s_%s_%s.pkl' % (
                args['network'], args['process_im'], scheme, simu, stage),
                     'rb'))
            # find trajectory
            trajs = data['trajectories']
            for l, e_idx in enumerate(envs_idx):
                for eps in range(len(trajs)):
                    # extract the trajectory
                    for tr in trajs[eps][e_idx]:
                        for x in tr: visited_nodes[e_idx][x[0] % maze_size][x[0] // maze_size] += 1

        for e_idx in envs_idx:
            visited_nodes[e_idx] /= np.sum(visited_nodes[e_idx])
        for e_idx in envs_idx:
            fig, axs = plt.subplots(figsize=(8.5, 6.5))
            im = axs.imshow(visited_nodes[e_idx], vmin=0.01, vmax=0.07,
                            cmap=colormaps[i], origin='lower')
            axs.set_xticks([])
            axs.set_yticks([])
            cb_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
            cbar = fig.colorbar(im, cax=cb_ax)
            cbar.set_ticks(np.arange(0.01, 0.072, 0.02))
            axs.set_title(cage_labels[e_idx])
            #plt.tight_layout()


plt.show()
