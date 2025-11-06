import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import scipy.ndimage as nd
from scipy.spatial.distance import pdist

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
    state_idx = node * 4 + np.where(orientations == ori)[0][0]
    return state_idx
# define plotting functions
def plot_1D(values, figsize=(5, 5)):
    fig, axs = plt.subplots(figsize=figsize)
    axs.plot(values, color='blue', zorder=2)
    # indicating goal change and reach
    axs.vlines(np.arange(goal_change_freq, episode_length, goal_change_freq), 0, np.max(values), linewidth=1.8,
               color='black', zorder=1)
    axs.set_ylabel('value')
    axs.set_xlabel('time step')
    plt.tight_layout()
    return fig, axs
def plot_2D(values, figsize=(5, 5), axs=None, fig=None):
    if axs is None:
        fig, axs = plt.subplots(figsize=figsize)
    im = axs.imshow(values.T, cmap='viridis', aspect='auto')
    # indicating goal change and reach
    axs.vlines(np.arange(goal_change_freq, episode_length, goal_change_freq), values.shape[1], values.shape[1] + 30,
               linewidth=1.8,
               color='black')
    axs.set_ylabel('dimension')
    axs.set_xlabel('time step')
    axs.set_ylim([values.shape[1] + 15, 0])
    # axs.set_xlim([20, 100])
    cbar = fig.colorbar(im, ax=axs)
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



def calculate_pdist(fields_sliced, place_indices):
    """ Calculate pairwise distances between field centers for specified fields.

    Keyword arguments :
    fields_sliced : fields for which the pairwise distances must be calculated
    place_indices : (legacy, remove)
    """
    centers = np.zeros((place_indices.size, 4, 2))
    for f, i in zip(fields_sliced, range(len(fields_sliced))):
        for angle, j in zip(f, range(len(f))):
            centers[i, j, :] = divmod((np.argmax(angle)), 5) # store the x, y coordinate of the center
    pdist_centers = [pdist(centers[i]) for i in range(len(centers))]
    return pdist_centers, centers

def classify_place_field(activity_map):
    """ Decide whether or not a given spatial activity map contains place fields.

    Keyword arguments :
    activity_map : 2d numpy array of spatial activations

    """
    maze_size = activity_map.shape[0]
    is_field = True
    # find clusters where activity falls off to 15% and calculate a boolean mask of clusters
    # we iterate from small guassian filters to bigger ones
    sigs = [0.5, 1, 2, 3, 4]
    sigmas = []
    for i in range(len(sigs)):
        for j in range(i, len(sigs)):
            if sigs[i] == 4 and sigs[j] == 4:
                pass
            else: sigmas.append([sigs[i], sigs[j]])
    activity_map = np.where(activity_map < 0.2, 0, activity_map)
    for sigma in sigmas:
        activity_map = nd.gaussian_filter(activity_map, sigma=sigma)
        mask = activity_map > 0
        # label clusters
        labels, nb = nd.label(mask)

        if nb != 0:
            # find largest cluster
            sizes = []
            for i in range(nb + 1):
                slice_id = nd.find_objects(labels == i)
                # sizes.append(np.shape(activity_map[slice_id[0]]))
                if not slice_id:
                    sizes.append((maze_size, maze_size))
                else:
                    sizes.append(np.shape(activity_map[slice_id[0]]))
            pixels = np.prod(sizes, axis=1)
            l = np.argmax(pixels[1:]) + 1

            field = activity_map

        else:
            field = labels
        # is it a field?
        if nb == 0:
            is_field = False
            #print(sigma, "no cluster -- not place like")
        elif nb > 2:
            is_field = False
            #print(sigma, "too many -- not place like")
        elif pixels[l] > 1.5 * pixels[0]:
            non_zero = np.count_nonzero(field)
            fraction_active = non_zero / (maze_size**2)
            is_field = False
            #print(sigma, "too large, not place like : ", fraction_active)
        else:
            is_field = True
            #print(sigma, 'Find place field')
        if is_field:
            return is_field, field
    return is_field, field

def place_like(fields):
    fields = np.squeeze(fields)
    n_hds, n_units = fields.shape[0], fields.shape[3]
    maze_size = fields.shape[1]

    fields = fields.reshape((n_hds, maze_size**2, n_units))
    mean_fields = np.mean(fields, axis=0).reshape((1, maze_size**2, n_units))
    stacked = np.vstack((fields, mean_fields))

    is_field = np.zeros((n_hds + 1, n_units))
    largest_field = np.zeros((n_hds + 1, maze_size**2, n_units))
    #print("Place")
    for f, i in zip(np.rollaxis(stacked, 2), range(n_units)):
        #print("Unit", i)
        for hd, j in zip(f, range(n_hds + 1)):
            #print("Head direction", j)
            hd = hd.reshape(maze_size, maze_size)
            isit, big = classify_place_field(hd)
            is_field[j, i] = isit
            largest_field[j, :, i] = big.flatten()

    # if all fields AND mean are classified, it may be a place like representation
    a = np.all(is_field, axis=0)
    ids = np.atleast_1d(np.squeeze(np.array(np.where(a))))  # ids where fields are place like
    if ids.shape != ():
        f = np.array([largest_field[:, :, p] for p in ids])
    else:
        f = np.array([largest_field[:, :, ids]])

    if ids.size > 0:
        pd, pc = calculate_pdist(f[:, :n_hds, :], ids) # return place fields distance and centers
    else:
        #print("no slice found--", ids)
        pd = [15 * np.ones((15,))]
        pc = [15 * np.ones((15,))]
    # if there is too much directional modulation, not field
    x_all = np.all(np.array(pd) < 8, axis=1)
    ids = np.where(x_all, ids, -1)
    ids = ids[ids != -1]
    f = np.array([largest_field[:, :, p] for p in ids])
    if ids.size > 0:
        pd, pc = calculate_pdist(f[:, :n_hds, :], ids)
    else:
        #print("no slice found--", ids)
        pd = [15 * np.ones((15,))]

    return f, ids, pd, pc

def modulated(fields):
    fields = np.squeeze(fields)
    n_hds, n_units = fields.shape[0], fields.shape[3]
    maze_size = fields.shape[1]

    fields = fields.reshape((n_hds, maze_size**2, n_units))
    is_field = np.zeros((n_hds, n_units))
    largest_field = np.zeros((n_hds, maze_size**2, n_units))

    for f, i in zip(np.rollaxis(fields, 2), range(n_units)):
        #print("Unit", i)
        for hd, j in zip(f, range(n_hds)):
            #print("Head direction", j)
            hd = hd.reshape(maze_size, maze_size)
            isit, big = classify_place_field(hd)
            is_field[j, i] = isit
            largest_field[j, :, i] = big.flatten()

    a = np.all(is_field, axis=0)
    ids = np.atleast_1d(np.squeeze(np.array(np.where(a))))
    if ids.shape != ():
        f = np.array([largest_field[:, :, p] for p in ids])
    else:
        f = np.array([largest_field[:, :, ids]])
    if ids.size > 0:
        pd, pc = calculate_pdist(f[:, :n_hds, :], ids)
    else:
        pd = [np.zeros((15,))]
        pc = [np.zeros((15,))]
    x_any = np.any(np.array(pd) < 8, axis=1)
    ids = np.where(x_any, ids, -1)
    ids = ids[ids != -1]

    f = np.array([largest_field[:, :, p] for p in ids])
    if ids.size > 0:
        pd, pc = calculate_pdist(f[:, :n_hds, :], ids)
    else:
        #print("no slice found--", ids)
        pd = [np.zeros((15,))]
    return f, ids, pd, pc