'''
This script generates all the figures or subpanels used for the auto-association task.
Please turn any of the "draw*" to "True" if you want to plot the corresponding figure.
'''
import os
import pickle
import numpy as np
import torch
from collections import Counter
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import matplotlib.pyplot as plt
import matplotlib as mat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm
from sklearn.metrics.pairwise import cosine_similarity
from recall_train import CnnFWM, plot_predict_results
device = torch.device('cuda:0')
torch.cuda.empty_cache()
## set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 30}
mat.rc('font', **font)
# define plotting functions
def plot_1D(values, figsize=(5, 5)):
    fig, axs = plt.subplots(figsize=figsize)
    axs.plot(values, color='blue', linewidth = 2.5, marker='o', ms=8, mec='black')
    # indicating goal change and reach
    axs.set_ylabel('value')
    axs.set_xlabel('time step')
    plt.tight_layout()
    return fig, axs
def plot_2D(values,  figsize=(5, 5), axs=None, fig=None):
    if axs is None:
        fig, axs = plt.subplots(figsize=figsize)
    im = axs.imshow(values.T, cmap='viridis', aspect='auto')
    # indicating goal change and reach
    axs.set_ylabel('dimension')
    axs.set_xlabel('time step')
    axs.set_ylim([values.shape[1] + 5, 0])
    cbar = fig.colorbar(im,ax=axs)
    plt.tight_layout()
    return fig, axs
def plot_embedding(data, dim, coloring=[], embedding=None, method='pca'):
    components = None
    if embedding is None:
        if method == 'umap':
            reducer = umap.UMAP(n_components=dim)
            embedding = reducer.fit_transform(data)
        elif method == 'pca':
            pca = PCA(n_components=dim).fit(data)
            components = pca.components_
            # print(pca.explained_variance_ratio_)
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
    #axs.set_xticks([])
    #axs.set_yticks([])
    return components, embedding, axs, sc
def reverse_per(X):
    idx = list(range(len(X)))
    return [y for _, y in sorted(zip(X, idx))]
def flat_data(data, N=50):
    flattened_data = []
    for i in range(len(data) - N):
        flattened_data.append(np.mean(data[i:i + N]))
    return np.asarray(flattened_data)
# we specify the experiment that we want to plot
save_dir = 'data_test'
img_dim = [28, 28]
activation = 'relu'

###### For Fig 2a, 2b #######
## The script below plots the writing strength, the reconstructed writing and reading values
draw1 = False
if draw1:
    simu = 1   # specify which simulation to plot
    data = pickle.load(open(save_dir + '/recall_cnn_fwm_%s_kv%s.pkl' % (simu, activation), 'rb'))
    memory_traces = data['memory_traces']
    num_test = len(memory_traces)
    seq_len = 0
    while seq_len != 10:
        episode = np.random.choice(range(num_test), 1)[0]
        item_type = 'key'  # key | value
        w_beta = np.squeeze(memory_traces[episode]['w_beta'])
        w_item = np.squeeze(memory_traces[episode]['w_%s' % item_type])
        # w_item *= np.tile([w_beta], (w_item.shape[1], 1)).T
        r_item = np.squeeze(memory_traces[episode]['r_%s' % item_type])
        per_idx = np.squeeze(data['per_idx'][episode])
        seq_len = len(w_item) // 2
    # plot writing strength
    fig, axs = plot_1D(w_beta, (14, 1))
    axs.set_xticks([1, 6, 11, 16, 21])
    axs.set_xlabel('trial')
    axs.set_xlim([-0.2+seq_len+1, seq_len-1+0.5+seq_len+1])
    axs.set_ylim([-0.03, np.max(contents)*1.2])
    axs.vlines(seq_len, 0, np.max(contents), linewidth=1, color='red', linestyle='dashed', zorder=1)
    axs.set_title('%s, S%se%s' % (item, simu, episode), fontsize=12)

    # cos similarity between w key during memorization, and r_key during prediction
    fig, axs = plt.subplots()
    xticks_location = np.arange(seq_len)
    sim = cosine_similarity(w_item[:seq_len], r_item[seq_len+1:])
    #sim = np.matmul(w_item[seq_len+1:], r_item[seq_len+1:].T)
    im = axs.imshow(sim, cmap='viridis', aspect='auto')
    cbar = fig.colorbar(im, ax=axs)
    axs.set_ylabel('(w %s) time step (input index)'%item_type)
    axs.set_xlabel('(r %s) time step'%item_type)
    axs.set_xticks(xticks_location)
    axs.set_yticks(xticks_location)
    axs.set_yticklabels(xticks_location+1)
    axs.set_xticklabels(xticks_location+seq_len+1)
    axs1 = axs.twiny()
    axs1.set_xlim(axs.get_xbound()[0], axs.get_xbound()[1])
    axs1.set_xticks(xticks_location)
    axs1.set_xticklabels(xticks_location[per_idx])
    axs1.set_xlabel('(r %s) input index' % item_type)
    # reverse the index of the input in the prediction part back to the first phase
    fig, axs = plt.subplots()
    rever_idx = reverse_per(per_idx)
    sim = cosine_similarity(w_item[:seq_len], r_item[seq_len+1:][rever_idx])
    #sim = np.matmul(w_item[:seq_len], r_item[seq_len+1:][rever_idx].T)
    im = axs.imshow(sim.T, cmap='viridis', aspect='auto')
    axs.set_ylabel('(w %s) time step (input index)' % item_type)
    axs.set_xlabel('(r %s) time step' % item_type)
    axs.set_xticks(xticks_location)
    axs.set_yticks(xticks_location)
    axs.set_yticklabels(xticks_location+1)
    axs.set_xticklabels((xticks_location + seq_len+2)[rever_idx])
    axs1 = axs.twiny()
    axs1.set_xlim(axs.get_xbound()[0], axs.get_xbound()[1])
    axs1.set_xticks(xticks_location)
    axs1.set_xticklabels((xticks_location[per_idx]+1)[rever_idx])
    axs1.set_xlabel('(r %s) input index' % item_type)
    cbar = fig.colorbar(im, ax=axs)

    # define a decoder function
    ########### we load the original model and only use the DS decoder #########
    model_path = "models/recall_cnn_fwm_%s_kv%s.pt" % (simu, activation)
    model = CnnFWM(data['param'])
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.requires_grad_(False)
    def ds_decoder(x):
        T = x.shape[0]
        x = torch.cat((torch.ones(x.shape[0], 256).to(device) * 0.0, model.l1.linear(x)), -1)
        B = 1  # batch size is always 1 in test phase
        latent1 = model.mem_to_latent(x)
        # convert latent shape from (T,B,latent_dim) to (T*B, C_Hidden, H_hidden, W_hidden)
        latent1 = latent1.view(T * B, -1)
        # the height and width of the feature map passing to the first deconv layer
        H_hidden = int(img_dim[0] // 4)  # has been passed to two conv layers with stride 2
        latent1 = latent1.view(-1, 64, H_hidden, H_hidden)
        out = model.decoder(latent1)
        # convert the final output shape from (T*B=1,C=1,H,W) to (T,H,W)
        out = out.view(T, img_dim[0], img_dim[1])
        return out
    # extract r_value, w_value, new_w_value and old_w_value
    r_v = np.squeeze(memory_traces[episode]['r_value'])
    w_v = np.squeeze(memory_traces[episode]['w_value'])
    delta_w_v = np.squeeze(memory_traces[episode]['new_w_value'])
    w_beta = np.squeeze(memory_traces[episode]['w_beta'])
    old_w_v = w_v - delta_w_v
    #########################
    # task images
    input_1 = np.squeeze(data['input1'][episode])
    output_1 = np.squeeze(data['prediction1'][episode])
    input_2 = np.squeeze(data['input2'][episode])
    output_2 = np.squeeze(data['prediction2'][episode])
    target_2 = np.squeeze(data['target2'][episode])
    # plot_predict_results(input_1, input_2, target_2, output_1, output_2)
    # decoded r/w values
    plot_imgs = {}
    plot_imgs['input_1'] = input_1
    #decoded_w_v = -delta_w_v[:seq_len] * np.tile([w_beta[:seq_len]], (w_v.shape[1], 1)).T
    plot_imgs['decoded_delta_w_v_1'] = ds_decoder(3*torch.FloatTensor(delta_w_v[:seq_len]).to(device)).cpu().numpy()
    plot_imgs['input_2'] = input_2
    plot_imgs['output_2'] = output_2
    plot_imgs['target_2'] = target_2
    plot_imgs['decoded_r_v_2'] = ds_decoder(torch.FloatTensor(4.5*r_v[seq_len + 1:]).to(device)).cpu().numpy()

    # plot all results
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    subfigs = fig.subfigures(nrows=len(list(plot_imgs.keys())), ncols=1)
    for row, (subfig, item_key) in enumerate(zip(subfigs, list(plot_imgs.keys()))):
        axs = subfig.subplots(nrows=1, ncols=seq_len)
        for col, ax in enumerate(axs):
            ax.imshow(plot_imgs[item_key][col])
            if '1' in item_key:
                step = col + 1
            else:
                step = col + 1 + seq_len + 1
            #ax.set_xlabel('trial %s' % step, fontsize=22)
            ax.tick_params(axis="both", which="both", length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

#### For Fig 2c; the PCA projection parts in Supplementary Fig 1 ####
draw2 = False
if draw2:
    # load the data
    simu = 1
    data = pickle.load(open(save_dir + '/recall_cnn_fwm_%s_kv%s.pkl' % (simu, activation), 'rb'))
    memory_traces = data['memory_traces']
    num_test = len(memory_traces)
    digit_labels = data['input_label']
    per_idx = data['per_idx']
    # prepare data dictionary
    data_to_vis = {'w_key_1': [], 'w_value_1': [], 'r_key_1': [], 'r_value_1': [],
                   'r_key_2': [], 'r_value_2': [], 'w_key_2': [], 'w_value_2': [],
                   'original': [], 'cue': []}
    digit_1, digit_2 = [], []
    ss= []
    for episode in range(num_test):
        seq_len = len(digit_labels[episode])
        ss.append(seq_len)
        data_to_vis['w_key_1'].extend(np.squeeze(memory_traces[episode]['w_key'][:seq_len]))
        data_to_vis['w_value_1'].extend(np.squeeze(memory_traces[episode]['new_w_value'][:seq_len]))
        data_to_vis['r_key_1'].extend(np.squeeze(memory_traces[episode]['r_key'][:seq_len]))
        data_to_vis['r_value_1'].extend(np.squeeze(memory_traces[episode]['r_value'][:seq_len]))
        data_to_vis['w_key_2'].extend(np.squeeze(memory_traces[episode]['w_key'][seq_len+1:]))
        data_to_vis['w_value_2'].extend(np.squeeze(memory_traces[episode]['new_w_value'][seq_len+1:]))
        data_to_vis['r_key_2'].extend(np.squeeze(memory_traces[episode]['r_key'][seq_len+1:]))
        data_to_vis['r_value_2'].extend(np.squeeze(memory_traces[episode]['r_value'][seq_len+1:]))
        data_to_vis['original'].extend(np.squeeze(data['input1'][episode]))
        data_to_vis['cue'].extend(np.squeeze(data['input2'][episode]))
        digit_1.extend(digit_labels[episode])
        digit_2.extend(digit_labels[episode][per_idx[episode]])
    ### calculate the distances between original images and corresponding mem items
    # randomly select N images, keys or values
    targets = [['w_key_1','w_key_1'], ['r_key_2','r_key_2'], ['w_key_1','r_key_2'],
               ['w_value_1','w_value_1'], ['r_value_2','r_value_2'], ['w_value_1','r_value_2']]
    #labels = ['w keys', 'w values', 'r keys', 'r values']
    s_size = 20  # sample size
    n_sample = 500
    num_class = 18
    for target in targets:
        # extract indice for different digits
        digit_indice = [{}, {}]
        image_pool = []
        for n in range(2):
            for digit in range(num_class):
                if '1' in target[n]:
                    digit_indice[n][digit] = np.where(np.array(digit_1) == digit)[0]
                    image_pool.append(data_to_vis['original'])
                elif '2' in target[n]:
                    digit_indice[n][digit] = np.where(np.array(digit_2) == digit)[0]
                    image_pool.append(data_to_vis['cue'])
        same_distances, diff_distances = np.zeros((n_sample, 2)), np.zeros((n_sample, 2))
        for t in range(n_sample):
            s_digit_1 = np.random.randint(0, num_class)
            s_digit_2 = s_digit_1
            while s_digit_2 == s_digit_1: s_digit_2 = np.random.randint(0, num_class)

            indice_same_1 = np.random.choice(digit_indice[0][s_digit_1], size=s_size)
            indice_same_2 = np.random.choice(digit_indice[1][s_digit_1], size=s_size)
            indice_diff_2 = np.random.choice(digit_indice[1][s_digit_2], size=s_size)
            images_same_1, mem_same_1 = np.array(image_pool[0])[indice_same_1], np.array(data_to_vis[target[0]])[indice_same_1]
            images_same_2, mem_same_2 = np.array(image_pool[1])[indice_same_2], np.array(data_to_vis[target[1]])[indice_same_2]
            images_diff_2, mem_diff_2 = np.array(image_pool[1])[indice_diff_2], np.array(data_to_vis[target[1]])[indice_diff_2]

            images_same_1 = np.reshape(images_same_1, (s_size, -1))
            images_same_2 = np.reshape(images_same_2, (s_size, -1))
            images_diff_2 = np.reshape(images_diff_2, (s_size, -1))
            # for within digit class distances
            same_distances[t][0] = 1 - np.sum(cosine_similarity(images_same_1, images_same_2)*(1-np.eye(s_size)))/(s_size**2-s_size)
            same_distances[t][1] = 1 - np.sum(cosine_similarity(mem_same_1, mem_same_2)*(1-np.eye(s_size)))/(s_size**2-s_size)
            # for different digit classes distances
            diff_distances[t][0] = 1 - np.sum(cosine_similarity(images_same_1, images_diff_2)*(1-np.eye(s_size)))/(s_size**2-s_size)
            diff_distances[t][1] = 1 - np.sum(cosine_similarity(mem_same_1, mem_diff_2)*(1-np.eye(s_size)))/(s_size**2-s_size)
        fig, axs = plt.subplots(figsize=(4, 5))
        axs.scatter(same_distances[:, 0], same_distances[:, 1], color = 'blue', label='within class')
        b, a = np.polyfit(same_distances[:, 0], same_distances[:, 1], deg=1)
        xseq = np.linspace(min(same_distances[:, 0]), max(same_distances[:, 0]), num=100)
        #axs.plot(xseq, a + b * xseq, color="k", lw=4, zorder=2)
        axs.scatter(diff_distances[:, 0], diff_distances[:, 1], color='red' , label='between class')
        b, a = np.polyfit(diff_distances[:, 0], diff_distances[:, 1], deg=1)
        xseq = np.linspace(min(diff_distances[:, 0]), max(diff_distances[:, 0]), num=100)
        axs.set_xlim([0.3, 0.9])
        axs.set_ylim([0.0, 1.1])
        axs.set_title('%s vs. %s'%(target[0], target[1]), fontsize=14)
        plt.tight_layout()

    # plot projections for each
    method = 'lda'
    for item_key in ['w_key_1', 'w_value_1', 'r_key_2', 'r_value_2']:
        if '1' in item_key:
            coloring = digit_1
            stage = 1
        elif '2' in item_key:
            coloring = digit_2
            stage = 2
        components, embedding, axs, sc = plot_embedding(data_to_vis[item_key], 2, coloring, method=method)
        handles = sc.legend_elements(num=range(0, num_class))[0]  # extract the handles from the existing scatter plot
        #axs.legend(title='Digit', handles=handles, labels=range(0, 10), loc="upper right")
        axs.set_xticks([])
        axs.set_yticks([])
        #plt.title('%s in stage %s'%(item_key[:-2], stage))

    # concatenate w_value_1, r_value_2, colored by digits
    method = 'pca'
    concatenation = np.concatenate((data_to_vis['w_key_1'], data_to_vis['r_key_2']))
    coloring = np.concatenate((digit_1, digit_2))
    components, embedding, axs, sc = plot_embedding(concatenation, 3, coloring, method=method)
    handles = sc.legend_elements(num=range(0, num_class))[0]  # extract the handles from the existing scatter plot
    #axs.legend(title='Digit', handles=handles, labels=range(0, 10), loc="upper right")
    # plt.title('2D pca of w_key, r_key in stage 1, and r_key in stage 2, colored by digits')
    # concatenate w_value_1, and r_value_2, colored by type
    num = len(data_to_vis['w_key_1'])
    coloring = np.concatenate((np.ones(num) * 1, np.ones(num) * 3))
    components, embedding, axs, sc = plot_embedding(concatenation, 3, coloring, method=method, embedding=embedding)
    handles = sc.legend_elements(num=[1, 3])[0]  # extract the handles from the existing scatter plot
    axs.legend(title='', handles=handles, labels=['w_key in stage 1', 'r_key in stage 2'], loc="upper right")

### For Fig. 2d, 2e ####
draw3 = False
if draw3:
    # load the data
    simus = [1]
    class_rep = []  # check if all classes are represented
    if_store = True  # whether to store unit activity images
    for simu in simus:
        data = pickle.load(open(save_dir + '/recall_cnn_fwm_%s_kv%s.pkl' % (simu, activation), 'rb'))
        memory_traces = data['memory_traces']
        num_test = len(memory_traces)
        digit_labels = data['input_label']
        per_idx = data['per_idx']
        # prepare data dictionary
        data_to_vis = {'w_key_1': [], 'w_value_1': [], 'r_key_1': [], 'r_value_1': [],
                       'r_key_2': [], 'r_value_2': [], 'w_key_2': [], 'w_value_2': [],
                       'original': [], 'cue': []}
        digit_1, digit_2 = [], []
        for episode in range(num_test):
            seq_len = len(digit_labels[episode])
            data_to_vis['w_key_1'].extend(np.squeeze(memory_traces[episode]['w_key'][:seq_len]))
            data_to_vis['w_value_1'].extend(np.squeeze(memory_traces[episode]['new_w_value'][:seq_len]))
            data_to_vis['r_key_2'].extend(np.squeeze(memory_traces[episode]['r_key'][seq_len+1:]))
            data_to_vis['r_value_2'].extend(np.squeeze(memory_traces[episode]['r_value'][seq_len+1:]))
            data_to_vis['original'].extend(np.squeeze(data['input1'][episode]))
            data_to_vis['cue'].extend(np.squeeze(data['input2'][episode]))
            digit_1.extend(digit_labels[episode])
            digit_2.extend(digit_labels[episode][per_idx[episode]])

        # compute the average vector for each digit class
        target = 'r_key_2'
        target_item = np.array(data_to_vis[target])
        if '1' in target: digits = digit_1
        else:  digits = digit_2
        digits = np.array(digits)
        avg_reaction = np.zeros((18, target_item.shape[1]))
        for dig in range(18):
            idx = np.where(digits == dig)[0]
            avg_reaction[dig] = np.mean(target_item[idx], axis=0)
        ## store the bar plot of the avg reaction for each unit
        image_folder = 'activity/simu%s_%s_%s' % (simu, target, activation)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        if if_store:
            for unit in range(target_item.shape[1]):
                fig, axs = plt.subplots()
                x = np.arange(1, 19)
                axs.bar(x, avg_reaction[:, unit])
                # axs.set_ylim([0, 1.0])
                axs.set_ylabel('Firing rate')
                axs.set_xlabel('class index')
                axs.set_xticks([1, 6, 12 ,18])
                axs.set_yticks([0, 0.5, 1.0])
                plt.tight_layout()
                plt.savefig(image_folder + '/unit%s' % unit)
                plt.close(fig)
                #plt.show()
        class_field = {}  # we store the pairing between a unit and its reacted class idx
        for unit in range(target_item.shape[1]):
            if np.max(avg_reaction[:, unit]) > 0.1:
                class_rep.append(np.argmax(avg_reaction[:, unit])+1) # +1 just for visualization
                class_field[unit] = np.argmax(avg_reaction[:, unit])

    fig, axs = plt.subplots()
    axs.hist(class_rep, bins=18, edgecolor = 'k', facecolor = 'b')
    axs.set_ylabel('# of units (out of 128)')
    axs.set_xlabel('Class index (1 ~ 18)')
    axs.set_xticks([5.3, 10, 14.7])
    axs.set_xticklabels([5, 10, 15])
    plt.tight_layout()

    # plot the firing rate of certain units to individual images
    unit = np.random.choice(list(class_field.keys()), 1)[0]
    print(unit)
    class_idx = class_field[unit]
    N = 6  # number of images to show
    index_fire = np.random.choice(np.where(digit_1==class_idx)[0], N)
    index_silent = np.random.choice(digit_1, N)
    index = np.concatenate((index_fire, index_silent))
    np.random.shuffle(index)
    reactions = target_item[index][:, unit]
    images = np.array(data_to_vis['original'])[index]
    fig1, axs1 = plt.subplots(1, N*2, figsize=(8, 2))
    for i in range(N*2):
        axs1[i].imshow(images[i])
        axs1[i].set_xticks([])
        axs1[i].set_yticks([])
        plt.setp(axs1[i].get_xticklabels(), visible=False)
        plt.setp(axs1[i].get_yticklabels(), visible=False)
    pos_left = axs1[0].get_position()
    pos_right = axs1[-1].get_position()
    axs2 = fig1.add_axes([pos_left.x0-pos_left.width/3, pos_left.y0+pos_left.height, pos_right.x0
                          - pos_left.x0+1.75*pos_right.width, 0.3])
    axs2.bar(np.arange(0, len(reactions)), height=reactions, width=0.2)
    plt.axis('off')

### The generalization experiment ###
### For Supplementary Fig 2e, f ###
draw4 = False
if draw4:
    simus = [2]
    num_labels = 36
    gener_error = np.zeros(num_labels)
    predicts, targets, classes = [], [], []
    for simu in simus:
        data = pickle.load(open(save_dir + '/recall_cnn_fwm_%s_kv%s_general.pkl' % (simu, activation), 'rb'))
        for eps in range(len(data['input1'])):
            predicts.extend(data['prediction2'][eps])
            targets.extend(data['target2'][eps])
            classes.extend(data['input_label'][eps][data['per_idx'][eps]])
    predicts = np.squeeze(predicts)
    targets = np.squeeze(targets)
    classes = np.squeeze(classes)
    for c in range(num_labels):
        idx = np.where(classes == c)[0]
        gener_error[c] = np.mean(np.linalg.norm(predicts[idx] - targets[idx], axis=(1, 2)))

    # plot bar graph
    fig, axs = plt.subplots(figsize=(15, 8))
    x = np.arange(0, num_labels)
    w = 0.35
    axs.bar(x[:18], height=gener_error[:18], edgecolor='k', facecolor='b', width=w)
    axs.bar(x[18:], height=gener_error[18:], edgecolor='k', facecolor='orange', width=w)
    axs.set_xlabel('Class index')
    axs.set_ylabel('mean error')

    num_labels = 18
    gener_error = np.zeros(num_labels)
    predicts, targets, classes = [], [], []
    for simu in simus:
        data = pickle.load(open(save_dir + '/recall_cnn_fwm_%s_kv%s.pkl' % (simu, activation), 'rb'))
        for eps in range(len(data['input1'])):
            predicts.extend(data['prediction2'][eps])
            targets.extend(data['target2'][eps])
            classes.extend(data['input_label'][eps][data['per_idx'][eps]])
    predicts = np.squeeze(predicts)
    targets = np.squeeze(targets)
    classes = np.squeeze(classes)
    for c in range(num_labels):
        idx = np.where(classes == c)[0]
        gener_error[c] = np.mean(np.linalg.norm(predicts[idx] - targets[idx], axis=(1, 2)))

    axs.bar(x[:18] - w, height=gener_error[:18], edgecolor='k', facecolor='gray', width=w)
    axs.set_xticks([0, 18, 35])
    axs.set_xticklabels([1, 19, 36])
    axs.set_ylim([0, 9.5])
    plt.tight_layout()

plt.show()