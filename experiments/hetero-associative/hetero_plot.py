'''
This script generates all the figures or subpanels used for the hetero-association task.
Please turn any of the "draw*" to "True" if you want to plot the corresponding figure.
'''

import os
import pickle
import json
import itertools
import numpy as np
import torch
import random
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import matplotlib.pyplot as plt
import matplotlib as mat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.cm
from sklearn.metrics.pairwise import cosine_similarity
from predict_train import CnnFWM, plot_predict_results

#plt.switch_backend('Qt5Agg')
device = torch.device('cuda:0')
torch.cuda.empty_cache()
## set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 38}
mat.rc('font', **font)

# define plotting functions
def plot_1D(values, figsize=(5, 5)):
    fig, axs = plt.subplots(figsize=figsize)
    axs.plot(values, color='blue', zorder=2)
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
            reducer = umap.UMAP(n_components=dim, n_neighbors=20, metric='cosine')
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
    axs.set_xticks([])
    axs.set_yticks([])
    return components, embedding, axs, sc
def reverse_per(X):
    idx = list(range(len(X)))
    return [y for _, y in sorted(zip(X, idx))]
# we specify the experiment that we want to plot
save_dir = 'data_test'
model_path = 'models'
activation = 'tanh'
img_dim = [28, 28]


### For Fig. 2f, 2g #####
draw1 = False
if draw1:
    # load the data
    simu = 1
    data = pickle.load(open(save_dir + '/predict_cnn_fwm_%s_kv%s.pkl' % (simu, activation), 'rb'))
    memory_traces = data['memory_traces']
    num_test = len(memory_traces)
    item_type = 'key'  # key | value
    seq_len = 0
    while seq_len != 10:
        episode = np.random.choice(range(num_test), 1)[0]
        w_item = np.squeeze(memory_traces[episode]['w_%s'%item_type])
        r_item = np.squeeze(memory_traces[episode]['r_%s'%item_type])
        per_idx = np.squeeze(data['per_idx'][episode])
        seq_len = len(w_item) // 2
    r_beta = np.squeeze(memory_traces[episode]['r_beta'])
    # cos similarity between w key during memorization, and r_key during prediction
    fig, axs = plt.subplots()
    xticks_location = np.arange(seq_len-1)
    cos_sim = cosine_similarity(w_item[0:seq_len-1], r_item[seq_len+1:])
    im = axs.imshow(cos_sim, cmap='viridis', aspect='auto')
    cbar = fig.colorbar(im, ax=axs)
    axs.set_ylabel('(w_%s) time step (input index)'%item_type)
    axs.set_xlabel('(r_%s) time step'%item_type)
    axs.set_xticks(xticks_location)
    axs.set_yticks(xticks_location)
    axs.set_yticklabels(xticks_location+1)
    axs.set_xticklabels(xticks_location+seq_len+1)
    axs1 = axs.twiny()
    axs1.set_xlim(axs.get_xbound()[0], axs.get_xbound()[1])
    axs1.set_xticks(xticks_location)
    axs1.set_xticklabels(xticks_location[per_idx]+1)
    axs1.set_xlabel('(r_%s) input index' % item_type)
    # reverse the index of the input in the prediction part back to the first phase
    fig, axs = plt.subplots()
    rever_idx = reverse_per(per_idx)
    cos_sim = cosine_similarity(w_item[0:seq_len-1], r_item[seq_len+1:][rever_idx])
    im = axs.imshow(cos_sim, cmap='viridis', aspect='auto')
    axs.set_ylabel('(w_%s) time step (input index)' % item_type)
    axs.set_xlabel('(r_%s) time step' % item_type)
    axs.set_xticks(xticks_location)
    axs.set_yticks(xticks_location)
    axs.set_yticklabels(xticks_location + 1)
    axs.set_xticklabels((xticks_location + seq_len+1)[rever_idx])
    axs1 = axs.twiny()
    axs1.set_xlim(axs.get_xbound()[0], axs.get_xbound()[1])
    axs1.set_xticks(xticks_location)
    axs1.set_xticklabels(xticks_location[per_idx][rever_idx]+1)
    axs1.set_xlabel('(r_%s) input index' % item_type)
    cbar = fig.colorbar(im, ax=axs)
    ########### we load the original model and only use the DS decoder #########
    model_path = model_path + "/predict_cnn_fwm_%s_kv%s.pt" % (simu, activation)
    model = CnnFWM(data['param'])
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.requires_grad_(False)
    # define a decoder function
    def ds_decoder(x):
        T = x.shape[0]
        x = torch.cat((torch.ones(x.shape[0], 256).to(device) * 0.0,
                       model.l1.linear(x)*2), -1)
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
    #################################
    # extract r_value, w_value, new_w_value and old_w_value
    r_v = np.squeeze(memory_traces[episode]['r_value'])
    w_v = np.squeeze(memory_traces[episode]['w_value'])
    delta_w_v = np.squeeze(memory_traces[episode]['new_w_value'])
    w_beta = np.squeeze(memory_traces[episode]['w_beta'])
    v_len = delta_w_v.shape[1]
    #delta_w_v *= np.tile(np.expand_dims(w_beta, axis=1), (1, v_len))
    ## task images
    input_1 = np.squeeze(data['input1'][episode])
    output_1 = np.squeeze(data['prediction1'][episode])
    input_2 = np.squeeze(data['input2'][episode])
    output_2 = np.squeeze(data['prediction2'][episode])
    target_2 = np.squeeze(data['target2'][episode])
    plot_predict_results(input_1, input_2, output_1, target_2, output_2)
    # decoded r/w values
    plot_imgs = {}
    plot_imgs['input_1'] = input_1
    # decoded_w_v = -delta_w_v[:seq_len] * np.tile([w_beta[:seq_len]], (w_v.shape[1], 1)).T
    plot_imgs['decoded_w_v_1'] = ds_decoder(torch.FloatTensor(delta_w_v[:seq_len]).to(device)).cpu().numpy()
    plot_imgs['input_2'] = input_2
    plot_imgs['output_2'] = output_2
    plot_imgs['decoded_r_v_2'] = ds_decoder(torch.FloatTensor( r_v[seq_len + 1:]).to(device)).cpu().numpy()
    plot_imgs['decoded_w_v_2'] = ds_decoder(torch.FloatTensor(delta_w_v[seq_len + 1:]).to(device)).cpu().numpy()

    # plot all results
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    subfigs = fig.subfigures(nrows=len(list(plot_imgs.keys())), ncols=1)
    for row, (subfig, item_key) in enumerate(zip(subfigs, list(plot_imgs.keys()))):
        axs = subfig.subplots(nrows=1, ncols=seq_len)
        for col, ax in enumerate(axs):
            if '1' in item_key:
                step = col + 1
                ax.imshow(plot_imgs[item_key][col])
            else:
                step = col + 1 + seq_len + 1
                if col < seq_len-1: ax.imshow(plot_imgs[item_key][col])
            # ax.set_xlabel('trial %s' % step, fontsize=22)
            ax.tick_params(axis="both", which="both", length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

### For Fig. 2h #####
draw2 = False
if draw2:
    ### we examine the pca projections of the writing keys in stage one and see if they are
    ### more senstive to the digit's semantic meaning, instead of sensory details; then the
    ### writing values and see if their sensitivity selection is the opposite
    # load the data
    simu = 1
    data = pickle.load(open(save_dir + '/predict_cnn_fwm_%s_kv%s.pkl' % (simu, activation), 'rb'))
    memory_traces = data['memory_traces']
    num_test = len(memory_traces)
    digit_labels = data['input_label']
    per_idx = data['per_idx']
    # prepare data dictionary
    data_to_vis = {'w_key_1': [], 'w_value_1': [], 'r_key_2': [], 'r_value_2': []}
    digit_1_previous, digit_1_current = [], [] # previous and current digit label for stage 1
    digit_2_previous, digit_2_current = [], []

    digit_step = {}
    count = 0
    for episode in range(num_test):
        seq_len = len(digit_labels[episode])
        w_key_1 = np.squeeze(memory_traces[episode]['w_key'][1:seq_len])
        # w_beta_1 = np.squeeze(memory_traces[episode]['w_beta'][1:seq_len])
        # w_key_1 *= np.tile(np.expand_dims(w_beta_1, axis=1), (1, w_key_1.shape[1]))
        data_to_vis['w_key_1'].extend(w_key_1)
        data_to_vis['w_value_1'].extend(np.squeeze(memory_traces[episode]['new_w_value'][1:seq_len]))
        data_to_vis['r_key_2'].extend(np.squeeze(memory_traces[episode]['r_key'][seq_len+1:]))
        data_to_vis['r_value_2'].extend(np.squeeze(memory_traces[episode]['r_value'][seq_len+1:]))
        ll = []
        for step, class_ in enumerate(digit_labels[episode][:-1]):
            ss = (digit_labels[episode][step], digit_labels[episode][step+1])
            if ss not in list(digit_step.keys()):
                digit_step[ss] = count
                count += 1
                ll.append(digit_step[ss])
            else: ll.append(digit_step[ss])

        digit_1_previous.extend(digit_labels[episode][:-1])
        digit_1_current.extend(digit_labels[episode][1:])
        digit_2_previous.extend(digit_labels[episode][:-1][per_idx[episode]])
        digit_2_current.extend(digit_labels[episode][1:][per_idx[episode]])
    # plot projections for each
    method = 'umap'
    for item_key in list(data_to_vis.keys())[:2]:
        if item_key[-1] == '1':
            previous = digit_1_previous
            current = digit_1_current
        else:
            previous = digit_2_previous
            current = digit_2_current
        N = 7000
        components, embedding, axs, sc = plot_embedding(data_to_vis[item_key][:N], 2, previous[:N], method=method)
        handles = sc.legend_elements(num=range(0, 18))[0]  # extract the handles from the existing scatter plot
        #axs.legend(title='Digit', handles=handles, labels=range(0, 10), loc="upper right")
        plt.title('2D %s of %s in stage %s, colored by digits of the previous step'%(method, item_key[:-2], item_key[-1]))
        ###############
        components, embedding, axs, sc = plot_embedding(data_to_vis[item_key][:N], 2, current[:N], method=method, embedding=embedding)
        handles = sc.legend_elements(num=range(0, 18))[0]  # extract the handles from the existing scatter plot
        #axs.legend(title='Digit', handles=handles, labels=range(0, 10), loc="upper right")
        plt.title(
            '2D %s of %s in stage %s, colored by digits of the current step' % (method, item_key[:-2], item_key[-1]))

    # concatenate w_value_1, r_value_2, colored by digits
    # concatenation = np.concatenate((data_to_vis['w_key_1'], data_to_vis['r_key_2']))
    # coloring = np.concatenate((digit_1_previous, digit_2_previous))
    # components, embedding, axs, sc = plot_embedding(concatenation, 3, coloring, method=method)
    # handles = sc.legend_elements(num=range(0, 10))[0]  # extract the handles from the existing scatter plot
    # axs.legend(title='Digit', handles=handles, labels=range(0, 10), loc="upper right")
    # # plt.title('2D pca of w_key, r_key in stage 1, and r_key in stage 2, colored by digits')
    # # concatenate w_value_1, and r_value_2, colored by type
    # num = len(data_to_vis['w_key_1'])
    # coloring = np.concatenate((np.ones(num) * 1, np.ones(num) * 3))
    # components, embedding, axs, sc = plot_embedding(concatenation, 3, coloring, method=method, embedding=embedding)
    # handles = sc.legend_elements(num=[1, 3])[0]  # extract the handles from the existing scatter plot
    # axs.legend(title='', handles=handles, labels=['w key in stage 1', 'r key in stage 2'], loc="upper right")

### For supplementary Fig. 2a, 2b #####
# test for self-generated sequences
draw3 = False
if draw3:
    # # load the data
    # simu = 1
    # data = pickle.load(open(save_dir + '/predict_cnn_fwm_%s_kv%s_seq.pkl' % (simu, activation), 'rb'))
    # eps = np.random.choice(np.arange(len(data['input1'])), 1)[0]
    # sequence_length = len(data['input1'][eps])
    # fig, axe = plt.subplots(3, sequence_length, figsize=(10, 3))
    # for i in range(sequence_length):
    #     # reshape vector back to image
    #     axe[0][i].imshow(np.squeeze(data['input1'][eps][i]))
    #     if i < sequence_length - 1:
    #         axe[1][i+1].imshow(np.squeeze(data['prediction_cue'][eps][i]))
    #         axe[2][i + 1].imshow(np.squeeze(data['prediction_self'][eps][i]))
    #     for row in range(3):
    #         axe[row][i].tick_params(axis="both", which="both", length=0)
    #         plt.setp(axe[row][i].get_xticklabels(), visible=False)
    #         plt.setp(axe[row][i].get_yticklabels(), visible=False)

    max_len = 18 - 1 # maximum of the generated sequence length
    seq_lens = np.array([8, 12, 16])
    colors = ['blue', 'purple', 'red']
    error_seq_pos = [np.zeros((2, s)) for s in seq_lens]  # compute error based on the position of the image in a sequence
    count_seq_pos = np.zeros(len(seq_lens))
    simus = [1, 2]
    r_error, r_count = 0, 0
    for simu in simus:
        data = pickle.load(open(save_dir + '/predict_cnn_fwm_%s_kv%s_seq.pkl' % (simu, activation), 'rb'))
        for eps in range(len(data['input1'])):
            input = np.squeeze(data['input1'][eps])
            seq_len = len(input) - 1
            if seq_len not in seq_lens: continue
            else:
                ll = np.where(seq_lens==seq_len)[0][0]
                cue_seq = np.squeeze(data['prediction_cue'][eps])
                self_seq = np.squeeze(data['prediction_self'][eps])

                error_seq_pos[ll][0] += np.linalg.norm(input[1:] - cue_seq, axis=(1,2))
                error_seq_pos[ll][1] += np.linalg.norm(input[1:] - self_seq, axis=(1, 2))
                count_seq_pos[ll] += 1
                # compute the avg error for random sequences
                sh = input[1:].shape
                r_seq = np.random.rand(sh[0], sh[1], sh[2])
                r_error += np.mean(np.linalg.norm(input[1:] - r_seq, axis=(1,2)))
                r_count += 1

    fig, axs = plt.subplots()
    for i in range(len(seq_lens)):
        error_seq_pos[i] /= count_seq_pos[i]
        axs.plot(np.arange(1, seq_lens[i]+1), error_seq_pos[i][0], '-o', linestyle='dashed', label=seq_lens[i], color=colors[i], linewidth=2.5)
        axs.plot(np.arange(1, seq_lens[i]+1), error_seq_pos[i][1], '-o', linestyle='solid', label=seq_lens[i], color=colors[i], linewidth=2.5)
        axs.set_xlabel('Position in the sequence')
        axs.set_ylabel('mean error')
        axs.hlines(r_error/r_count, 1, max(seq_lens), color='k', linewidth=2.5)

        # axs.plot(error_seq_len[0], '-x', linestyle='dashed', label='cue-based', color='orange')
    # axs.plot(error_seq_len[1], '-x', linestyle='solid', label='self-generated', color='orange')

    #plt.legend()
    plt.tight_layout()

### For supplementary Fig. 2c, 2d #####
# generalization experiment
draw4 = False
if draw4:
    simus = [1]
    num_labels = 36
    gener_error = np.zeros(num_labels)
    predicts, targets, classes = [], [], []
    for simu in simus:
        data = pickle.load(open(save_dir + '/predict_cnn_fwm_%s_kv%s_general.pkl' % (simu, activation), 'rb'))
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
        data = pickle.load(open(save_dir + '/predict_cnn_fwm_%s_kv%s.pkl' % (simu, activation), 'rb'))
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

    axs.bar(x[:18]-w, height=gener_error[:18], edgecolor='k', facecolor='gray', width=w)
    axs.set_xticks([0, 18, 35])
    axs.set_xticklabels([1, 19, 36])
    axs.set_ylim([0, 9.5])
    plt.tight_layout()


plt.show()