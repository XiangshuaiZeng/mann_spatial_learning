import random
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from agents.simple_FW import FWMRNN
import numpy as np
import pickle


# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
    print(torch.backends.cudnn.enabled)
else:
    print("Device set to : cpu")

class CnnFWM(nn.Module):
    '''
    This is a FWM agent that output binary vectors of [0, 1]
    '''
    def __init__(self, p):
        super(CnnFWM, self).__init__()
        # build encoder
        self.encoder = nn.Sequential(
                        # in_channels, out_channels, kernel_size, stride=1, padding=0
                        nn.Conv2d(1, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Flatten(start_dim=1),
                        nn.Linear(((p['img_dim'][0]//4)**2)*64, p['i_size'])
                       ).to(device)
        # lstm + memory part
        if p['network'] == 'fwm':
            self.l1 = FWMRNN(p['slownet'], p['i_size'], p['hidden'], p['k_size'], p['v_size'], p['n_roll'],
                             p['k_activation'], p['v_activation']).to(device)
            out_size = 2 * p['hidden']
        elif p['network'] == 'fwm_v1':
            self.l1 = FWMRNN_V1(p['slownet'], p['i_size'], p['hidden'], p['k_size'], p['v_size'], p['n_roll'],
                                p['k_activation'], p['v_activation']).to(device)
            out_size = p['hidden']
        # buffer layer from output to latent dim
        self.mem_to_latent = nn.Linear(out_size, ((p['img_dim'][0]//4)**2)*64).to(device)
        # build decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
            nn.Sigmoid()  # output should lie between [0, 1]
        ).to(device)

        self.p = p
        self.reset_hidden()
        self.reset_parameters()

    def reset_hidden(self, keep_hidden=False, hiddens=None):
        if not keep_hidden:
            if self.p['slownet'] == 'lstm':
                rnn_hidden = (torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device),
                               torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device))
            elif self.p['slownet'] == 'rnn':
                rnn_hidden = torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device)
            if self.p['k_activation'] == 'dpfp':
                fwm_hidden = torch.zeros(self.p['batch_size'], self.p['v_size'],
                                         self.p['k_size']*2*self.p['n_roll']).to(device)
            else:
                fwm_hidden = torch.zeros(self.p['batch_size'], self.p['v_size'], self.p['k_size'] ).to(device)
            if self.p['network'] == 'fwm_v1':
                last_retrieved = torch.zeros(self.p['batch_size'], self.p['v_size']).to(device)
        else:
            if hiddens is None:
                if self.p['network'] == 'fwm':
                    rnn_hidden, fwm_hidden = self.hidden
                elif self.p['network'] == 'fwm_v1':
                    rnn_hidden, fwm_hidden, last_retrieved = self.hidden
            else:
                if self.p['network'] == 'fwm':
                    rnn_hidden, fwm_hidden = hidden
                elif self.p['network'] == 'fwm_v1':
                    rnn_hidden, fwm_hidden, last_retrieved = hidden
            if self.p['slownet'] == 'lstm':
                rnn_hidden = (rnn_hidden[0].detach(), rnn_hidden[1].detach())
            elif self.p['slownet'] == 'rnn':
                rnn_hidden = rnn_hidden.detach()
            fwm_hidden = fwm_hidden.detach()

        if self.p['network'] == 'fwm':
            self.hidden = (rnn_hidden, fwm_hidden)
        elif self.p['network'] == 'fwm_v1':
            self.hidden = (rnn_hidden, fwm_hidden, last_retrieved)

    def reset_parameters(self):
        # reset lstm or fwm
        self.l1.reset_parameters()

    def get_hidden(self):
        return self.hidden

    def clear_trace(self):
        self.l1.fwm.clear_trace()

    def __call__(self, x):
        x = x.to(device)
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        T, B = x.shape[0], x.shape[1]
        # convert input shape from (T,B,C,H,W) to (T*B,C,H,W)
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        latent = self.encoder(x)
        # convert latent shape from (T*B, latent_dim) to (T,B,latent_dim)
        latent = latent.view(T, B, -1)
        # pass to memory layer
        mem_out, self.hidden, _ = self.l1(latent, self.hidden)
        latent1 = self.mem_to_latent(mem_out)
        # convert latent shape from (T,B,latent_dim) to (T*B, C_Hidden, H_hidden, W_hidden)
        latent1 = latent1.view(T*B, -1)
        # the height and width of the feature map passing to the first deconv layer
        H_hidden = int(self.p['img_dim'][0]//4) # has been passed to two conv layers with stride 2
        latent1 = latent1.view(-1, 64, H_hidden, H_hidden)
        out = self.decoder(latent1)
        # convert the final output shape from (T*B,C,H,W) to (T,B,C,H,W)
        out = out.view(T, B, 1, self.p['img_dim'][0], self.p['img_dim'][1])
        return out

data_dir = '/local/xzeng/phd_projects/mann_spatial_learning/eminst_byclass'
# load digit data
train_img, train_label = loadlocal_mnist(
    images_path=data_dir + '/emnist-balanced-train-images-idx3-ubyte',
    labels_path=data_dir + '/emnist-balanced-train-labels-idx1-ubyte')

test_img, test_label = loadlocal_mnist(
    images_path=data_dir + '/emnist-balanced-test-images-idx3-ubyte',
    labels_path=data_dir + '/emnist-balanced-test-labels-idx1-ubyte')
img_dim = [28, 28]
train_img = np.array(train_img, dtype=float) / 255.0
test_img = np.array(test_img, dtype=float) / 255.0
# reshape the images to 2D
train_img = np.fliplr(np.rot90(np.reshape(train_img, (len(train_img), img_dim[0], img_dim[1])), k=1, axes=(1,2)))
test_img = np.fliplr(np.rot90(np.reshape(test_img, (len(test_img), img_dim[0], img_dim[1])), k=1, axes=(1,2)))


def plot_predict_results(inputs1, inputs2, targets2, outputs1, outputs2):
    sequence_length = len(targets2)
    fig, axe = plt.subplots(5, sequence_length, figsize=(10, 5))
    fig.suptitle('Stage 1: encoding,    stage 2: recall')
    for i in range(sequence_length):
        # reshape vector back to image
        axe[0][i].imshow(inputs1[i])
        plt.text(0.04, 0.16 * 5, 'stage 1, inputs:', fontsize=13, transform=plt.gcf().transFigure)
        axe[1][i].imshow(outputs1[i])
        plt.text(0.04, 0.16 * 4, 'stage 1, outputs:', fontsize=13, transform=plt.gcf().transFigure)
        axe[2][i].imshow(inputs2[i])
        plt.text(0.04, 0.16 * 3, 'stage 2, inputs:', fontsize=13, transform=plt.gcf().transFigure)
        axe[3][i].imshow(outputs2[i])
        plt.text(0.04, 0.16 * 2, 'stage 2, outputs:', fontsize=13, transform=plt.gcf().transFigure)
        axe[4][i].imshow(targets2[i])
        plt.text(0.04, 0.16 * 1, 'stage 2, groundtruth:', fontsize=13, transform=plt.gcf().transFigure)
        for row in range(5):
            axe[row][i].tick_params(axis="both", which="both", length=0)
            plt.setp(axe[row][i].get_xticklabels(), visible=False)
            plt.setp(axe[row][i].get_yticklabels(), visible=False)

def get_training_sequence(sequence_min_length, sequence_max_length, batch_size=1, test=False, valid_label = np.arange(0, 18)):
    # we only use the first 18 labels, which corresponds to the 10 digits and some uppercase letters
    # extract labels for each type of digit or letter
    train_label_indice, test_label_indice = {}, {}
    for lb in valid_label:
        train_label_indice[lb] = np.where(train_label == lb)[0]
        test_label_indice[lb] = np.where(test_label == lb)[0]

    if not test:
        label_indice = train_label_indice
        img = train_img
        labels = train_label
    else:
        label_indice = test_label_indice
        img = test_img
        labels = test_label
    sequence_length = random.randint(sequence_min_length, sequence_max_length)
    encode_idx = []
    # sample dotted digits idx randomly
    for _ in range(batch_size):
        arr = np.arange(len(valid_label))
        np.random.shuffle(arr)
        for lb in arr[:sequence_length]:
            encode_idx.append(np.random.choice(label_indice[lb], 1)[0])
    # based on the encoding idx, sample another sample of the digit type for each idx for retrieval
    retrieval_idx = []
    for lb in labels[encode_idx]:
        retrieval_idx.append(np.random.choice(label_indice[lb], 1)[0])
    # start building the inputs
    encode_digits = np.reshape(img[encode_idx], (sequence_length, batch_size, img_dim[0], img_dim[1]))
    retrieval_digits = np.reshape(img[retrieval_idx], (sequence_length, batch_size, img_dim[0], img_dim[1]))
    encode_digits = torch.FloatTensor(np.expand_dims(encode_digits, axis=2))
    retrieval_digits = torch.FloatTensor(np.expand_dims(retrieval_digits, axis=2))

    # input in the memorization stage
    input1 = torch.zeros(sequence_length + 1, batch_size, 1, img_dim[0], img_dim[1])
    input1[:sequence_length] = encode_digits
    input1[-1] = torch.ones((1, img_dim[0], img_dim[1]))
    # input and output in the prediction stage
    input2 = torch.zeros(sequence_length, batch_size, 1, img_dim[0], img_dim[1])
    output = torch.zeros(sequence_length, batch_size, 1, img_dim[0], img_dim[1])
    # randonly permute the previous binary vector
    per_idx = torch.randperm(sequence_length)
    input2 = torch.clone(retrieval_digits[per_idx])
    output = encode_digits[per_idx]
    return input1, input2, output, per_idx, labels[encode_idx]

def train(model_path, epochs, param):
    sequence_min_length = param['sequence_length_range'][0]
    sequence_max_length = param['sequence_length_range'][1]
    model = CnnFWM(param)
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-3)
    feedback_frequency = 100
    total_loss = []
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        #start = time.time()
        input1, input2, target2, per_idx, _ = get_training_sequence(sequence_min_length, sequence_max_length, param['batch_size'])
        #print('Time for sampling images: ', time.time() - start)
        model.reset_hidden(param['keep_hidden'])
        model.clear_trace()
        target1 = torch.zeros(target2.size())
        target1 = input1[:-1]
        y_out1 = torch.zeros(target1.size())
        # stage 1: perception
        #start = time.time()
        for j, vector in enumerate(input1[0:-1]):
            y_out1[j] = model(vector.unsqueeze(0))
        #print('Time for stage 1: ', time.time() - start)
        # buffer point
        _ = model(input1[-1].unsqueeze(0))
        # stage 2: recall
        y_out2 = torch.zeros(target2.size())
        for j in range(len(target2)):
            y_out2[j] = model(input2[j].unsqueeze(0))
        loss = F.binary_cross_entropy(y_out1, target1) + \
               F.binary_cross_entropy(y_out2, target2)*1.2
        #start = time.time()
        loss.backward()
        optimizer.step()
        #print('Time for update: ', time.time() - start)
        total_loss.append(loss.item())
        if (epoch+1) % feedback_frequency == 0:
            running_loss = sum(total_loss) / len(total_loss)
            print(f"Loss at step {epoch+1}: {running_loss}")
    torch.save(model.state_dict(), model_path)
    return param, total_loss

def eval(model_path, test_epochs, param):
    param['batch_size'] = 1
    sequence_min_length = param['sequence_length_range'][0]
    sequence_max_length = param['sequence_length_range'][1]
    model = CnnFWM(param)
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.requires_grad_(False)
    test_log = {'memory_traces': [], 'input1': [], 'prediction1': [], 'input2': [],  'prediction2': [], 'target2': [],
                'per_idx':[], 'param': param, 'input_label': []}
    for n in range(test_epochs):
        if n % 10 == 0:
            print('Test episode: %s'%n)
        input1, input2, target2, per_idx, input_label = get_training_sequence(sequence_min_length, sequence_max_length, test=True)
        model.reset_hidden(param['keep_hidden'])
        model.clear_trace()
        target1 = torch.zeros(target2.size())
        target1 = input1[:-1]
        y_out1 = torch.zeros(target1.size())
        # stage 1: perception
        for j, vector in enumerate(input1[0:-1]):
            y_out1[j] = model(vector.unsqueeze(0))
        # buffer point
        _ = model(input1[-1].unsqueeze(0))
        # stage 2: recall
        y_out2 = torch.zeros(target2.size())
        for j in range(len(target2)):
            y_out2[j] = model(input2[j].unsqueeze(0))
        # plot_predict_results(torch.squeeze(input1), torch.squeeze(input2),
        #                      torch.squeeze(target2), torch.squeeze(y_out1), torch.squeeze(y_out2))
        # plt.show()
        # log data
        memory_trace = model.l1.get_trace()
        copy_trace = {}
        for key in list(memory_trace.keys()):
            copy_trace[key] = np.array(memory_trace[key])
        test_log['memory_traces'].append(copy_trace)
        test_log['input1'].append(input1[0:-1].detach().cpu().numpy())
        test_log['prediction1'].append(y_out1.detach().cpu().numpy())
        test_log['input2'].append(input2.detach().cpu().numpy())
        test_log['prediction2'].append(y_out2.detach().cpu().numpy())
        test_log['target2'].append(target2.detach().cpu().numpy())
        test_log['per_idx'].append(per_idx.detach().cpu().numpy())
        test_log['input_label'].append(input_label)
    return test_log

def eval_general(model_path, test_epochs, param):
    param['batch_size'] = 1
    sequence_min_length = param['sequence_length_range'][0]
    sequence_max_length = param['sequence_length_range'][1]
    model = CnnFWM(param)
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.requires_grad_(False)
    test_log = {'memory_traces': [], 'input1': [], 'prediction1': [], 'input2': [],  'prediction2': [], 'target2': [],
                'per_idx':[], 'param': param, 'input_label': []}
    for n in range(test_epochs):
        if n % 10 == 0:
            print('Test episode: %s'%n)
        input1, input2, target2, per_idx, input_label = get_training_sequence(sequence_min_length, sequence_max_length,
                                                                test=True, valid_label=np.arange(0, 36))
        model.reset_hidden(param['keep_hidden'])
        model.clear_trace()
        target1 = torch.zeros(target2.size())
        target1 = input1[:-1]
        y_out1 = torch.zeros(target1.size())
        # stage 1: perception
        for j, vector in enumerate(input1[0:-1]):
            y_out1[j] = model(vector.unsqueeze(0))
        # buffer point
        _ = model(input1[-1].unsqueeze(0))
        # stage 2: recall
        y_out2 = torch.zeros(target2.size())
        for j in range(len(target2)):
            y_out2[j] = model(target2[j].unsqueeze(0))
        # plot_predict_results(torch.squeeze(input1), torch.squeeze(input2),
        #                      torch.squeeze(target2), torch.squeeze(y_out1), torch.squeeze(y_out2))
        # plt.show()
        # log data
        memory_trace = model.l1.get_trace()
        copy_trace = {}
        for key in list(memory_trace.keys()):
            copy_trace[key] = np.array(memory_trace[key])
        test_log['memory_traces'].append(copy_trace)
        test_log['input1'].append(input1[0:-1].detach().cpu().numpy())
        test_log['prediction1'].append(y_out1.detach().cpu().numpy())
        test_log['input2'].append(input2.detach().cpu().numpy())
        test_log['prediction2'].append(y_out2.detach().cpu().numpy())
        test_log['target2'].append(target2.detach().cpu().numpy())
        test_log['per_idx'].append(per_idx.detach().cpu().numpy())
        test_log['input_label'].append(input_label)
    return test_log

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    # define parameters for the model and training
    param = {}
    param['network'] = 'fwm' # only used by the FWM model
    param['slownet'] = 'lstm' # only used by the FWM model
    param['sequence_length_range'] = [8, 18]
    param['img_dim'] = img_dim
    param['batch_size'] = 10
    param['k_size'] = 128
    param['v_size'] = 128
    param['hidden'] = 256
    param['i_size'] = 256
    param['n_roll'] = 2  # only used for dpfp activation function
    activation = 'relu'
    param['k_activation'] = activation
    param['v_activation'] = activation
    param['weight_l2'] = 0.0  # apply l2 regulerization to all weights
    param['keep_hidden'] = False
    for simu in range(1, 10+1):
        seed = np.random.randint(0, 2**32)
        set_seed(seed)
        model_path = "models/recall_cnn_fwm_%s_kv%s.pt" % (simu, activation)
        if_train = True
        if_eval = True
        epochs = 80000
        test_epochs = 300
        if if_train:
            _, total_loss = train(model_path, epochs, param)
        if if_eval:
            test_log = eval(model_path, test_epochs, param)
            test_log['training_loss'] = total_loss
            #store testing data
            pickle.dump(test_log, open('data_test/recall_cnn_fwm_%s_kv%s.pkl' % (simu, activation),
                'wb'))
            # generalization experiment
            test_log = eval_general(model_path, test_epochs, param)
            #store testing data
            pickle.dump(test_log, open('data_test/recall_cnn_fwm_%s_kv%s_general.pkl' % (simu, activation),
                'wb'))

