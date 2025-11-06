import torch
import math
from torch import nn
from torch.autograd import Variable
from torch.nn import LayerNorm
from torch.nn.utils.parametrizations import orthogonal
import torch.nn.functional as F
import numpy as np
import pdb

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()



class FWMRNN(nn.Module):
    def __init__(self, slownet, isize, hsize, k_size, v_size, n_roll,
                 k_activate='tanh', v_activate='linear'):
        super().__init__()
        q_size = k_size
        self.n_roll = n_roll
        self.slownet = slownet
        if slownet == 'lstm':
            self.rnn = nn.LSTM(isize, hsize, 1, dropout=0).to(device)
        elif slownet == 'rnn':
            self.rnn = MyRNN(isize, hsize).to(device)

        self.fwm = FWM(hsize, k_size, v_size, q_size, k_activate, v_activate)
        self.fwm.n_roll = n_roll
        self.linear = nn.Linear(v_size, hsize).to(device)

        self.isize = isize
        self.hsize = hsize
        self.reset_parameters()

    def reset_parameters(self):
        # reset lstm parameters
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p.data)
            elif 'bias_ih' in name:
                p.data.fill_(0)
                # Set forget-gate bias to 1
                n = p.size(0)
                p.data[(n // 4):(n // 2)].fill_(1)
        # reset memory heads parameters
        self.fwm.reset_parameters()
        # reset linear head parameters
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def get_trace(self):
        return self.fwm.m_trace

    def __call__(self, inputs, hidden):
        rnn_hidden, F = hidden
        h, rnn_hidden = self.rnn(inputs, rnn_hidden)
        outputs, ks, qs = [], [], []
        # here x is the hidden states of the rnn for each time step
        for t, h_t in enumerate(h):
            # we perform one-step reading
            F, ks_t = self.fwm.write(h_t, F)
            o_t, qs_t = self.fwm(h_t, F)
            outputs.append(o_t)
            ks.append(ks_t)
            qs.append(qs_t)
        s = torch.stack(outputs, dim=0)
        ks = torch.stack(ks, dim=0)
        qs = torch.stack(qs, dim=0)
        output = torch.cat((h*0, self.linear(s)), 2)
        keys = torch.cat((ks, qs), 2)
        hidden = (rnn_hidden, F)
        return output, hidden, keys

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        nn.init.zeros_(m.bias.data)

class FWM(nn.Module):
    def __init__(self, hidden_size, k_size, v_size, q_size, k_acti='tanh', v_acti='linear'):
        """
        hidden_size: lstm hidden side
        k: key
        v: value to write into the memory
        q: query for reading
        k_acti: activation function for w and r keys
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.k_size = k_size
        self.v_size = v_size
        self.q_size = q_size
        self.k_activate = k_acti
        self.v_activate = v_acti
        self.b_size = 1      # controls writing strength
        self.m_size = 1      # controls reading strength
        self.n_roll = 2

        # write
        self.W_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.k_size)
        ).to(device)

        self.W_v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.v_size)
        ).to(device)

        self.W_b = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.b_size)
        ).to(device)


        # read
        self.W_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.q_size)
        ).to(device)

        self.W_m = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.m_size)
        ).to(device)

        self.reset_parameters()

        # record the writing key, value, beta and reading key, value for each time step
        self.m_trace = {'w_key':[], 'w_value':[], 'w_beta':[], 'r_key':[],'new_w_value':[],
                        'r_value':[], 'r_beta':[], 'rnn_out':[], 'rnn_cell':[]}

    def clear_trace(self):
        for key in list(self.m_trace.keys()):
            del self.m_trace[key][:]

    def reset_parameters(self):
        self.W_k.apply(weights_init)
        self.W_v.apply(weights_init)
        self.W_b.apply(weights_init)
        self.W_q.apply(weights_init)
        self.W_m.apply(weights_init)


    def write(self, z, Fw):
        # z: [batch_size, hidden_size]
        # F: [batch_size, k_size, r_size]

        ks = self.W_k(z)
        if self.k_activate == 'tanh':
            ks = F.tanh(ks)
        elif self.k_activate == 'relu':
            ks = F.relu(ks)
        elif self.k_activate == 'elu':
            ks = F.elu(ks) + 1
        elif self.k_activate == 'lrelu':
            ks = F.leaky_relu(ks)
        elif self.k_activate == 'sigmoid':
            ks = F.sigmoid(ks)
        elif self.k_activate == 'ptanh':  # postive tanh
            ks = (F.tanh(ks) + 1) / 2
        elif self.k_activate == 'linear':
            pass

        vs = self.W_v(z)
        if self.v_activate == 'relu':
            vs = F.relu(vs)
        elif self.v_activate == 'elu':
            vs = F.elu(vs) + 1
        elif self.v_activate == 'lrelu':
            vs = F.leaky_relu(vs)
        elif self.v_activate == 'tanh':
            vs = F.tanh(vs)
        elif self.v_activate == 'sigmoid':
            vs = F.sigmoid(vs)
        elif self.v_activate == 'ptanh':  # postive tanh
            vs = (F.tanh(vs) + 1) / 2
        elif self.v_activate == 'linear':
            pass

        bs = torch.sigmoid(self.W_b(z))
        # *: [batch_size, *_size]
        # book keeping
        self.m_trace['w_key'].append(ks.detach().cpu().numpy())
        self.m_trace['w_value'].append(vs.detach().cpu().numpy())
        self.m_trace['w_beta'].append(bs.detach().cpu().numpy())

        # slow implementation
        vs_old = torch.einsum('bij, bj->bi', Fw, ks)
        new_v =  vs - vs_old
        self.m_trace['new_w_value'].append(new_v.detach().cpu().numpy())

        deltaF = torch.einsum("bi,bj->bij", bs * new_v, ks)
        Fw = Fw + deltaF

        # scale F down if norm is > 1
        F_norm = Fw.view(Fw.shape[0], -1).norm(dim=-1)
        F_norm = torch.relu(F_norm - 1) + 1
        Fw = Fw / F_norm.unsqueeze(1).unsqueeze(1)

        return Fw, ks

    def __call__(self, z, Fw):
        qs = self.W_q(z)
        if self.k_activate == 'tanh':
            qs = F.tanh(qs)
        elif self.k_activate == 'relu':
            qs = F.relu(qs)
        elif self.k_activate == 'elu':
            qs = F.elu(qs) + 1
        elif self.k_activate == 'lrelu':
            qs = F.leaky_relu(qs)
        elif self.k_activate == 'sigmoid':
            qs = F.sigmoid(qs)
        elif self.k_activate == 'ptanh':  # postive tanh
            qs = (F.tanh(qs) + 1) / 2
        elif self.k_activate == 'linear':
            pass

        # : [q_size]
        ms = torch.sigmoid(self.W_m(z))
        # ms = torch.zeros(ms.shape).to(device)
        outputs = torch.einsum('bij, bj->bi', Fw, ms*qs)

        # book-keeping
        self.m_trace['r_key'].append(qs.detach().cpu().numpy())
        self.m_trace['r_beta'].append(ms.detach().cpu().numpy())
        self.m_trace['r_value'].append(outputs.detach().cpu().numpy())

        return outputs, qs