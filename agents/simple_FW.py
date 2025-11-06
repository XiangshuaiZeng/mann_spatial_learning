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

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size, hidden_size).to(device)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size).to(device)

    def forward(self, x, hidden):
        output = []
        for x_t in x:
            hidden = F.elu(self.in2hidden(x_t) + self.hidden2hidden(hidden)) + 1
            output.append(hidden)
        output = torch.stack(output)
        output = torch.squeeze(output, dim=1)
        return output, hidden

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
        if self.slownet == 'lstm':
            self.fwm.m_trace['rnn_out'].append(rnn_hidden[0].detach().cpu().numpy())
            self.fwm.m_trace['rnn_cell'].append(rnn_hidden[1].detach().cpu().numpy())
        elif self.slownet == 'rnn':
            self.fwm.m_trace['rnn_out'].append(rnn_hidden.detach().cpu().numpy())
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
        output = torch.cat((h, self.linear(s)), 2)
        keys = torch.cat((ks, qs), 2)
        hidden = (rnn_hidden, F)
        return output, hidden, keys

class FWMRNN_V1(nn.Module):
    def __init__(self, slownet, isize, hsize, k_size, v_size, n_roll,
                 k_activate='tanh', v_activate='linear'):
        super().__init__()
        q_size = k_size
        self.n_roll = n_roll
        self.slownet = slownet
        if slownet == 'lstm':
            self.rnn = nn.LSTM(isize+v_size, hsize, 1, dropout=0).to(device)
        elif slownet == 'rnn':
            self.rnn = MyRNN(isize+v_size, hsize).to(device)

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
        rnn_hidden, F, last_retrieved = hidden
        outputs, qs, ks = [], [], []
        # here x is the hidden states of the rnn for each time step
        for t, x_t in enumerate(inputs):
            x_t = torch.cat((x_t, last_retrieved), 1)
            x_t = torch.unsqueeze(x_t, 0)
            h_t, rnn_hidden = self.rnn(x_t, rnn_hidden)
            if self.slownet == 'lstm':
                self.fwm.m_trace['rnn_out'].append(rnn_hidden[0].detach().cpu().numpy())
                self.fwm.m_trace['rnn_cell'].append(rnn_hidden[1].detach().cpu().numpy())
            elif self.slownet == 'rnn':
                self.fwm.m_trace['rnn_out'].append(rnn_hidden.detach().cpu().numpy())
            h_t = torch.squeeze(h_t,dim=0)
            # we perform one-step reading
            F, ks_t = self.fwm.write(h_t, F)
            last_retrieved, qs_t = self.fwm(h_t, F)
            outputs.append(h_t)
            ks.append(ks_t)
            qs.append(qs_t)

        output = torch.stack(outputs, dim=0)
        ks = torch.stack(ks, dim=0)
        qs = torch.stack(qs, dim=0)
        keys = torch.cat((ks, qs), 2)
        hidden = (rnn_hidden, F, last_retrieved)
        return output, hidden, keys


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
        self.W_k = nn.Linear(hidden_size, self.k_size).to(device)
        self.W_v = nn.Linear(hidden_size, self.v_size).to(device)
        self.W_b = nn.Linear(hidden_size, self.b_size).to(device)

        # read
        self.W_q = nn.Linear(hidden_size, self.q_size).to(device)
        self.W_m = nn.Linear(hidden_size, self.m_size).to(device)

        self.reset_parameters()

        # record the writing key, value, beta and reading key, value for each time step
        self.m_trace = {'w_key':[], 'w_value':[], 'w_beta':[], 'r_key':[],'new_w_value':[],
                        'r_value':[], 'r_beta':[], 'rnn_out':[], 'rnn_cell':[]}

    def clear_trace(self):
        for key in list(self.m_trace.keys()):
            del self.m_trace[key][:]

    def reset_parameters(self):
        nn.init.orthogonal_(self.W_k.weight)
        nn.init.orthogonal_(self.W_v.weight)
        nn.init.orthogonal_(self.W_q.weight)
        nn.init.orthogonal_(self.W_b.weight)
        nn.init.orthogonal_(self.W_m.weight)

        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.W_q.bias)

    def mul_roll_repeat(self, x):
        rolls = []
        for i in range(1, self.n_roll + 1):
            rolls.append(x * x.roll(shifts=i, dims=-1))
        return torch.cat(rolls, dim=-1)

    def write(self, z, Fw):
        # z: [batch_size, hidden_size]
        # F: [batch_size, k_size, r_size]
        ks = self.W_k(z)
        if self.k_activate == 'dpfp':
            ks = torch.cat([F.relu(ks), F.relu(-ks)], dim=-1)
            ks = self.mul_roll_repeat(ks)
            # normalize k and q, crucial for stable training.
            ks = ks / ks.sum(dim=-1, keepdim=True)
        elif self.k_activate == 'tanh':
            ks = F.tanh(ks)
        elif self.k_activate == 'relu':
            ks = F.relu(ks)
        elif self.k_activate == 'elu':
            ks = F.elu(ks) + 1
        elif self.k_activate == 'lrelu':
            ks = F.leaky_relu(ks)
        elif self.k_activate == 'sigmoid':
            ks = F.sigmoid(ks)
        elif self.k_activate == 'linear':
            pass
        #ks = ks / ks.sum(dim=-1, keepdim=True)

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
        elif self.v_activate == 'linear':
            pass
        #vs = vs / vs.sum(dim=-1, keepdim=True)
        bs = torch.sigmoid(self.W_b(z))
        #if bs[0][0] < 0.03: bs[0][0]= 0
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
        F_norm = torch.relu(F_norm - 2) + 1
        Fw = Fw / F_norm.unsqueeze(1).unsqueeze(1)

        return Fw, ks

    def __call__(self, z, Fw):
        qs = self.W_q(z)
        if self.k_activate == 'dpfp':
            qs = torch.cat([F.relu(qs), F.relu(-qs)], dim=-1)
            qs = self.mul_roll_repeat(qs)
            # normalize k and q, crucial for stable training.
            qs = qs / qs.sum(dim=-1, keepdim=True)
        elif self.k_activate == 'tanh':
            qs = F.tanh(qs)
        elif self.k_activate == 'relu':
            qs = F.relu(qs)
        elif self.k_activate == 'elu':
            qs = F.elu(qs) + 1
        elif self.k_activate == 'lrelu':
            qs = F.leaky_relu(qs)
        elif self.k_activate == 'sigmoid':
            qs = F.sigmoid(qs)
        elif self.k_activate == 'linear':
            pass

        #qs = qs / qs.sum(dim=-1, keepdim=True)

        # : [q_size]
        ms = torch.sigmoid(self.W_m(z))
        # ms = torch.zeros(ms.shape).to(device)
        outputs = torch.einsum('bij, bj->bi', Fw, ms*qs)
        # if self.v_activate == 'relu':
        #     outputs = F.relu(outputs)
        # elif self.v_activate == 'lrelu':
        #     outputs = F.leaky_relu(outputs)
        # elif self.v_activate == 'tanh':
        #     outputs = F.tanh(outputs)
        # elif self.v_activate == 'sigmoid':
        #     outputs = F.sigmoid(outputs)
        # elif self.v_activate == 'linear':
        #     pass
        # book-keeping
        self.m_trace['r_key'].append(qs.detach().cpu().numpy())
        self.m_trace['r_beta'].append(ms.detach().cpu().numpy())
        self.m_trace['r_value'].append(outputs.detach().cpu().numpy())

        return outputs, qs