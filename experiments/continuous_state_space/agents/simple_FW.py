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
    def __init__(self, isize, hsize, k_size, v_size, num_reads):
        super().__init__()
        q_size = k_size
        self.num_reads = num_reads
        self.rnn = nn.LSTM(isize, hsize, 1, dropout=0).to(device)

        self.fwm = FWM(hsize, k_size, v_size, q_size)
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
            # elif 'bias_hh' in name:
            #     p.data.fill_(0)

        # reset memory heads parameters
        self.fwm.reset_parameters()
        # reset linear head parameters
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def get_trace(self):
        return self.fwm.m_trace

    def __call__(self, inputs, hidden):
        lstm_hidden, F = hidden
        h, lstm_hidden = self.rnn(inputs, lstm_hidden)

        self.fwm.m_trace['rnn_out'].append(lstm_hidden[0].detach().cpu().numpy())
        self.fwm.m_trace['rnn_cell'].append(lstm_hidden[1].detach().cpu().numpy())
        outputs = []
        # here x is the hidden states of the rnn for each time step
        for t, h_t in enumerate(h):
            # we perform one-step reading
            F = self.fwm.write(h_t, F)
            o_t = self.fwm(h_t, F)
            # # we perform multi-step reading
            # zeros = torch.tensor(np.zeros(x_t.shape), dtype=torch.float32).to(device)
            # o_t = torch.cat((x_t, zeros), 1)
            # for _ in range(self.num_reads):
            #     read_out = self.fwm(o_t, F)
            #     o_t = torch.cat((x_t, self.linear(read_out)), 1)
            outputs.append(o_t)

        s = torch.stack(outputs, dim=0)

        output = torch.cat((h, self.linear(s)), 2)
        #output = self.linear(s) #ONLY FWM OUTPUT, WITHOUT LSTM

        hidden = (lstm_hidden, F)
        return output, hidden


class FWM(nn.Module):
    def __init__(self, hidden_size, k_size, v_size, q_size):
        """
        hidden_size: lstm hidden side
        k: key
        v: value to write into the memory
        q: query for reading
        b: beta, writing strength
        """
        self.i=0
        super().__init__()
        self.hidden_size = hidden_size
        self.k_size = k_size
        self.v_size = v_size
        self.q_size = q_size
        self.b_size = 1      # controls writing strength
        self.m_size = 1      # controls reading strength

        # write
        self.W_k = nn.Linear(hidden_size, self.k_size).to(device)
        self.W_v = nn.Linear(hidden_size, self.v_size).to(device)
        self.W_b = nn.Linear(hidden_size, self.b_size).to(device)

        # read
        self.W_q = nn.Linear(hidden_size, self.q_size).to(device)
        self.W_m = nn.Linear(hidden_size, self.m_size).to(device)

        self.reset_parameters()

        # record the writing key, value, beta and reading key, value for each time step
        self.m_trace = {'w_key':[], 'w_value':[], 'w_beta':[], 'r_key':[], 'new_w_value':[], 'fwm':[], 'delta_fwm':[],
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


    def write(self, z, Fw):
        # z: [batch_size, hidden_size]
        # F: [batch_size, k_size, r_size]

        #ks = F.tanh(self.W_k(z))  ##with tanh activation function
        #ks = F.sigmoid(self.W_k(z))
        ks = F.relu(self.W_k(z))
        # normalize k, crucial for stable training.
        # ks = ks / ks.sum(-1, keepdim=True)
        vs = F.relu(self.W_v(z))
        #vs = F.sigmoid(self.W_v(z))
        #vs = F.relu(self.W_v(z))
        #vs = F.tanh(vs) # add activation function
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
        # scale F down if norm is > 1
        F_norm = Fw.view(Fw.shape[0], -1).norm(dim=-1)
        F_norm = torch.relu(F_norm - 2) + 1
        #self.m_trace['delta_fwm'].append(F_norm.detach().cpu().numpy())

        Fw = Fw / F_norm.unsqueeze(1).unsqueeze(1)

        self.m_trace['fwm'].append(Fw.detach().cpu().numpy())
        #self.m_trace['delta_fwm'].append(deltaF.detach().cpu().numpy())

        return Fw

    def __call__(self, z, Fw):
        #qs = F.sigmoid(self.W_q(z)) ##with tanh activation function
        qs = F.relu(self.W_q(z))
        # : [q_size]
        ms = torch.sigmoid(self.W_m(z))
        outputs = torch.einsum('bij, bj->bi', Fw, ms*qs)
        #outputs = F.relu(outputs) #ruins the performance
        self.i+=1

        # book-keeping
        self.m_trace['r_key'].append(qs.detach().cpu().numpy())
        self.m_trace['r_beta'].append(ms.detach().cpu().numpy())
        self.m_trace['r_value'].append(outputs.detach().cpu().numpy())

        return outputs