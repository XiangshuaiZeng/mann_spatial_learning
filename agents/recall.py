import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import wandb
import os
import time
import pickle
from datetime import datetime
import gym
import numpy as np
import cupy as cp
from agents.simple_FW import FWMRNN, FWMRNN_V1

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
    print(torch.backends.cudnn.enabled)
else:
    print("Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.goal_predictions = []
        self.goals_idx = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.goal_predictions[:]
        del self.goals_idx[:]

class callbacks():
    def __init__(self, rl_parent, custom_callbacks={}):
        '''
        Callback class. Used for visualization and scenario control.

        Parameters
        ----------
        rl_parent :                         Reference to the RL agent.
        custom_callbacks :                  The custom callbacks defined by the user.

        Returns
        ----------
        None
        '''
        # store the hosting class
        self.rl_parent = rl_parent
        # store the trial end callback function
        self.custom_callbacks = custom_callbacks

    def on_trial_end(self, logs: dict):
        '''
        The following function is called whenever a trial ends, and executes callbacks defined by the user.

        Parameters
        ----------
        logs :                              The trial log.

        Returns
        ----------
        None
        '''
        logs['rl_parent'] = self.rl_parent
        if 'on_trial_end' in self.custom_callbacks:
            for custom_callback in self.custom_callbacks['on_trial_end']:
                custom_callback(logs)

class AgentFWM(nn.Module):
    def __init__(self, p):
        super(AgentFWM, self).__init__()
        if p['network'] == 'fwm':
            out_size = 2 * p['hidden']
            self.l1 = FWMRNN(p['slownet'], p['i_size'], p['hidden'], p['k_size'], p['v_size'], p['n_roll'],
                             p['k_activation'], p['v_activation']).to(device)
        elif p['network'] == 'fwm_v1':
            self.l1 = FWMRNN_V1(p['slownet'], p['i_size'], p['hidden'], p['k_size'], p['v_size'], p['n_roll'],
                                p['k_activation'], p['v_activation']).to(device)
            out_size = p['hidden']

        self.relu = nn.ReLU().to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        self.merge = nn.Linear(p['obs_dim']+400, p['i_size']).to(device)
        # the output has the same size of sensory inputs
        self.output = nn.Linear(out_size, p['obs_dim']).to(device)

        if p['ds_nonlinearity']:
            self.h_middle = nn.Linear(out_size, out_size).to(device)

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
        # reset linear head parameters
        nn.init.orthogonal_(self.output.weight)

    def get_hidden(self):
        return self.hidden

    def clear_trace(self):
        self.l1.fwm.clear_trace()

    def __call__(self, x):
        x = x.to(device)
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        x = self.relu(self.merge(x))
        out, self.hidden, keys = self.l1(x, self.hidden)
        out = out.reshape(-1, out.size(-1))
        # compute actions and value functions
        if self.p['ds_nonlinearity']:
            out = self.relu(self.h_middle(out))
        predicts = self.output(out)

        return predicts

class PPO:
    def __init__(self, interface_OAI, network='fwm', gamma=0.9, process_im='RP', args=None, custom_callbacks={}):

        self.image_dim = interface_OAI.observation_space.shape
        self.action_dim = interface_OAI.action_space.n
        self.num_envs = interface_OAI.num_envs
        self.network = network

        self.interface_OAI = interface_OAI
        self.lr = args['lr']
        self.process_im = process_im
        # the dimension of the inputs to the rnn in the model
        self.rnn_in_dim = args['i_size']
        self.args = args
        self.num_states = len(list(self.interface_OAI.modules_list[0]['world'].env.keys())[:-3])
        if process_im == 'RP':
            self.init_projection = False
            self.rp_dim = args['obs_dim'] #250
        elif process_im == 'one_hot':
            self.args['obs_dim'] = self.num_states
        elif process_im == 'r_binary':
            self.init_projection = False

        self.buffer = RolloutBuffer()
        args['n_actions'] = self.action_dim
        args['batch_size'] = self.num_envs

        self.policy = AgentFWM(args)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.engaged_callbacks = callbacks(self, custom_callbacks)
        self.keep_hidden = args['keep_hidden']

    def train(self, total_episodes=1000, collect_time_steps=150, predict_time_steps=30, checkpoint_path=''):
        episode = 0
        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        # these logs are saved to analyse the training afterwards
        training_log = {'episode': [], 'episode_time_step': [], 'avg_episode_reward': [],
                        'step_reward': [], 'loss': [], 'g_loss':[]}
        # Training loop
        print("============================================================================================")
        print("Start training")
        while episode < total_episodes:
            start = time.time()
            self.time_step = 0
            self.targets = 0
            # clear up all trajatories to release memory
            for i in range(self.num_envs):
                self.interface_OAI.modules_list[i]['spatial_representation'].clear_trajectories()
            # reset all environments
            states, states_idx = self.interface_OAI.reset()
            # reset / initialize hidden states
            self.policy.reset_hidden(keep_hidden=self.keep_hidden)
            self.policy.clear_trace()
            # we also reinitialize random observation process
            #self.init_projection = False
            loss = 0
            # initialize the previous action, reward, terminal flag
            rewards = np.zeros(self.num_envs)
            actions = ['reset' for _ in range(self.num_envs)]
            # choose a random time step between 0 and collection_steps where the states
            # need to be remembered
            mem_step = np.random.randint(1, collect_time_steps, 1)
            while self.time_step < (collect_time_steps+predict_time_steps):
                # pre-process each image
                states = self.process_observaton(states, states_idx)
                if self.time_step == mem_step:
                    self.targets = states
                    rewards = np.ones(self.num_envs)
                else:
                    rewards = np.zeros(self.num_envs)
                # prepare tensors of last reward
                rewards_input = np.ones((self.num_envs, 100))*np.tile(np.array([rewards]).T,(1, 100))
                rewards_input = np.expand_dims(rewards_input, axis=0)
                # prepare tensors of last action
                actions_input = np.zeros((self.num_envs, self.action_dim*100))
                # when the collection time ends, turn action to predict mode
                if self.time_step > collect_time_steps:
                    actions = ['predict' for _ in range(self.num_envs)]
                    # build targets
                    targets = torch.FloatTensor(self.targets).to(device)
                else:
                    targets = torch.zeros(states.shape).to(device)

                for i, a in enumerate(actions):
                    if a == 'reset':
                        actions_input[i] = np.zeros(self.action_dim*100)
                    elif a == 'predict':
                        actions_input[i] = np.ones(self.action_dim * 100)
                    else:
                        actions_input[i] = np.tile(np.eye(self.action_dim)[a], (1, 100))
                actions_input = np.expand_dims(actions_input, axis=0)
                # convert numpy array to tensor
                states = torch.FloatTensor(np.expand_dims(states, axis=0))
                states = torch.cat((states, torch.FloatTensor(rewards_input)), -1)
                states = torch.cat((states, torch.FloatTensor(actions_input)), -1)
                # generate predictions
                predicts = self.policy(states.to(device))
                if self.time_step > collect_time_steps:
                    # compute the loss for the current time step
                    loss += torch.mean(self.mse_loss(predicts, targets),1)
                # randomly sample actions for the next step
                actions = np.random.randint(low=0, high=self.action_dim, size=self.num_envs)
                # book-keeping
                self.buffer.states.append(states)
                self.buffer.actions.append(actions)
                # move to next step
                self.time_step += 1
                states, _, _, _, _ = self.interface_OAI.step(actions)
                self.buffer.rewards.append(np.array(rewards))
            # update the model
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # clear buffer
            self.buffer.clear()

            episode += 1
            loss_record = loss.mean().detach().cpu().numpy()
            # logging
            training_log['episode'].append(episode)
            training_log['episode_time_step'].append(self.time_step)
            # This log is for the callback function
            logs = {'trial_reward': loss_record, 'trial': episode - 1}
            self.engaged_callbacks.on_trial_end(logs)

            training_log['loss'].append(loss_record)
            print("Episode : {} \t Loss: {}".format(episode, loss_record))
            # save model
            if checkpoint_path != '' and episode%self.args['save_freq'] == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path + '_%s'%episode)
                self.save(checkpoint_path+ '_%s'%episode)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Total training time: ", end_time - start_time)

        return training_log

    def save(self, checkpoint_path):
        # save the model's parameter
        torch.save(self.policy.state_dict(), checkpoint_path + '.pth')
        # save the random project matrix, if there is one
        if self.process_im == 'RP':
            print('Saving RP')
            pickle.dump(self.random_projections, open(checkpoint_path + '_RP_%s.pkl'%self.rp_dim, 'wb'))

    def load(self, checkpoint_path):
        # load the model's parameters
        self.policy.load_state_dict(torch.load(checkpoint_path+'.pth', map_location=lambda storage, loc: storage))
        # load the random project matrix, if there is one
        if self.process_im == 'RP':
            self.random_projections = pickle.load(open(checkpoint_path + '_RP.pkl', 'rb'))
            self.init_projection = True

    def test(self, total_episodes=1000, collect_time_steps=150, predict_time_steps=30):
        # these logs are saved to analyse the training afterwards
        test_log = {'trajectories': [], 'memory_traces': [], 'prediction':[],
                    'target': [], 'step_reward': []}
        # turn off the training flag of the model
        self.policy.train(False)
        # Test loop
        print("====================================================================")
        print("Start testing")
        episode = 0
        while episode < total_episodes:
            self.time_step = 0
            loss = 0  # only for recording accuracy
            test_log['prediction'].append([])
            test_log['target'].append([])
            test_log['step_reward'].append([])
            # reset environments
            states, states_idx = self.interface_OAI.reset()
            # reset / initialize hidden states
            self.policy.reset_hidden(keep_hidden=self.keep_hidden)
            self.policy.clear_trace()
            # we also reinitialize random observation process
            #self.init_projection = False
            # initialize the previous action, reward
            rewards = np.zeros(self.num_envs)
            actions = ['reset' for _ in range(self.num_envs)]
            # choose a random time step between 0 and collection_steps where the states
            # need to be remembered
            mem_step = np.random.randint(1, collect_time_steps, 1)
            while self.time_step < collect_time_steps + predict_time_steps:
                # pre-process each image
                states = self.process_observaton(states, states_idx)
                if self.time_step == mem_step:
                    self.targets = states
                    rewards = np.ones(self.num_envs)
                else:
                    rewards = np.zeros(self.num_envs)
                # prepare tensors of last reward
                rewards_input = np.ones((self.num_envs, 100)) * np.tile(np.array([rewards]).T, (1, 100))
                rewards_input = np.expand_dims(rewards_input, axis=0)
                # prepare tensors of last action
                actions_input = np.zeros((self.num_envs, self.action_dim * 100))
                # when the collection time ends, turn action to predict mode
                if self.time_step > collect_time_steps:
                    actions = ['predict' for _ in range(self.num_envs)]
                    # build targets
                    targets = torch.FloatTensor(self.targets).to(device)
                else:
                    targets = torch.zeros(states.shape).to(device)
                for i, a in enumerate(actions):
                    if a == 'reset':
                        actions_input[i] = np.zeros(self.action_dim * 100)
                    elif a == 'predict':
                        actions_input[i] = np.ones(self.action_dim * 100)
                    else:
                        actions_input[i] = np.tile(np.eye(self.action_dim)[a], (1, 100))
                actions_input = np.expand_dims(actions_input, axis=0)
                # convert numpy array to tensor
                states = torch.FloatTensor(np.expand_dims(states, axis=0))
                states = torch.cat((states, torch.FloatTensor(rewards_input)), -1)
                states = torch.cat((states, torch.FloatTensor(actions_input)), -1)
                # generate predictions
                predicts = self.policy(states.to(device))

                if self.time_step > collect_time_steps:
                    loss += np.mean((predicts.detach().cpu().numpy()-targets.detach().cpu().numpy())**2)
                # randomly sample actions for the next step
                actions = np.random.randint(low=0, high=self.action_dim, size=self.num_envs)
                # book-keeping
                test_log['prediction'][-1].append(predicts.detach().cpu().numpy())
                test_log['target'][-1].append(targets.detach().cpu().numpy())
                test_log['step_reward'][-1].append(rewards)
                # move to next step
                self.time_step += 1
                states, _, _, _, _ = self.interface_OAI.step(actions)

            episode += 1
            # extract the trajectory for this episode
            test_log['trajectories'].append(self.interface_OAI.modules_list[0]['spatial_representation'].trajectories)

            if self.network in ['fwm_v1', 'fwm']:
                # extract the memories writing and reading traces for this episode
                memory_trace = self.policy.l1.get_trace()
                # this cubersomeness is just for the convience of data storage
                copy_trace = {}
                for key in list(memory_trace.keys()):
                    copy_trace[key] = np.array(memory_trace[key])
                test_log['memory_traces'].append(copy_trace)

            # clear buffer and trajectories
            self.buffer.clear()
            self.interface_OAI.modules_list[0]['spatial_representation'].clear_trajectories()

            # callback
            # This log is for the callback function only
            logs = {'trial_reward': np.mean(loss), 'trial': episode-1}
            self.engaged_callbacks.on_trial_end(logs)
            print("Episode : {} \t\t Prediction loss : {}".format(episode, np.mean(loss)))

        return test_log

    def process_observaton(self, observations, obs_idx=None):
        if self.process_im == 'RP':
            # convert numpy array to cupy array to use GPU computing
            observations = cp.array(observations)
            # we randomly project the image to a 1d space
            # flatten the observation list into a single vector
            observations = observations.reshape(observations.shape[0], np.prod(observations.shape[1:]))
            # if for the first time, generate a random projection
            if not self.init_projection:
                self.init_projection = True
                # dim_h = observations.shape[1]
                # self.random_projections = np.random.randn(dim_h, self.rp_dim)
                # # normalize the columns of RP matrix to make them unit-length vector
                # for i in range(self.rp_dim):
                #     self.random_projections[:, i] /= np.sqrt(np.sum(self.random_projections[:, i]**2))
                # # convert to cupy array for GPU computation
                # self.random_projections = cp.array(self.random_projections)
                # load existing rp matrix
                self.random_projections = pickle.load(open('/local/xzeng/phd_projects/mann_spatial_learning/experiments/ppo_homecage_5_RP_%s.pkl'%self.rp_dim, 'rb'))
            processed_obs = cp.matmul(observations, self.random_projections)
            # # normalize between 0, 1
            # max_value = cp.max(np.abs(processed_obs), axis=1)
            # processed_obs /= cp.tile(max_value, (self.rp_dim, 1)).T
            # processed_obs = (processed_obs + 1) / 2

        elif self.process_im == 'one_hot':
            processed_obs = np.eye(self.num_states)[obs_idx]

        elif self.process_im == 'r_binary':
            if not self.init_projection:
                # generate a number of num_states of random binary vectors
                self.binary_array = [np.random.choice([0, 1], size=self.args['obs_dim']) for _ in
                                     range(self.num_states)]
                self.binary_array = np.array(self.binary_array)
                self.init_projection = True
            processed_obs = self.binary_array[obs_idx]

        return processed_obs