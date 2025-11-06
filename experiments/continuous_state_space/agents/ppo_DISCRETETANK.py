import torch
import torch.nn as nn
from torchvision import transforms

import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import wandb
import matplotlib.pyplot as plt
import os
import time
import pickle
from datetime import datetime
import gym
import numpy as np
import cupy as cp
from agents.simple_FW import FWMRNN

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

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


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

##########################################
class AgentFWM(nn.Module):
    def __init__(self, p):
        super(AgentFWM, self).__init__()
        self.l1 = FWMRNN(p['in_size'], p['hidden'], p['k_size'], p['v_size'], p['num_reads']).to(device)

        out_size = 2 * p['hidden']

        self.merge = nn.Linear(650, p['in_size']).to(device)
        self.relu = nn.ReLU().to(device)

        self.h_c = nn.Linear(out_size, p['n_actions']).to(device)
        self.h_v = nn.Linear(out_size, 1).to(device)   # here *2 means the output of lstm and fwm are concatanated

        self.p = p
        self.softmax = nn.Softmax(dim=-1)
        self.reset_hidden()
        self.reset_parameters()

    def reset_hidden(self, keep_hidden=False, hiddens=None):
        if not keep_hidden:
            lstm_hidden = (torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device),
                           torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device))
            fwm_hidden = torch.zeros(self.p['batch_size'],
                                     self.p['v_size'],
                                     self.p['k_size']).to(device)
        else:
            if hiddens is None:
                lstm_hidden, fwm_hidden = self.hidden
            else:
                lstm_hidden, fwm_hidden = hiddens
            lstm_hidden = (lstm_hidden[0].detach(), lstm_hidden[1].detach())

            fwm_hidden = fwm_hidden.detach()

        self.hidden = (lstm_hidden, fwm_hidden)

    def reset_parameters(self):
        # reset lstm or fwm
        self.l1.reset_parameters()
        # reset linear head parameters
        nn.init.orthogonal_(self.h_c.weight)
        nn.init.orthogonal_(self.h_v.weight)

    def get_hidden(self):
        return self.hidden

    def clear_trace(self):
        self.l1.fwm.clear_trace()

    def __call__(self, x):
        x = x.to(device)
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        x = self.relu(self.merge(x))
        out, self.hidden = self.l1(x, self.hidden)
        out = out.reshape(-1, out.size(-1))
        act = self.h_c(out)
        val = self.h_v(out)
        # print('value:', val)
        # print('!!!!!!!!!!!!!!!!!!!!')
        act = self.softmax(act)  # needed so that the output of the FWM is between 0 and 1 and sums up to 1
        return act, val

class AgentFWM_CNN(nn.Module):
    def __init__(self, p):
        super(AgentFWM_CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 16, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 2, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        ).to(device)
        self.merge = nn.Linear(656, p['in_size']).to(device)
        self.relu = nn.ReLU().to(device)
        self.l1 = FWMRNN(p['in_size'], p['hidden'], p['k_size'], p['v_size'], p['num_reads']).to(device)

        out_size = 2 * p['hidden']

        self.h_c = nn.Linear(out_size, p['n_actions']).to(device)
        self.h_v = nn.Linear(out_size, 1).to(device)   # here *2 means the output of lstm and fwm are concatanated

        self.p = p
        self.softmax = nn.Softmax(dim=-1)
        self.reset_hidden()
        self.reset_parameters()

    def reset_hidden(self, keep_hidden=False, hiddens=None):
        if not keep_hidden:
            lstm_hidden = (torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device),
                           torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device))
            fwm_hidden = torch.zeros(self.p['batch_size'],
                                     self.p['v_size'],
                                     self.p['k_size']).to(device)
        else:
            if hiddens is None:
                lstm_hidden, fwm_hidden = self.hidden
            else:
                lstm_hidden, fwm_hidden = hiddens
            lstm_hidden = (lstm_hidden[0].detach(), lstm_hidden[1].detach())

            fwm_hidden = fwm_hidden.detach()

        self.hidden = (lstm_hidden, fwm_hidden)

    def reset_parameters(self):
        # reset lstm or fwm
        self.l1.reset_parameters()
        # reset cnn parameters
        for name, p in self.cnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p.data)
        # reset linear head parameters
        nn.init.orthogonal_(self.h_c.weight)
        nn.init.orthogonal_(self.h_v.weight)

    def get_hidden(self):
        return self.hidden

    def clear_trace(self):
        self.l1.fwm.clear_trace()

    def __call__(self, x):
        # x is the tuple (images_batch, actions_batch, rewards_batch)
        # each has the dimension (T, B, C, H, W), (T, B, num_actions), (T, B, 1)
        image_tensors, reward_tensors, action_tensors = x
        image_tensors = image_tensors.to(device)
        reward_tensors = reward_tensors.to(device)
        action_tensors = action_tensors.to(device)
        # reshape the images to be of shape (T*B, C, H, W)
        shape = image_tensors.shape
        image_tensors = image_tensors.view(shape[0]*shape[1], shape[2], shape[3], shape[4])
        latents = self.cnn(image_tensors)
        # reshape the latents back to (T, B, latent_size)
        latents = latents.view(shape[0], shape[1], latents.shape[1])
        # add extra dimension to the latents tensors
        latents = torch.cat((latents, reward_tensors), -1)
        latents = torch.cat((latents, action_tensors), -1)
        lstm_inputs = self.relu(self.merge(latents))
        out, self.hidden = self.l1(lstm_inputs, self.hidden)
        out = out.reshape(-1, out.size(-1))
        act = self.h_c(out)
        val = self.h_v(out)
        act = self.softmax(act)  # needed so that the output of the FWM is between 0 and 1 and sums up to 1
        return act, val


class PPO:
    def __init__(self, interface_OAI, network='fwm', gamma=0.9, process_im='RP', args=None, custom_callbacks={}):

        #####
        self.image_dim = interface_OAI.observation_space.shape
        self.action_dim = interface_OAI.action_space.n
        self.num_envs = interface_OAI.num_envs
        self.network = network

        self.interface_OAI = interface_OAI
        self.lr_actor = args['lr']
        self.gamma = args['gamma']
        self.process_im = process_im
        self.entropy_coff = args['entropy_coef']
        self.val_coff = args['value_coef']
        # the dimension of the inputs to the rnn in the model
        self.rnn_in_dim = args['i_size']

        self.eps_clip = args['eps_clip']
        self.K_epochs = args['K_epochs']
        self.args = args

        if process_im == 'RP':
            self.init_projection = False
            self.rp_dim = 250


        elif process_im == 'one_hot':
            self.num_states = len(list(self.interface_OAI.modules_list[0]['world'].env.keys())[:-3])
            self.obs_dim = self.num_states

        self.buffer = RolloutBuffer()

        p = dict()
        p['batch_size'] = self.num_envs
        p['in_size'] = self.rnn_in_dim
        p['n_actions'] = self.action_dim
        if args is None:
            p['hidden'] = 200
            p['k_size'] = 100
            p['v_size'] = 100
            p['num_reads'] = 1
        else:
            p['hidden'] = args['hidden']
            p['k_size'] = args['k_size']
            p['v_size'] = args['v_size']
            p['num_reads'] = args['num_reads']

        if network == 'fwm':
            if args['process_im'] == 'cnn_train':
                self.policy = AgentFWM_CNN(p)
                self.policy_old = AgentFWM_CNN(p)
            else:
                self.policy = AgentFWM(p)
                self.policy_old = AgentFWM(p)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_actor, eps=1e-5)
        self.mse_loss = nn.MSELoss(reduction='none')

        self.engaged_callbacks = callbacks(self, custom_callbacks)

        self.keep_hidden = args['keep_hidden']

        # wandb, monitor
        self.log_wandb = False
        if self.log_wandb:
            wandb.login(key='943caa2ba669ddf5fd19193829619dc1ee7efb81')
            wandb.init(project='mann_wandb')


    def select_action(self, state):
        # choose an action for a given state
        with torch.no_grad():
            action_probs, state_val = self.policy_old(state)
            dist = Categorical(action_probs)  # same as multinomial (cannot have input =<0)
            action = dist.sample()
            action_logprob = dist.log_prob(action).detach()
            state_val = state_val.detach()

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return list(action.cpu().numpy())

    def update(self, update_length=200):
        # convert list to tensor. Otherwise PyTorch can't calculate the new parameters for the NN's
        if self.args['process_im'] == 'cnn_train':
            # old_states is a list of tuple (images_batch, actions_batch, rewards_batch)
            # each tuple item has the dimension (B, C, H, W), (B, num_actions), (B, 1), tensors
            # the length of list is the num of steps
            ## now, extract the images, actions, rewards
            images, actions, rewards = [], [], []
            for item in self.buffer.states:
                images.append(item[0][0])
                actions.append(item[1][0])
                rewards.append(item[2][0])
            # convert each array to tensors
            images = torch.stack(images, dim=0)
            actions = torch.stack(actions, dim=0)
            rewards = torch.stack(rewards, dim=0)
            old_states = (images, actions, rewards)
        else:
            old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), 1).detach()

        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(
            device)  # shape (num_time_steps, batch_size)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(
            device)  # shape (num_time_steps, batch_size)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(
            device)  # shape (num_time_steps, batch_size)

        # Monte Carlo estimate of returns and advantages
        rewards_to_go = []
        discounted_reward = np.zeros((1, self.num_envs))
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            for batch_j in range(self.num_envs):
                if is_terminal[batch_j]:
                    discounted_reward[0][batch_j] = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)

        rewards_to_go = torch.tensor(np.array(rewards_to_go), dtype=torch.float32).to(
            device)  # shape (num_time_steps, 1, batch_size)
        rewards_to_go = torch.reshape(rewards_to_go, (rewards_to_go.shape[0], rewards_to_go.shape[2])).detach().to(
            device)  # shape (num_time_steps, batch_size)
        # normalizing the rewards (1e-7 to avoid dividing by 0)
        # rewards_to_go = (rewards_to_go-rewards_to_go.mean()) / (rewards_to_go.std() + 1e-7)
        # calculate advantages
        advantages = rewards_to_go.detach() - old_state_values.detach()  # shape (num_time_steps, batch_size)

        # determine each start and end points in the sequence for the updates
        total_length = rewards_to_go.shape[0]
        # Optimize policy with the current segment for K epochs
        for _ in range(self.K_epochs):
            start = 0
            while start < total_length:
                end = min(start + update_length, total_length)
                indices_old = torch.LongTensor(range(0, start))
                indices = torch.LongTensor(range(start, end))
                # reset hidden states of updated policy
                self.policy.reset_hidden(keep_hidden=self.keep_hidden, hiddens=self.old_hiddens)

                if self.args['process_im'] == 'cnn_train':
                    if start > 0:
                        # feed the alreay used segments to policy and get the right hidden states
                        self.policy((old_states[0].index_select(0, indices_old),
                                     old_states[1].index_select(0, indices_old),
                                     old_states[2].index_select(0, indices_old)))
                        # detach the current hiddens
                        self.policy.reset_hidden(keep_hidden=True)
                    # Evaluating old actions and values
                    action_probs, state_values = self.policy((old_states[0].index_select(0, indices),
                                                              old_states[1].index_select(0, indices),
                                                              old_states[2].index_select(0, indices)))
                else:
                    if start > 0:
                        # feed the alreay used segments to policy and get the right hidden states
                        self.policy(old_states.index_select(0, indices_old))
                        # detach the current hiddens
                        self.policy.reset_hidden(keep_hidden=True)
                    # Evaluating old actions and values
                    action_probs, state_values = self.policy(old_states.index_select(0, indices))

                action_probs = torch.reshape(action_probs,
                                             (indices.shape[0], self.num_envs, self.action_dim))
                dist = Categorical(action_probs)

                logprobs = dist.log_prob(old_actions.index_select(0, indices.to(device)))
                dist_entropy = dist.entropy()

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.reshape(state_values, (indices.shape[0], self.num_envs))

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.index_select(0, indices.to(device)).detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages.index_select(0, indices.to(device))  # shape: (num_episode, batch_size)
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) \
                        * advantages.index_select(0, indices.to(device))
                # final loss of clipped objective PPO
                # inverted signs to do gradient descent instead of gradient ascent afterwards
                loss = -torch.min(surr1, surr2) + self.val_coff * self.mse_loss(state_values, rewards_to_go.index_select(0, indices.to(device))) \
                       - self.entropy_coff * dist_entropy  # shape: (num_time_steps, batch_size)
                loss = torch.mean(loss, 1)  # shape: (num_time_steps)

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()

                self.optimizer.step()
                # move to the next segment
                start += update_length

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # we also set the hidden states of the policy_old to those of the online one
        self.policy_old.reset_hidden(keep_hidden=self.keep_hidden, hiddens=self.policy.get_hidden())

        # clear buffer
        self.buffer.clear()

        return loss.detach().cpu().numpy()

    def train(self, total_episodes=1000, episode_time_steps=250, update_length=200, checkpoint_path=''):
        episode = 0
        cumulative_reward = 0

        # track total training time
        start_time = datetime.now().replace(microsecond=0)

        # these logs are saved to analyse the training afterwards
        training_log = {'episode': [], 'episode_time_step': [], 'avg_episode_reward': [],
                        'step_reward': []}
        if self.log_wandb:
            wandb.watch(self.policy, log='all', log_freq=1)
        # Training loop
        print("============================================================================================")
        print("Start training")

        while episode < total_episodes:

            start = time.time()
            self.time_step = 0
            # set up the goal at step 0 in unity
            if self.time_step == 0:
                # self.interface_OAI.modules_list[0]['spatial_representation'].reset_goal_nodes_Unity_ONLINE(
                #    self.interface_OAI.modules_list[0]['spatial_representation'].get_goal(),
                #    0)
                self.interface_OAI.modules_list[0]['spatial_representation'].reset_goal_nodes_Unity_ONLINE(
                    -1,
                    0)
            self.current_ep_reward = 0

            trial_steps = []
            training_log['step_reward'].append([])

            # reset all environments
            states, states_idx = self.interface_OAI.reset()
            # reset / initialize hidden states
            self.policy.reset_hidden(keep_hidden=self.keep_hidden)
            self.policy_old.reset_hidden(keep_hidden=self.keep_hidden)
            self.policy.clear_trace()
            self.policy_old.clear_trace()

            # we also record the hidden states of the policy_old for updating purpose
            if self.keep_hidden:
                self.old_hiddens = self.policy_old.get_hidden()
            else:
                self.old_hiddens = None

            # initialize the previous action, reward, terminal flag
            rewards = np.zeros(self.num_envs)
            actions = ['reset' for _ in range(self.num_envs)]
            dones = np.zeros(self.num_envs)
            while self.time_step < episode_time_steps:
                #################
                # pre-process each image
                states = self.process_observaton(states, states_idx)
                # prepare tensors of last reward
                # rewards_input = np.expand_dims([rewards], axis=-1)
                rewards_input = np.ones((self.num_envs, 100))*np.tile(np.array([rewards]).T,(1, 100))
                rewards_input = np.expand_dims(rewards_input, axis=0)
                # prepare tensors of last action
                actions_input = np.zeros((self.num_envs, self.action_dim*100))
                for i, a in enumerate(actions):
                    if a == 'reset':
                        actions_input[i] = np.zeros(self.action_dim*100)
                    else:
                        actions_input[i] = np.tile(np.eye(self.action_dim)[a], (1, 100))
                actions_input = np.expand_dims(actions_input, axis=0)
                # combine the observation, last reward, last action
                if self.args['process_im'] == 'cnn_train':
                    # convert numpy array to tensor
                    states = (torch.FloatTensor(np.array([states])),
                              torch.FloatTensor(rewards_input),
                              torch.FloatTensor(actions_input))
                else:
                    # convert numpy array to tensor
                    states = torch.FloatTensor(np.expand_dims(states, axis=0))
                    states = torch.cat((states, torch.FloatTensor(rewards_input)), -1)
                    states = torch.cat((states, torch.FloatTensor(actions_input)), -1)

                # select action with policy
                actions = self.select_action(states)

                # at this point, we trigger a reset action in the env where the terminal flag is reached
                reset_idx = np.where(np.array(dones))[0]
                for i in reset_idx:
                    actions[i] = 'reset'

                states, rewards, dones, states_idx, _ = self.interface_OAI.step(actions)
                self.time_step += 1

                # saving reward
                self.buffer.rewards.append(np.array(rewards))
                self.buffer.is_terminals.append(dones)

                # accumulate reward, for printing purpose
                self.current_ep_reward += np.sum(rewards)/self.num_envs

                training_log['step_reward'][-1].append(rewards)

                if rewards[0]>0:
                    trial_steps.append(self.time_step)

            episode += 1
            cumulative_reward += self.current_ep_reward

            # logging
            training_log['episode'].append(episode)
            training_log['avg_episode_reward'].append(self.current_ep_reward)
            training_log['episode_time_step'].append(self.time_step)

            # This log is for the callback function
            logs = {'trial_reward': self.current_ep_reward, 'trial': episode-1}
            self.engaged_callbacks.on_trial_end(logs)
            trial_steps = [0] + trial_steps
            sss = []
            for j in range(1, len(trial_steps)):
                sss.append(trial_steps[j] - trial_steps[j - 1])

            # update the model
            loss = self.update(update_length)
            if self.log_wandb:
                wandb.log({'loss': loss, 'episode reward': self.current_ep_reward})

            print("Episode : {} \t Avg reward : {} \t Loss: {} \t Trial steps of world 1: {}".format(episode, self.current_ep_reward,
                                                                                                 np.mean(loss), sss))
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
        print("A reward of", cumulative_reward, "was reached during", total_episodes, "episodes")
        print("============================================================================================")

        return training_log

    def save(self, checkpoint_path):
        # save the model's parameter
        torch.save(self.policy.state_dict(), checkpoint_path + '.pth')
        # save the random project matrix, if there is one
        if self.process_im == 'RP':
            print('Saving RP')
            pickle.dump(self.random_projections, open(checkpoint_path + '_RP.pkl', 'wb'))

    def load(self, checkpoint_path):
        # load the model's parameters
        self.policy.load_state_dict(torch.load(checkpoint_path+'.pth', map_location=lambda storage, loc: storage))
        self.policy_old.load_state_dict(self.policy.state_dict())
        # load the random project matrix, if there is one
        if self.process_im == 'RP':
            self.random_projections = pickle.load(open(checkpoint_path + '_RP.pkl', 'rb'))
            self.init_projection = True

    def test(self, total_episodes=1000, episode_time_steps=250):
        # these logs are saved to analyse the training afterwards
        test_log = {'step_reward': [], 'goal_location': [], 'trajectories': [],
                    'memory_traces': [], 'values': [], 'retrieving': [], 'reached': [], 'action': []}

        # turn off the training flag of the model
        self.policy.train(False)

        # Test loop
        print("============================================================================================")
        print("Start testing")
        episode = 0
        cumulative_reward = 0
        while episode < total_episodes:
            self.time_step = 0
            # set up the invisible goal in unity
            if self.time_step == 0:
                self.interface_OAI.modules_list[0]['spatial_representation'].reset_goal_nodes_Unity_ONLINE(
                    -1,
                    0)
            self.current_ep_reward = 0
            test_log['step_reward'].append([])
            test_log['goal_location'].append([])
            test_log['retrieving'].append([])
            test_log['reached'].append([])
            test_log['action'].append([])

            trial_steps = []

            # reset environments
            states, states_idx = self.interface_OAI.reset()
            # reset / initialize hidden states
            self.policy.reset_hidden(keep_hidden=self.keep_hidden)
            self.policy_old.reset_hidden(keep_hidden=self.keep_hidden)
            self.policy.clear_trace()
            self.policy_old.clear_trace()

            # initialize the previous action, reward
            rewards = np.zeros(self.num_envs)
            actions = ['reset' for _ in range(self.num_envs)]
            dones = np.zeros(self.num_envs)
            while self.time_step < episode_time_steps:

                goal_n = self.interface_OAI.modules_list[0]['spatial_representation'].goal_node
                test_log['goal_location'][-1].append(
                    [goal_n].copy())
                # pre-process each image
                states = self.process_observaton(states, states_idx)
                # prepare tensors of last reward
                # rewards_input = np.expand_dims([rewards], axis=-1)
                rewards_input = np.ones((self.num_envs, 100)) * np.tile(np.array([rewards]).T, (1, 100))
                rewards_input = np.expand_dims(rewards_input, axis=0)
                # prepare tensors of last action
                actions_input = np.zeros((self.num_envs, self.action_dim * 100))
                for i, a in enumerate(actions):
                    if a == 'reset':
                        actions_input[i] = np.zeros(self.action_dim * 100)
                    else:
                        actions_input[i] = np.tile(np.eye(self.action_dim)[a], (1, 100))
                actions_input = np.expand_dims(actions_input, axis=0)

                # combine the observation, last reward, last action
                if self.args['process_im'] == 'cnn_train':
                    # convert numpy array to tensor
                    states = (torch.FloatTensor(np.array([states])),
                              torch.FloatTensor(rewards_input),
                              torch.FloatTensor(actions_input))
                else:
                    # convert numpy array to tensor
                    states = torch.FloatTensor(np.expand_dims(states, axis=0))
                    states = torch.cat((states, torch.FloatTensor(rewards_input)), -1)
                    states = torch.cat((states, torch.FloatTensor(actions_input)), -1)
                # select action with policy
                actions = self.select_action(states)

                # at this point, we trigger a reset action in the env where the terminal flag is reached
                reset_idx = np.where(np.array(dones))[0]
                for i in reset_idx:
                    actions[i] = 'reset'

                states, rewards, dones, states_idx, _ = self.interface_OAI.step(actions)
                self.time_step += 1
                # step log
                test_log['step_reward'][-1].append(rewards)
                # extract the goal location for each step
                # goal_state = self.interface_OAI.modules_list[0]['spatial_representation'].goal_node
                # test_log['goal_location'][-1].append(self.interface_OAI.modules_list[0]['spatial_representation'].goal_node.copy())
                retrieving = self.interface_OAI.modules_list[0]['spatial_representation'].retrieving
                test_log['reached'][-1].append(self.interface_OAI.modules_list[0][
                                                   'spatial_representation'].goal_reached)  # goal_reached for only one reward #reached for multiple
                test_log['retrieving'][-1].append(
                    self.interface_OAI.modules_list[0]['spatial_representation'].retrieving)
                test_log['action'][-1].append(actions)
                # define orientation of agent
                ori = self.interface_OAI.modules_list[0]['world'].env_data['pose'][2]
                # calculate reward-to-go
                self.current_ep_reward += np.sum(rewards) / self.num_envs

                if rewards[0] > 0:
                    trial_steps.append(self.time_step)

            episode += 1
            cumulative_reward += self.current_ep_reward
            # extract the trajectory for this episode
            test_log['trajectories'].append(self.interface_OAI.modules_list[0]['spatial_representation'].trajectories)
            self.interface_OAI.modules_list[0]['spatial_representation'].clear_trajectories()

            if self.network in ['fwm_v1', 'fwm', 'lstm']:
                # extract the memories writing and reading traces for this episode
                memory_trace = self.policy_old.l1.get_trace()
                # this cubersomeness is just for the convience of data storage
                copy_trace = {}
                for key in list(memory_trace.keys()):
                    copy_trace[key] = np.array(memory_trace[key])
                test_log['memory_traces'].append(copy_trace)

            # also log the state values for each step
            state_values = torch.stack(self.buffer.state_values, dim=0)
            test_log['values'].append(np.copy(state_values.cpu()))

            # clear buffer
            self.buffer.clear()

            trial_steps = [0] + trial_steps
            sss = []
            for j in range(1, len(trial_steps)):
                sss.append(trial_steps[j] - trial_steps[j - 1])

            # callback
            # This log is for the callback function

            logs = {'trial_reward': self.current_ep_reward, 'trial': episode - 1}
            self.engaged_callbacks.on_trial_end(logs)
            print("Episode : {} \t\t Reward : {} \t\t Goal reached at step: {}".format(episode, self.current_ep_reward,
                                                                                       sss))

        # print total test reward
        print("============================================================================================")
        print("A reward of", cumulative_reward, "was reached during", total_episodes, "episodes in the test")
        print("============================================================================================")
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
                self.random_projections = pickle.load(open('../experiments/%s_%s_%s_RP.pkl'%(self.args['agent'],
                                                          'homecage_5', self.network), 'rb'))

            processed_obs = cp.matmul(observations, self.random_projections)


        elif self.process_im == 'one_hot':
            processed_obs = np.eye(self.num_states)[obs_idx]

        elif self.process_im == 'cnn_train':
            # reshape the images to be channel first
            processed_obs = np.moveaxis(observations, -1, 1)

        return processed_obs