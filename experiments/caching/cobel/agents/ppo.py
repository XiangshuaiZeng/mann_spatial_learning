import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import os
import time
from datetime import datetime
import gym
import numpy as np
#from cobel.agents.rl_agent import callbacks


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
# Maybe change the buffer variable names and make it similar to those of dyna_dqn etc
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

# same callback as in cobel.agents.rl_agent
# write an own callback function with the right variables, so I don't need two log dictionaries
# I either have to store two log dictionaries or change the log dictionary I want want to print and save in a csv file
# callbacks are monitored with RewardMonitor in cobel->analysis->rl_monitoring->rl_performance_monitors
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

# we use two separate NN for actor and critic. Maybe change it later so that we only use one NN
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # build the actor NN for predicting the policy
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # build the critic NN for predicting the advantage?
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        # calculate the probabilities for choosing a specific action
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, interface_OAI, observations=None, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=80, eps_clip=0.2,
                 has_continuous_action_space=False, action_std_init=0.6, custom_callbacks={}):
        # I should define number_of_states so PPO works also if no observations are given.
        # number_of_states should be defined somewhere in a class where we build the discrete world
        '''# prepare observations
        if observations is None or observations.shape[0] != self.number_of_states:
            # one-hot encoding of states
            self.observations = np.eye(self.number_of_states)
        else:
            self.observations = observations'''

        self.has_continuous_action_space = has_continuous_action_space
        self.state_dim = interface_OAI.observation_space.shape[0]
        if has_continuous_action_space:
            self.action_dim = interface_OAI.action_space.shape[0]
        else:
            self.action_dim = interface_OAI.action_space.n
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.interface_OAI = interface_OAI
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(self.state_dim, self.action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(self.state_dim, self.action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        # initialize a log dictionary to store rewards for steps. Those should also be used for the callbacks, but are not yet
        self.trial_log = {'episode': [], 'time step': [], 'reward': []}
        self.engaged_callbacks = callbacks(self, custom_callbacks)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        # choose an action for a given state
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor. Otherwise PyTorch can't calculate the new parameters for the NN's
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def train(self, number_of_trials=5000, max_number_of_steps=30, action_std_decay_rate=0.05,
              min_action_std=0.1, action_std_decay_freq=2.5e5, checkpoint_path=''):
        save_model_freq = number_of_trials / 5
        log_freq   = max_number_of_steps * 2
        update_timestep = max_number_of_steps * 4

        log_running_reward     = 0
        log_running_episodes   = 0

        time_step              = 0
        i_episode              = 0

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        # Training loop
        while time_step <= number_of_trials:
            # log cumulative reward (those are just for the callback function, so I should try to get rid of it)
            logs = {'trial_reward': 0, 'trial': i_episode, 'trial_session': time_step}
            # reset environment
            state = self.interface_OAI.reset()
            current_ep_reward = 0

            for t in range(1, max_number_of_steps+1):
                # select action with policy
                action = self.select_action(state)
                state, reward, done, _ = self.interface_OAI.step(action)
        
                # saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)
        
                time_step +=1
                current_ep_reward += reward
                # update cumulative reward
                logs['trial_reward'] += current_ep_reward

                # update PPO agent
                if time_step % update_timestep == 0:
                    self.update()

                # if continuous action space; then decay action std of ouput action distribution
                if self.has_continuous_action_space and time_step % action_std_decay_freq == 0:
                    self.decay_action_std(action_std_decay_rate, min_action_std)

                # log in logging file
                if time_step % log_freq == 0:

                    # log average reward til last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    # Probably this could be done in one line, but I don't know how yet
                    self.trial_log['episode'].append(i_episode)
                    self.trial_log['time step'].append(time_step)
                    self.trial_log['reward'].append(log_avg_reward)
                    
                    # print average reward til last episode
                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, log_avg_reward))

                    log_running_reward = 0
                    log_running_episodes = 0

            
                # save model weights
                if checkpoint_path != '' and time_step % save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path + '.pth')
                    self.save(checkpoint_path + '.pth')
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")
            
                # break; if the episode is over
                if done:
                    break

            # callback
            self.engaged_callbacks.on_trial_end(logs)

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")
        return logs

    
    def save_log(self,file_path):
        log_array = np.array([self.trial_log['episode'],self.trial_log['time step'],self.trial_log['reward']])
        log_array = log_array.T
        np.set_printoptions(formatter={'float_kind':'{:.4f}'.format})
        np.savetxt(file_path+'.csv', log_array, delimiter=',', header='episode, time step, reward', fmt='%.4e', comments='')


    def test(self, number_of_trails=100, max_number_of_steps=200, render=False):

        test_running_reward = 0
        print("Testing PPO agent in interface_OAIironment")

        # Test loop
        for ep in range(1, number_of_trails+1):
            ep_reward = 0
            state = self.interface_OAI.reset()
    
            for t in range(1, max_number_of_steps+1):
                action = self.select_action(state)
                state, reward, done, _ = self.interface_OAI.step(action)
                ep_reward += reward

                # render is just needed if we work outside the CoBeL framework with a gym environment
                if render:
                    self.interface_OAI.render(mode = "human")
                    time.sleep(0.01)
        
                if done:
                    break

            # clear buffer    
            self.buffer.clear()

            test_running_reward +=  ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
            ep_reward = 0



        print("============================================================================================")

        avg_test_reward = test_running_reward / number_of_trails
        avg_test_reward = round(avg_test_reward, 2)
        print("average test reward : " + str(avg_test_reward))

        print("============================================================================================")