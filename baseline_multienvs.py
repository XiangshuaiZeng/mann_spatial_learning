# basic imports
import numpy as np


class InterfaceMultiple():

    def __init__(self, modules_list: list, with_GUI=True, reward_callback=None):
        '''
        This is the Open AI gym interface class. The interface wraps the control path and ensures communication between the agent and the environment.
        The class descends from gym.Env, and is designed to be minimalistic (currently!).
        This class is able to communicate with multiple same envs.
        Currently, the envs have to be offline

        Parameters
        ----------
        modules_list :                      A list that contains multiple framework modules.
        with_GUI :                          If true, observations and policy will be visualized.on.

        Returns
        ----------
        None
        '''
        self.modules_list = modules_list
        # memorize the reward callback function
        self.reward_callback = reward_callback
        self.num_envs = len(modules_list)

        self.worlds = [self.modules_list[i]['world'] for i in range(self.num_envs)]
        # retrieve action space
        self.action_space = modules_list[0]['spatial_representation'].get_action_space()
        # retrieve observation space
        self.observation_space = modules_list[0]['observation'].get_observation_space()

        self.observations = np.zeros([self.num_envs] + list(self.observation_space.shape))

        self.rl_agent = None

    def step(self, action_list: list):
        '''
        AI Gym's step function.
        Executes the agent's action and propels the simulation.

        Parameters
        ----------
        action :                            The action selected by the agent.

        Returns
        ----------
        observations :                       A list of observations of the new current state in each env.
        rewards :                            A list of rewards retrieved.
        end_trials :                         A list of flags indicating whether the trial ended.
        logs :                              The (empty) logs dictionary.
        '''
        rewards, end_trials = [], []
        obs_idx = []
        for i in range(self.num_envs):
            callback_value = self.modules_list[i]['spatial_representation'].generate_behavior_from_action(action_list[i])
            callback_value['rlAgent'] = self.rl_agent
            callback_value['modules'] = self.modules_list[i]
            callback_value['env_idx'] = i
            reward, end_trial = self.reward_callback(callback_value)
            rewards.append(reward)
            end_trials.append(end_trial)
            self.observations[i] = np.copy(self.modules_list[i]['observation'].observation)
            pose = self.modules_list[i]['world'].env_data['pose']
            node, ori = pose[1], pose[2]
            idx = int(node*4 + int(pose[2]/90) % 4)
            obs_idx.append(idx)

        return self.observations, rewards, end_trials, obs_idx, {}

    def reset(self) -> np.ndarray:
        '''
        AI Gym's reset function.
        Resets the environment and the agent's state.

        Parameters
        ----------

        Returns
        ----------
        observation :                       The observation of the new current state.
        '''
        obs_idx = []
        for i in range(self.num_envs):
            self.modules_list[i]['spatial_representation'].generate_behavior_from_action('reset')
            self.observations[i] = np.copy(self.modules_list[i]['observation'].observation)
            pose = self.modules_list[i]['world'].env_data['pose']
            node, ori = pose[1], pose[2]
            idx = int(node * 4 + int(pose[2] / 90) % 4)
            obs_idx.append(idx)

        return self.observations, obs_idx

    def get_position(self) -> np.ndarray:
        '''
        This function returns the agent's position in the environment.

        Parameters
        ----------
        None

        Returns
        ----------
        position :                          Numpy array containing the agent's position.
        '''
        current_nodes = [self.modules_list[i]['spatial_representation'].nodes[self.current_node]
                         for i in range(self.num_envs)]
        nodes_idx = [self.modules_list[i]['spatial_representation'].current_node for i in range(self.num_envs)]
        positions = [[current_nodes[i].x, current_nodes[i].y] for i in range(self.num_envs)]

        return np.array(positions), np.array(nodes_idx)

    def get_goals_idx(self) -> list:
        '''
        This function returns the goal location.
        '''
        goals = []
        for i in range(self.num_envs):
            g = self.modules_list[i]['spatial_representation'].get_goal()
            goals.append(g[0]*4+(g[1]//90)%4)
        return goals