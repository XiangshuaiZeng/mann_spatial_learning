# basic imports
import os
import argparse, json
import pickle
import random
import time
import PyQt5 as qt
import numpy as np
import pyqtgraph as pg

# framework imports
from cobel.frontends.frontends_unity import FrontendUnityOfflineInterface
from four_connected_graph_rotation import FourConnectedGraphRotation
from cobel.observations.image_observations import ImageObservationBaseline
from baseline_multienvs import InterfaceMultiple
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
from agents.ppo import PPO
# pytorch speeding up
import torch
torch.backends.cudnn.benchmark = True

# shall the system provide visual output while performing the experiments?
visual_output = False
# define environments
double_cage = 'double_homecage_shape'
single_cages = ['homecage_square', 'homecage_circle']
test_cages = ['homecage_circle', 'homecage_4_4', 'homecage_3_5', 'homecage_2_6',
              'homecage_1_7', 'homecage_square']
global cage_idx
cage_idx = np.random.choice([0, 1], 1)[0]  # only used in single training scheme
global double_shut  # indicator whether the double maze is disconnected
double_shut = False
double_square_nodes = list(np.arange(0, 25, 1))
double_circle_nodes = list(np.arange(28, 53, 1))
p = -0.03

def single_train(network, reward_callback, model_path, args):
    global cage_idx
    global double_shut
    # the length of the edge in the topology graph
    step_size = 1.0
    modules_list = []
    if args['scheme'] == 'single':
        world_interfaces = [[] for _ in range(args['batch_envs'])]
    for i in range(args['batch_envs']):
        # this is the main window for visual output
        main_window = None
        visualize = False
        # if visual output is required, activate an output window
        if visual_output and i==0:
            main_window = pg.GraphicsWindow(title="training")
            visualize = True
            # initialize reward monitor
            reward_monitor = RewardMonitor(args['training_episodes'], main_window, visualize, [0, 30])
        # a dictionary that contains all employed modules
        modules = dict()
        # we load a maze
        if args['scheme'] == 'double':
            env_path = os.path.abspath('') + '/../../unity_envs/offline_unity/%s_ss%s_Infos.pkl' % (double_cage, step_size)
            modules['world'] = FrontendUnityOfflineInterface(env_path)
        elif args['scheme'] == 'single':
            for cage in single_cages:
                env_path = os.path.abspath('') + '/../../unity_envs/offline_unity/%s_ss%s_Infos.pkl' % (cage, step_size)
                world_interfaces[i].append(FrontendUnityOfflineInterface(env_path))
            # we choose one of the cage for now
            modules['world'] = world_interfaces[i][cage_idx]

        modules['observation'] = ImageObservationBaseline(modules['world'], main_window, with_GUI=visualize, image_dims=(84,84,3))
        # extract the number of nodes from the world module
        num_states = len(list(modules['world'].env.keys())[:-3])
        num_nodes = num_states // 4
        starts = range(num_nodes)
        # define goal node and orientation
        if args['scheme'] == 'double':
            goal_node = np.random.choice(double_square_nodes+double_circle_nodes, size=1)
        else:
            goal_node = np.random.choice(np.arange(num_nodes), size=1)
        orientations = [0, 90, -90, -180]
        start_ori = None   # so then the start orientation is randomly selected
        goal_ori = np.random.choice(orientations, 1)[0]
        modules['spatial_representation'] = FourConnectedGraphRotation(modules,
                                                                       {'start_nodes': starts,'start_ori': start_ori,
                                                                        'goal_nodes': goal_node,
                                                                        'goal_ori':goal_ori ,
                                                                        'clique_size': 4}, step_size=step_size, act_type=args['act_type'])
        if args['scheme'] == 'double':
            # add extra edges to the topology graph
            extra_edges = [(22, 25), (25, 22), (27, 30), (30, 27)]
            # construct the neighborhoods of each node in the graph
            for edge in extra_edges:
                modules['spatial_representation'].edges.append(edge)
                # first and second edge node
                a, b = edge
                modules['spatial_representation'].nodes[a].neighbors.append(
                    modules['spatial_representation'].nodes[b])

        modules['spatial_representation'].set_visual_debugging(visualize, main_window)
        # forward, left and right turn
        if args['act_type'] == 'egocentric':
            modules['spatial_representation'].num_actions = 3
        elif args['act_type'] == 'allocentric':
            modules['spatial_representation'].num_actions = 8
        modules_list.append(modules)

    # the rl interface which interact with multiple envs
    rl_interface = InterfaceMultiple(modules_list, visualize, reward_callback)
    # connect the rl interface back to each module
    for i in range(args['batch_envs']):
        modules_list[i]['rl_interface'] = rl_interface

    if not visual_output:
        reward_monitor = RewardMonitor(args['training_episodes'], main_window, visual_output, [0, 30])

    # initialize RL agent
    rl_agent = PPO(rl_interface, custom_callbacks={'on_trial_end': [reward_monitor.update]}, network=network,
                   process_im=args['process_im'], args=args)
    # eventually, allow the OAI class to access the robotic agent class
    rl_interface.rl_agent = rl_agent
    # # let the agent load the trained model
    # rl_agent.load(model_path+ '_%s'%args['training_episodes'])
    if args['scheme'] == 'single':
        # let the agent learn
        for episode in range(args['training_episodes']):
            rl_agent.train(total_episodes=1, episode_time_steps=args['max_step'],
                                      update_length=args['update_length'])
            # change maze to the other one
            cage_idx = 1 - cage_idx
            for i in range(args['batch_envs']):
                modules_list[i]['world'] = world_interfaces[i][cage_idx]
                modules_list[i]['observation'].world_module = modules_list[i]['world']

    elif args['scheme'] == 'double':
        for episode in range(args['training_episodes']):
            if double_shut:
                # make sure the goal node and the agent's start node are constrained in the
                # same maze at the beginning
                for i in range(args['batch_envs']):
                    if modules_list[i]['spatial_representation'].goal_node in double_square_nodes:
                        modules_list[i]['spatial_representation'].reset_start_nodes(double_square_nodes)
                    elif modules_list[i]['spatial_representation'].goal_node in double_circle_nodes:
                        modules_list[i]['spatial_representation'].reset_start_nodes(double_circle_nodes)
                    if modules_list[i]['spatial_representation'].visual_output:
                        modules_list[i]['spatial_representation'].update_visual_elements()

            rl_agent.train(total_episodes=1, episode_time_steps=args['max_step'],
                           update_length=args['update_length'])
            if not double_shut:
                if random.uniform(0, 1) < args_dict['disconnect_prob']:
                    # disconnect the square and circle maze
                    double_shut = True
                    for i in range(args['batch_envs']):
                        del modules_list[i]['spatial_representation'].edges[-4:]
                        for a in [22, 25, 27, 30]:
                            del modules_list[i]['spatial_representation'].nodes[a].neighbors[-1]
                        if modules_list[i]['spatial_representation'].visual_output:
                            modules_list[i]['spatial_representation'].update_visual_elements()
            else:
                double_shut = False
                for i in range(args['batch_envs']):
                    # if in the previous episode the double maze is disconnected, we connect it again
                    extra_edges = [(22, 25), (25, 22), (27, 30), (30, 27)]
                    for edge in extra_edges:
                        modules_list[i]['spatial_representation'].edges.append(edge)
                        a, b = edge
                        modules_list[i]['spatial_representation'].nodes[a].neighbors.append(
                            modules_list[i]['spatial_representation'].nodes[b])
                    modules_list[i]['spatial_representation'].reset_start_nodes(
                        double_square_nodes+double_circle_nodes)
                    if modules_list[i]['spatial_representation'].visual_output:
                        modules_list[i]['spatial_representation'].update_visual_elements()

    # store the model
    rl_agent.save(model_path+ '_%s'%args['training_episodes'])
    for i in range(args['batch_envs']):
        if args['scheme'] == 'single':
            world_interfaces[i][0].stop_unity()
            world_interfaces[i][1].stop_unity()
        else:
            modules_list[i]['world'].stop_unity()

    return rl_agent.training_log

def single_test(network, reward_callback, model_path, args):
    # the length of the edge in the topology graph
    step_size = 1.0
    modules_list = []
    args['batch_envs'] = len(test_cages)
    for i, env in enumerate(test_cages):
        # this is the main window for visual output
        main_window = None
        visualize = False
        # if visual output is required, activate an output window
        if visual_output and i == 0:
            main_window = pg.GraphicsWindow(title="Test phase")
            visualize = True
            # initialize reward monitor
            reward_monitor = RewardMonitor(args['training_episodes'], main_window, visualize, [0, 30])
        # a dictionary that contains all employed modules
        modules = dict()
        # we load a maze where all observations have been randomly projected to a 1-D space
        env_path = os.path.abspath('') + '/../../unity_envs/offline_unity/%s_ss%s_Infos.pkl' % (env, step_size)
        modules['world'] = FrontendUnityOfflineInterface(env_path)
        modules['observation'] = ImageObservationBaseline(modules['world'], main_window, with_GUI=visualize, image_dims=(84,84,3))

        # extract the number of nodes from the world module
        num_states = len(list(modules['world'].env.keys())[:-3])
        num_nodes = num_states // 4
        starts = range(num_nodes)
        # randomly define a new goal in each episode
        goal_node = np.random.randint(0, num_nodes, 1)
        orientations = [0, 90, -90, -180]
        start_ori = None  # so then the start orientation is randomly selected
        goal_ori = np.random.choice(orientations, 1)[0]

        modules['spatial_representation'] = FourConnectedGraphRotation(modules,
                                                                       {'start_nodes': starts,'start_ori': start_ori,
                                                                        'goal_nodes': goal_node,
                                                                        'goal_ori':goal_ori ,
                                                                        'clique_size': 4}, step_size=step_size, act_type=args['act_type'])
        modules['spatial_representation'].set_visual_debugging(visualize, main_window)

        # top_path = '../unity_envs/offline_unity/%s_ss1.0_Top.pickle' % args['env']
        # modules['spatial_representation'].store_topology(top_path)
        if args['act_type'] == 'egocentric':
            modules['spatial_representation'].num_actions = 3
        elif args['act_type'] == 'allocentric':
            modules['spatial_representation'].num_actions = 8
        modules_list.append(modules)

    # the rl interface which interact with multiple envs
    rl_interface = InterfaceMultiple(modules_list, visualize, reward_callback)
    for i in range(len(test_cages)):
        # connect the rl interface back to each module
        modules_list[i]['rl_interface'] = rl_interface

    if not visual_output:
        reward_monitor = RewardMonitor(args['training_episodes'], main_window, visual_output, [0, 30])
    # initialize RL agent
    rl_agent = PPO(rl_interface, custom_callbacks={'on_trial_end': [reward_monitor.update]}, network=network,
                   process_im=args['process_im'], args=args)
    # eventually, allow the OAI class to access the robotic agent class
    rl_interface.rl_agent = rl_agent
    ######## the real test part ##########
    # let the agent load the trained model
    rl_agent.load(model_path)
    # test the agent's performance, we log the trajectories, rewards on each time step, goal location
    # for each episode, encoded and retrieved memories for each time step, if fwm is used
    test_log = rl_agent.test(total_episodes=args['test_episodes'], episode_time_steps=args['test_max_step'])
    # close all envs
    for i in range(args['batch_envs']):
        modules_list[i]['world'].stop_unity()

    return test_log

def single_run(args, epoch):
    global cage_idx
    def reward_callback(values: dict) -> (float, bool):
        global double_shut
        # based on the trainnig scheme, reward will be different
        scheme = values['modules']['rl_interface'].rl_agent.args['scheme']
        elaped_steps = values['modules']['rl_interface'].rl_agent.time_step
        graph = values['modules']['spatial_representation']

        #if elaped_steps == 0: penalty = -0.1
        #penalty += -0.001
        if scheme == 'double':
            # if in the square maze and the corridor
            if values['current_node'].index < 28:
                penalty = p
                # if the agent is in the square part of the maze and reaches the corner
                # no penalty
                if values['current_node'].index in [0, 4, 20, 24]:
                    penalty = 0
            # if in the circle part of the maze
            elif values['current_node'].index >= 28:
                penalty = p * (25-4)/25
        if scheme == 'single':
            if cage_idx == 0: # if in the square maze
                if values['current_node'].index in [0, 4, 20, 24]:
                    penalty = 0
                else: penalty = p
            elif cage_idx == 1: # if in the circle maze
                penalty = p * (25-4)/25
                # if values['current_node'].index in [12]:
                #     penalty = min(-0.01, penalty/1.5)

        reward = penalty

        end_trial = False
        # if the agent reaches the goal
        if values['current_node'].goal_node: #
            end_trial = True
            values['modules']['spatial_representation'].goal_reached += 1
            reward = args['reward']
            if scheme == 'double' and double_shut:
                if values['modules']['spatial_representation'].goal_node in double_square_nodes:
                    values['modules']['spatial_representation'].reset_start_nodes(double_square_nodes)
                elif values['modules']['spatial_representation'].goal_node in double_circle_nodes:
                    values['modules']['spatial_representation'].reset_start_nodes(double_circle_nodes)
                if values['modules']['spatial_representation'].visual_output:
                    values['modules']['spatial_representation'].update_visual_elements()

        # force trials to end when the maximum number of steps was reached and reset the goal node
        if (elaped_steps+1) % args['g_change_freq_train'] == 0:
            end_trial = True
            orientations = [0, 90, -90, -180]
            num_nodes = len(values['modules']['spatial_representation'].nodes)
            if scheme == 'double':
                new_goal_node = np.random.choice(double_square_nodes + double_circle_nodes, size=1)
            else:
                new_goal_node = np.random.choice(np.arange(num_nodes), size=1)
            new_goal_ori = np.random.choice(orientations, 1)[0]
            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
            # reset the number of goal reachings
            values['modules']['spatial_representation'].goal_reached = 0
            if scheme == 'double' and not double_shut:
                num_states = len(list(values['modules']['world'].env.keys())[:-3])
                num_nodes = num_states // 4
                values['modules']['spatial_representation'].reset_start_nodes(range(num_nodes))
            elif scheme == 'double' and double_shut:
                if values['modules']['spatial_representation'].goal_node in double_square_nodes:
                    values['modules']['spatial_representation'].reset_start_nodes(double_square_nodes)
                elif values['modules']['spatial_representation'].goal_node in double_circle_nodes:
                    values['modules']['spatial_representation'].reset_start_nodes(double_circle_nodes)
            # update the goal location in the monitor
            if values['modules']['spatial_representation'].visual_output:
                values['modules']['spatial_representation'].update_visual_elements()

        return reward, end_trial

    def test_reward_callback(values: dict) -> (float, bool):
        # maximum number of simulation steps
        # retrieve the number of steps
        elaped_steps = values['modules']['rl_interface'].rl_agent.time_step
        graph = values['modules']['spatial_representation']
        reward = p
        end_trial = False

        # if the agent reaches the goal
        if values['current_node'].goal_node:
            end_trial = True
            values['modules']['spatial_representation'].goal_reached += 1
            reward = args['reward']

        # force trials to end when the maximum number of steps was reached and reset the goal node
        if (elaped_steps+1) % args['g_change_freq_test'] == 0:
            end_trial = True
            orientations = [0, 90, -180, -90]
            num_nodes = len(values['modules']['spatial_representation'].nodes)
            new_goal_node = np.random.choice(np.arange(num_nodes), size=1)
            new_goal_ori = np.random.choice(orientations, 1)[0]
            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
            # reset the number of goal reachings
            values['modules']['spatial_representation'].goal_reached = 0
            # update the goal location in the monitor
            if values['modules']['spatial_representation'].visual_output:
                values['modules']['spatial_representation'].update_visual_elements()

        return reward, end_trial

    # path to store and load models
    m_path = args['save_dir']+'/saved_models/%s_%s_%s_%s'%(args['network'], args['process_im'], args['scheme'], epoch)
    # train the agent
    logs = single_train(args['network'], reward_callback, model_path=m_path, args=args)
    pickle.dump(logs, open(args['save_dir']+'/data/training/%s_%s_%s_%s.pkl'%(args['network'], args['process_im'], args['scheme'], epoch), 'wb'))
    # # ## test the agent
    phase = args['training_episodes']
    model_path = m_path + '_%s'%phase
    logs = single_test(args['network'], test_reward_callback, model_path=model_path, args=args)
    ##store testing data
    pickle.dump(logs, open(args['save_dir']+ '/data/test/%s_%s_%s_%s_%s.pkl'%(args['network'], args['process_im'], args['scheme'], epoch, phase), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -- in front is for optional arguments in the ArgumentParser
    # environment
    parser.add_argument("--env", help="name of the environment", default='homecage_5')
    parser.add_argument("--reward", type=float, default=2.0)
    # experiment
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=2.5e-4)
    parser.add_argument("--batch_envs", type=int, help="number of environments in a batch", default=10)
    parser.add_argument("--max_step", type=int, help="length of episode", default=840)
    parser.add_argument("--test_max_step", type=int, help="length of test episode", default=600)
    parser.add_argument("--update_length", type=int, default=280)
    parser.add_argument("--g_change_freq_train", type=int, default=840 // 2)
    parser.add_argument("--g_change_freq_test", type=int, default=600 // 2)
    parser.add_argument("--training_episodes", type=int, default=3500)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--test_episodes", type=int, default=300)
    parser.add_argument("--with_rotation", type=bool, default=True)
    parser.add_argument("--save_dir", help="log dir", default='exp_1')   ###### specify where to store the data #######
    parser.add_argument('--keep_hidden', type=bool, help='keep the hidden state between episodes', default=False)
    parser.add_argument("--epochs", type=int, default = 1)
    parser.add_argument("--a_space", type=str, default='bio', help='bio | full')
    parser.add_argument("--auxiliary_task", default=None, help='None | predict_g | predict_dis2g')
    parser.add_argument("--act_type", type=str, default = 'egocentric')

    # model
    parser.add_argument("--agent", help="rl agent: ppo or a2c", default='ppo')
    parser.add_argument("--network", help="lstm | fwm | fwm_v1", default='fwm')
    parser.add_argument("--hidden", type=int, help="size of the rnn hidden layer", default=256)
    parser.add_argument("--k_size", type=int, help="size of fw key tensor", default=128)
    parser.add_argument("--v_size", type=int, help="size of fw value tensor", default=128)
    parser.add_argument("--i_size", type=int, help="size of the input tensor", default=256)
    parser.add_argument("--process_im", help="how to process images: RP | vqvae | one_hot | cnn", default='RP')
    parser.add_argument("--K_epochs", type=int, help="number of gradient descent for each update", default=2)
    parser.add_argument("--eps_clip", type=float, default=0.1)
    parser.add_argument("--k_activation", type=str, default='tanh', help='activation function for the w and r keys: '
                                                                         'tanh | relu | dpfp')
    parser.add_argument("--n_roll", type=int, default=2, help='used only in dpfp activation function')
    parser.add_argument("--slownet", type=str, default='lstm')

    # other
    parser.add_argument("--entropy_coef", type=float, help="coefficient for the entropy loss", default=0.01)
    parser.add_argument("--value_coef", type=float, help="coefficient for value prediction loss", default=0.2)
    parser.add_argument("--gamma", type=float, help="discounting factor for rewards", default=.95)
    args = parser.parse_args()
    argvars = vars(args)
    args_dict = {k: argvars[k] for k in argvars}

    ################ start the training and test  #######################
    ##### specify common conditions
    args_dict['epochs'] = 5
    activation = 'tanh'
    args_dict['k_activation'] = activation
    args_dict['v_activation'] = activation
    args_dict['ds_nonlinearity'] = False # if apply non-linear function in the downstream network
    args_dict['process_im'] = 'cnn'
    args_dict['network'] = 'fwm'
    args_dict['auxiliary_task'] = None # None, 'predict_dis2g', 'predict_g'
    args_dict['goal_encode'] = 'one_hot'  # one_hot | continous
    args_dict['goal_loss_coef'] = 0.4
    args_dict['num_nodes'] = 53 # always use number of nodes in double env: 25*2+3=53
    args_dict['disconnect_prob'] = 0.4 # the probability that the connection is shut down
                                       # in the double maze, given that the previous episode
                                       # with opening connection
    args_dict['act_type'] = 'egocentric'
    ### specify varying conditions
    training_schemes = ['double']
    ##############################################
    for scheme in training_schemes:
        # specify the folder
        args_dict['save_dir'] = '%s_%s_ds%s_kv%s_newFWM' % (scheme, args_dict['act_type'], args_dict['ds_nonlinearity'], activation)
        args_dict['scheme'] = scheme
        # we first build the corresponding directory for storing models and data
        data_folder = args_dict['save_dir'] + '/data/training'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        data_folder = args_dict['save_dir'] + '/data/test'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        model_folder = args_dict['save_dir'] + '/saved_models'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        # we store all the parameters as a json file
        with open(args_dict['save_dir'] + '/args.txt', 'w') as f:
            json.dump(args_dict, f, indent=2)
    ################# start training ####################
    for epoch in range(6, 9 + 1):
        for scheme in training_schemes:
            args_dict['save_dir'] = '%s_%s_ds%s_kv%s_newFWM' % (scheme, args_dict['act_type'],
                                                                args_dict['ds_nonlinearity'], activation)

            args_dict['scheme'] = scheme
            double_shut = False
            single_run(args_dict, epoch=epoch)
