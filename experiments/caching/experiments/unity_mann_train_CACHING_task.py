# basic imports
import os
import argparse
import json
import pickle
import numpy as np
import pyqtgraph as pg
import time

# framework imports
from cobel.frontends.frontends_unity import FrontendUnityOfflineInterface, FrontendUnityInterface
from four_connected_graph_rotation import FourConnectedGraphRotation
from cobel.observations.image_observations import ImageObservationBaseline
from baseline_multienvs import InterfaceMultiple
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
from agents.ppo import PPO

# pytorch speeding up
import torch
torch.backends.cudnn.benchmark = True

# shall the system provide visual output while performing the experiments?
visual_output = True


# Converts node ID into X, Z coordinates
def node_to_coordinates(node_id, num_nodes, world_limits):
    # Assuming rectangular maze
    grid_size = int(num_nodes ** 0.5)  # Grid size (5x5 for 25 nodes)
    row = node_id // grid_size
    col = node_id % grid_size

    # convert coordinates to x,z from -2 to 2
    z = world_limits[0] + (world_limits[2] - world_limits[0]) / (grid_size - 1) * col
    x = world_limits[1] + (world_limits[3] - world_limits[1]) / (grid_size - 1) * row

    return x, z


def single_train(network, env, reward_callback, model_path, args):
    # the length of the edge in the topology graph
    step_size = 1.0
    modules_list = []
    print(args)
    for i in range(args['batch_envs']):
        time.sleep(1)
        # this is the main window for visual output
        main_window = None
        visualize = False
        # if visual output is required, activate an output window
        if visual_output and i == 0:
            main_window = pg.GraphicsLayoutWidget(title="Unity demo")
            main_window.show()
            visualize = True
            # initialize reward monitor
            reward_monitor = RewardMonitor(args['training_episodes'], main_window, visualize, [0, 30])

        # a dictionary containing all employed modules
        modules = dict()
        # load the maze
        scenario_name = "homecage_Jon_5_clean"
        env_path = os.path.dirname(os.path.abspath('')) + '/unity_envs/online_unity/{}/unity_envs.x86_64'.format(scenario_name)
        # make sure the Unity environment path is available as an environment variable
        os.environ['UNITY_ENVIRONMENT_EXECUTABLE'] = env_path
        modules['world'] = FrontendUnityInterface(scenario_name=scenario_name)
        modules['observation'] = ImageObservationBaseline(modules['world'], main_window, with_GUI=visualize, image_dims=(84,84,3))

        # extract the number of nodes from the world module
        num_states = 100  # 5x5 grid * 4 orientations = 100
        num_nodes = num_states // 4
        orientations = [0, 90, -90, -180]
        if args['difficulty'] == 3:
            starts = range(num_nodes)
            start_ori = orientations

        if len(args['goals']) == 0:
            # if no goals are given, randomly define a new goal each episode
            goal_node = np.random.randint(0, num_nodes, 1)
            # Move the reward in Unity
            world_limits = modules['world'].world_limits + [1.,1.,-1.,-1.]
            coordinates = node_to_coordinates(goal_node, num_nodes, world_limits)
            modules['world'].move_object_only(coordinates[0][0], coordinates[1][0])  # sets the reward in the starting position

            if args['with_rotation']:
                # start orientation is randomly selected
                goal_ori = np.random.choice(orientations, 1)[0]
            else:
                start_ori = 90
                goal_ori = 90
        else:
            start_ori = None
            # otherwise, randomly select one of the provided goals
            g_idx = np.random.randint(0, len(args['goals']))
            goal_node = [args['goals'][g_idx][0]]

            # Move the reward in Unity
            world_limits = modules['world'].world_limits + [1., 1., -1., -1.]
            coordinates = node_to_coordinates(goal_node, num_nodes, world_limits)
            modules['world'].move_object_only(coordinates[0][0], coordinates[1][0])
            goal_ori = args['goals'][g_idx][1]

        modules['spatial_representation'] = FourConnectedGraphRotation(modules,
                                                                       {'start_nodes': starts, 'start_ori': start_ori,
                                                                        'goal_nodes': goal_node,
                                                                        'goal_ori': goal_ori,
                                                                        'clique_size': 4}, step_size=step_size)

        modules['spatial_representation'].set_visual_debugging(visualize, main_window)
        if args['with_rotation']:
            if args['a_space'] == 'full':
                modules['spatial_representation'].num_actions = 6
            elif args['a_space'] == 'bio':
                modules['spatial_representation'].num_actions = 3
            elif args['a_space'] == 'flap':
                modules['spatial_representation'].num_actions = 4
        else:
            modules['spatial_representation'].num_actions = 4
        modules_list.append(modules)

    # RL interface interacting with multiple environments
    rl_interface = InterfaceMultiple(modules_list, visualize, reward_callback)

    # connect RL interface back to each module
    for i in range(args['batch_envs']):
        modules_list[i]['rl_interface'] = rl_interface

    if not visual_output:
        reward_monitor = RewardMonitor(args['training_episodes'], main_window, visual_output, [0, 30])

    # initialize RL agent
    if args['agent'] == 'ppo':
        rl_agent = PPO(rl_interface, custom_callbacks={'on_trial_end': [reward_monitor.update]}, network=network,
                       process_im=args['process_im'], args=args)

    # allow OAI class to access the robotic agent class
    rl_interface.rl_agent = rl_agent

    # start training
    training_log = rl_agent.train(total_episodes=args['training_episodes'], episode_time_steps=args['max_step'],
                                  update_length=args['update_length'], checkpoint_path=model_path)

    for i in range(args['batch_envs']):
        modules_list[i]['world'].stop_unity()

    return training_log


def single_test(network, env, reward_callback, model_path, args):
    # the length of the edge in the topology graph
    step_size = 1.0
    main_window = None
    # if visual output is required, activate a window
    if visual_output:
        main_window = pg.GraphicsLayoutWidget(title="Unity demo")
        main_window.show()

    modules = dict()
    # load the maze
    scenario_name = "homecage_Jon_5_clean"
    env_path = os.path.dirname(os.path.abspath('')) + '/unity_envs/online_unity/{}/unity_envs.x86_64'.format(scenario_name)
    os.environ['UNITY_ENVIRONMENT_EXECUTABLE'] = env_path
    modules['world'] = FrontendUnityInterface(scenario_name=scenario_name)
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, with_GUI=visual_output, image_dims=(84,84,3))

    num_states = 100  # 5x5 grid * 4 orientations = 100
    num_nodes = num_states // 4
    orientations = [0, 90, -90, -180]
    if args['difficulty'] < 3:
        starts = [num_nodes // 2]
    elif args['difficulty'] == 3:
        starts = range(num_nodes)
        start_ori = orientations

    if len(args['goals']) == 0:
        goal_node = np.random.randint(0, num_nodes, 1)
        # Move the reward in Unity
        world_limits = modules['world'].world_limits + [1., 1., -1., -1.]
        coordinates = node_to_coordinates(goal_node, num_nodes, world_limits)
        modules['world'].move_object_only(coordinates[0][0], coordinates[1][0])

        if args['with_rotation']:
            goal_ori = np.random.choice(orientations, 1)[0]
        else:
            start_ori = 90
            goal_ori = 90
    else:
        start_ori = None
        g_idx = np.random.randint(0, len(args['goals']))
        goal_node = [args['goals'][g_idx][0]]
        world_limits = modules['world'].world_limits + [1., 1., -1., -1.]
        coordinates = node_to_coordinates(goal_node, num_nodes, world_limits)
        modules['world'].move_object_only(coordinates[0][0], coordinates[1][0])
        goal_ori = args['goals'][g_idx][1]

    modules['spatial_representation'] = FourConnectedGraphRotation(modules,
                                                                   {'start_nodes': starts, 'start_ori': start_ori,
                                                                    'goal_nodes': goal_node,
                                                                    'goal_ori': goal_ori,
                                                                    'clique_size': 4}, step_size=step_size)
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)

    if args['with_rotation']:
        if args['a_space'] == 'full':
            modules['spatial_representation'].num_actions = 6
        elif args['a_space'] == 'bio':
            modules['spatial_representation'].num_actions = 3
        elif args['a_space'] == 'flap':
            modules['spatial_representation'].num_actions = 4
    else:
        modules['spatial_representation'].num_actions = 4

    modules_list = [modules]
    rl_interface = InterfaceMultiple(modules_list, visual_output, reward_callback)
    modules_list[0]['rl_interface'] = rl_interface

    reward_monitor = RewardMonitor(args['test_episodes'], main_window, visual_output, [0, 30])

    if args['agent'] == 'ppo':
        rl_agent = PPO(rl_interface, custom_callbacks={'on_trial_end': [reward_monitor.update]}, network=network,
                       process_im=args['process_im'], args=args)

    rl_interface.rl_agent = rl_agent

    # load the trained model
    rl_agent.load(model_path)
    # test the agent’s performance and log trajectories
    test_log = rl_agent.test(total_episodes=args['test_episodes'], episode_time_steps=args['test_max_step'])
    modules['world'].stop_unity()

    if visual_output:
        main_window.close()

    return test_log

def single_run(args, epoch):
    def reward_callback(values: dict) -> (float, bool):
        # the number of step
        elaped_steps = values['modules']['rl_interface'].rl_agent.time_step
        graph = values['modules']['spatial_representation']
        reward = -0.01
        end_trial = False
        env_id = values['num_envs_id']
        action= values['actions']

        num_nodes=len(values['modules']['spatial_representation'].nodes)

        # adjusts visible goal
        if elaped_steps==0:
            values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(values['modules']['spatial_representation'].goal_node,
                                                                                      0)
        # if the agent caches the goal
        if (values['current_node'].goal_node) and reached[env_id] != 1 and action==3:
            current_goal = values['modules']['spatial_representation'].current_node
            reached[env_id] = reached[env_id] + 1
            end_trial = True
            # Reached goals list, in case we want to multiple goal caching in each phase
            list_reached_goals[env_id].append(
                int(current_goal.item()) if isinstance(current_goal, np.ndarray) else current_goal)
            #change position reward
            if len(args['goals'])==0:
                orientations = [0, 90, -90, -180]
                num_nodes = len(values['modules']['spatial_representation'].nodes)
                new_goal_node = np.random.randint(0, num_nodes, 1)

                if args['with_rotation']:
                    new_goal_ori = np.random.choice(orientations, 1)[0]
                else:
                    new_goal_ori = 90
            else:
                # otherwise, randomly select from one of the given goals
                g_idx = np.random.randint(0, len(args['goals']))
                new_goal_node = [args['goals'][g_idx][0]]
                new_goal_ori = args['goals'][g_idx][1]

            # It changes the goals in unity online env
            values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(new_goal_node, new_goal_ori)
            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
            reward = args['reward']
            values['modules']['spatial_representation'].goal_reached += 1
            if reached[env_id]==1:
                values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(-1, 0)
                values['modules']['spatial_representation'].reset_goal_nodes(list_reached_goals[env_id], new_goal_ori)
                end_trial = True
                discarted[env_id]  = True

        elif reached[env_id]==1:
            values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(-1, 0)
            timer[env_id] +=1
            if len(list_reached_goals[env_id])>0:
                if ((values['modules']['spatial_representation'].current_node) == list_reached_goals[env_id][0]) and action==3:
                    #add reward
                    reward = args['reward']
                    end_trial = True
                    list_reached_goals[env_id].remove(values['modules']['spatial_representation'].current_node)
                    values['modules']['spatial_representation'].reset_goal_nodes(list_reached_goals[env_id],
                                                                                 0)
                    values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(-1,
                                                                                              0)

                    if len(list_reached_goals[env_id])==0:
                        reached[env_id] = 0
                        # change position reward
                        if len(args['goals']) == 0:
                            orientations = [0, 90, -90, -180]
                            num_nodes = len(values['modules']['spatial_representation'].nodes)
                            new_goal_node = np.random.randint(0, num_nodes, 1)

                            if args['with_rotation']:
                                new_goal_ori = np.random.choice(orientations, 1)[0]
                            else:
                                new_goal_ori = 90
                        values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
                        values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(new_goal_node,
                                                                                                  new_goal_ori)

                        timer[env_id] = 0

                        end_trial = True

        # updating values to adjust unity reward visualization
        values['modules']['spatial_representation'].list_reached = list_reached_goals
        values['modules']['spatial_representation'].env_id = env_id
        values['modules']['spatial_representation'].discarted = discarted[env_id]

        # force trials to end when the maximum number of steps was reached and reset the goal node
        if elaped_steps == (args['max_step'] - 1):
            list_reached_goals[env_id].clear()
            reached[env_id]=0
            discarted[env_id] = False
            end_trial = True
            if len(args['goals']) == 0:
                orientations = [0, 90, -90, -180]
                num_nodes = len(values['modules']['spatial_representation'].nodes)
                new_goal_node = np.random.randint(0, num_nodes, 1)

                if args['with_rotation']:
                    new_goal_ori = np.random.choice(orientations, 1)[0]
                else:
                    new_goal_ori = 90
            else:
                # otherwise, randomly select from one of the given goals
                g_idx = np.random.randint(0, len(args['goals']))
                new_goal_node = [args['goals'][g_idx][0]]
                new_goal_ori = args['goals'][g_idx][1]

            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
            #values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(new_goal_node, new_goal_ori)

            # reset the number of goal reachings
            values['modules']['spatial_representation'].goal_reached = 0

        if values['modules']['spatial_representation'].visual_output:
            values['modules']['spatial_representation'].update_visual_elements()
        return reward, end_trial

    def test_reward_callback(values: dict) -> (float, bool):
        # maximum number of simulation steps
        elaped_steps = values['modules']['rl_interface'].rl_agent.time_step
        reward = -0.01
        end_trial = False
        retrieving = False
        action = values['actions']
        env_id = values['num_envs_id']
        # adjusts visible goal
        if elaped_steps==0:
            values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(values['modules']['spatial_representation'].goal_node,
                                                                                      0)
        # if the agent caches the goal
        if (values['current_node'].goal_node) and reached[
            env_id] != 1 and action==3:
            current_goal = values['modules']['spatial_representation'].current_node
            reached[env_id] = reached[env_id] + 1
            end_trial = True
            # Reached goals list
            list_reached_goals[env_id].append(
                int(current_goal.item()) if isinstance(current_goal, np.ndarray) else current_goal)
            # changes reward position
            if len(args['goals']) == 0:
                orientations = [0, 90, -90, -180]
                num_nodes = len(values['modules']['spatial_representation'].nodes)
                new_goal_node = np.random.randint(0, num_nodes, 1)

                if args['with_rotation']:
                    new_goal_ori = np.random.choice(orientations, 1)[0]
                else:
                    new_goal_ori = 90
            else:
                # otherwise, randomly select from one of the given goals
                g_idx = np.random.randint(0, len(args['goals']))
                new_goal_node = [args['goals'][g_idx][0]]
                new_goal_ori = args['goals'][g_idx][1]

            # It changes the goals in unity online env
            values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(new_goal_node, new_goal_ori)
            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
            reward = args['reward']
            values['modules']['spatial_representation'].goal_reached += 1
            if reached[env_id] == 1:
                # make goal disappear
                values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(-1, 0)
                values['modules']['spatial_representation'].reset_goal_nodes(list_reached_goals[env_id], new_goal_ori)
                # reset agent position
                end_trial = True
                discarted[env_id] = True

        elif reached[env_id] == 1:
            # makes goal invisible
            values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(-1, 0)
            retrieving = True
            timer[env_id] += 1
            if len(list_reached_goals[env_id]) > 0:
                if ((values['modules']['spatial_representation'].current_node) == list_reached_goals[env_id][0]) and action==3:
                    reward = args['reward']
                    end_trial = True
                    list_reached_goals[env_id].remove(values['modules']['spatial_representation'].current_node)
                    values['modules']['spatial_representation'].reset_goal_nodes(list_reached_goals[env_id],
                                                                                 0)
                    values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(-1, 0)

                    if len(list_reached_goals[env_id]) == 0:
                        retrieving = False
                        reached[env_id] = 0
                        # change position reward
                        if len(args['goals']) == 0:
                            orientations = [0, 90, -90, -180]
                            num_nodes = len(values['modules']['spatial_representation'].nodes)
                            new_goal_node = np.random.randint(0, num_nodes, 1)

                            if args['with_rotation']:
                                new_goal_ori = np.random.choice(orientations, 1)[0]
                            else:
                                new_goal_ori = 90
                        values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
                        values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(new_goal_node,
                                                                                                  new_goal_ori)
                        timer[env_id] = 0
                        end_trial = True

        # updating values for adjsting unity reward visualization
        values['modules']['spatial_representation'].list_reached = list_reached_goals
        values['modules']['spatial_representation'].env_id = env_id
        values['modules']['spatial_representation'].discarted = discarted[env_id]
        values['modules']['spatial_representation'].retrieving=retrieving
        values['modules']['spatial_representation'].reached = reached[env_id]

        # force trials to end when the maximum number of steps was reached and reset the goal node
        if elaped_steps == (args['test_max_step'] - 1):
            list_reached_goals[env_id].clear()
            reached[env_id] = 0
            discarted[env_id] = False
            end_trial = True
            if len(args['goals']) == 0:
                orientations = [0, 90, -90, -180]
                num_nodes = len(values['modules']['spatial_representation'].nodes)
                new_goal_node = np.random.randint(0, num_nodes, 1)

                if args['with_rotation']:
                    new_goal_ori = np.random.choice(orientations, 1)[0]
                else:
                    new_goal_ori = 90
            else:
                # otherwise, randomly select from one of the given goals
                g_idx = np.random.randint(0, len(args['goals']))
                new_goal_node = [args['goals'][g_idx][0]]
                new_goal_ori = args['goals'][g_idx][1]

            # It changes the goals in unity online env
            #values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(new_goal_node, new_goal_ori)
            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
            # reset the number of goal reachings
            values['modules']['spatial_representation'].goal_reached = 0

        if values['modules']['spatial_representation'].visual_output:
            values['modules']['spatial_representation'].update_visual_elements()

        return reward, end_trial

    for network in args['models']:
        # path to store and load models
        m_path = args['save_dir']+'/saved_models/%s_%s_%s_%s_v3_cach_action_new2_4000'%(args['agent'], args['env'], network, epoch)
        # train the agent
        logs = single_train(network, args['env'], reward_callback, model_path=m_path, args=args)
        # store the training data
        pickle.dump(logs, open(args['save_dir']+'/data/training/%s_%s_%s_%s_4000_v3_cach_action_new2.pkl'%(args['agent'],args['env'], network, epoch), 'wb'))
        ## test the agent
        # for phase in range(args['save_freq'], args['training_episodes'] + 1, args['save_freq']):
        phase = args['training_episodes']
        model_path = m_path + '_%s' % phase
        # to use a trained model add the .pth here
        #model_path= "/local/jon/repositories/mann_spatial_learning-main_OLD/experiments/new_homecage_Jon_5_caching_v3_ppo_diff_3_cnn_train_run2/saved_models/ppo_homecage_Jon_5_caching_v3_fwm_1_v3_cach_action_new_6000"
        logs = single_test(network, args['env'], test_reward_callback, model_path=model_path, args=args)
        # store testing data
        pickle.dump(logs, open(args['save_dir']+ '/data/test/%s_%s_%s_%s_%s_50ep_4000_v3_cach_action_test.pkl'%(args['agent'], args['env'], network, epoch, phase), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -- in front is for optional arguments in the ArgumentParser
    # environment
    parser.add_argument("--env", help="name of the environment", default='homecage_5_V9')
    parser.add_argument("--difficulty", type=int, default=1,          ### specify the task difficulty
        help="3 difficulties, 1: fixed start entire training, 2: different start every episode, 3: different start every trial")
    parser.add_argument("--reward", type=float, default=1.0)
    # experiment
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=2.5e-4)
    parser.add_argument("--batch_envs", type=int, help="number of environments in a batch", default=10)
    parser.add_argument("--max_step", type=int, help="length of episode", default=1000)
    parser.add_argument("--update_length", type=int,
                        help="length of segments used for update", default=250)
    parser.add_argument("--test_max_step", type=int, help="length of test episode", default=1000)
    parser.add_argument("--training_episodes", type=int, default=2500) ####CAMBIAR
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--test_episodes", type=int, default=250)
    parser.add_argument("--with_rotation", type=bool, help="whether have rotation in the action", default=True)
    parser.add_argument("--save_dir", help="log dir", default='exp_1')   ###### specify where to store the data #######
    parser.add_argument('--keep_hidden', type=bool, help='keep the hidden state between episodes', default=False)
    parser.add_argument("--epochs", type=int, default = 1)
    parser.add_argument("--a_space", type=str, default='bio', help='bio | full')

    # model
    parser.add_argument("--agent", help="rl agent: ppo or a2c", default='ppo')
    parser.add_argument("--models", help="model: lstm and fwm", default=['fwm'])
    parser.add_argument("--hidden", type=int, help="size of the lstm (hidden) layer", default=256)
    parser.add_argument("--k_size", type=int, help="size of fw key tensor", default=128)
    parser.add_argument("--v_size", type=int, help="size of fw value tensor", default=128)
    parser.add_argument("--i_size", type=int, help="size of the input tensor", default=256)
    parser.add_argument("--process_im", help="how to process images: RP | vqvae | one_hot | cnn_train", default='RP')
    parser.add_argument("--K_epochs", type=int, help="number of gradient descent for each update", default=2)
    parser.add_argument("--eps_clip", type=float, default=0.1)
    parser.add_argument("--num_reads", type=int, default=1, help='number of reads in fwm')


    # other
    parser.add_argument("--entropy_coef", type=float, help="coefficient for the entropy loss", default=0.01)
    parser.add_argument("--value_coef", type=float, help="coefficient for value prediction loss", default=0.2)
    parser.add_argument("--gamma", type=float, help="gammaR: discounting factor for rewards", default=.95)

    args = parser.parse_args()
    argvars = vars(args)
    args_dict = {k: argvars[k] for k in argvars if argvars[k] != None}

    ################ start the training and test for multiple times #######################
    for pro_im in ['cnn_train']:
        diff = 3
        # specify the folder and difficulty
        args_dict['env'] = 'homecage_Jon_5_caching_v3'
        args_dict['process_im'] = pro_im
        args_dict['save_dir'] = 'new_%s_%s_diff_%s_%s_run2'%(args_dict['env'], args_dict['agent'],
                                                    diff, args_dict['process_im'])

        args_dict['training_episodes'] = 4000 # training episodes
        args_dict['test_episodes'] = 50       # test episodes
        args_dict['save_freq'] = 1000         # save frequency

        args_dict['batch_envs'] = 2          # number of blocks
        num_envs = args_dict['batch_envs']
        list_reached_goals = [[] for _ in range(num_envs)]
        reached = [0 for _ in range(num_envs)]
        timer = [0 for _ in range(num_envs)]
        last_value = [[] for _ in range(num_envs)]
        discarted = [False] * num_envs
        retrieving = False

        args_dict['difficulty'] = diff
        args_dict['with_rotation'] = True
        args_dict['a_space'] = 'flap'
        args_dict['epochs'] = 3
        # we can also define goals
        args_dict['goals'] = []
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
        # start training
        for epoch in range(1, args_dict['epochs']+1):
            single_run(args_dict, epoch=epoch)
