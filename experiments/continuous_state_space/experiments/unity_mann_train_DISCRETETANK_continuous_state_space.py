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
from DiscreteTank2DEnv import Discrete2DEnv
from cobel.observations.image_observations import ImageObservationBaseline
from baseline_multienvs import InterfaceMultiple
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor
from agents.ppo_DISCRETETANK import PPO
# pytorch speeding up
import torch
torch.backends.cudnn.benchmark = True

# shall the system provide visual output while performing the experiments?
visual_output = True


# Converts node ID into X, Z coords
def node_to_coordinates(node_id, num_nodes, world_limits):
    # Asuming rectangular maze
    grid_size = int(num_nodes ** 0.5)  # Grid size (5x5 for 25 nodes)
    row = node_id // grid_size
    col = node_id % grid_size

    # converst coordinates to x,z from -2 to 2
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
        if visual_output and i==0:
            main_window = pg.GraphicsLayoutWidget(title="Unity demo")
            main_window.show()
            visualize = True
            # initialize reward monitor
            reward_monitor = RewardMonitor(args['training_episodes'], main_window, visualize, [0, 30])

        # a dictionary that contains all employed modules
        modules = dict()
        # we load a maze
        #env_path = os.path.abspath('') + '/../unity_envs/offline_unity/%s_ss%s_Infos.pkl' % (env, step_size)
        #modules['world'] = FrontendUnityOfflineInterface(env_path)
        # Define el path del entorno Unity y el nombre del escenario
        scenario_name = "homecage_redwall"
        env_path = os.path.dirname(os.path.abspath('')) + '/unity_envs/online_unity/{}/unity_envs.x86_64'.format(scenario_name)
        #env_path = os.path.join(os.path.dirname(os.path.abspath('')), 'unity_envs', scenario_name, 'unity_envs.exe')

        # Asegúrate de que el path del entorno Unity está disponible como una variable de entorno
        os.environ['UNITY_ENVIRONMENT_EXECUTABLE'] = env_path
        modules['world'] = FrontendUnityInterface(scenario_name=scenario_name)
        #modules['world'].stop_unity()
        modules['observation'] = ImageObservationBaseline(modules['world'], main_window, with_GUI=visualize, image_dims=(84,84,3))

        # extract the number of nodes from the world module
        #num_states = len(list(modules['world'].env.keys())[:-3])
        num_states = 100 ## TRY TO SET IT NOT MANUALLY -JON
        num_nodes = num_states // 4

        ## EXPLANATION
        if args['difficulty'] < 3:
            starts = [num_nodes//2]
        elif args['difficulty'] == 3:
            starts = range(num_nodes)

        if len(args['goals'])==0:
            # if no goals are given, randomly define a new goal in each episode
            goal_node = np.random.randint(0, num_nodes, 1)
            ## Move the reward in Unity
            world_limits = modules['world'].world_limits + [1.,1.,-1.,-1.]
            coordinates = node_to_coordinates(goal_node, num_nodes, world_limits)
            modules['world'].move_object_only(coordinates[0][0], coordinates[1][0]) ##SETS THE REWARD IN THE STARTING POSITION

            orientations = [0, 90, -90, -180]
            if args['with_rotation']:
                start_ori = None   # so then the start orientation is randomly selected
                goal_ori = np.random.choice(orientations, 1)[0]
            else:
                start_ori = 90
                goal_ori = 90
        else:
            start_ori = None
            # otherwise, randomly select from one of the given goals
            g_idx = np.random.randint(0, len(args['goals']))
            goal_node = [args['goals'][g_idx][0]]
            ## Move the reward in Unity
            world_limits = modules['world'].world_limits + [1., 1., -1., -1.]
            coordinates = node_to_coordinates(goal_node, num_nodes, world_limits)
            modules['world'].move_object_only(coordinates[0][0], coordinates[1][0])
            goal_ori = args['goals'][g_idx][1]

        modules['spatial_representation'] = Discrete2DEnv(modules,
                                            {'start_nodes': starts,'start_ori': start_ori,
                                                'goal_nodes': goal_node,
                                                'goal_ori':goal_ori ,
                                                'clique_size': 4},
                                              wheel_speed=1.0,
                                              track_width=0.5,
                                              dt=0.2
                                               )

        #print(modules['spatial_representation'].graph_info['goal_nodes'])
        modules['spatial_representation'].set_visual_debugging(visualize, main_window)
        if args['with_rotation']:
            if args['a_space'] == 'full':
                modules['spatial_representation'].num_actions = 6
            elif args['a_space'] == 'bio':
                modules['spatial_representation'].num_actions = 3
        else:
            modules['spatial_representation'].num_actions = 4
        modules_list.append(modules)

    # the rl interface which interact with multiple envs
    rl_interface = InterfaceMultiple(modules_list, visualize, reward_callback)

    # connect the rl interface back to each module
    for i in range(args['batch_envs']):
        modules_list[i]['rl_interface'] = rl_interface

    if not visual_output:
        reward_monitor = RewardMonitor(args['training_episodes'], main_window, visual_output, [0, 30])

    # initialize RL agent
    if args['agent'] == 'ppo':
        rl_agent = PPO(rl_interface, custom_callbacks={'on_trial_end': [reward_monitor.update]}, network=network,
                       process_im=args['process_im'], args=args)
    elif args['agent'] == 'a2c':
        rl_agent = A2C(rl_interface, custom_callbacks={'on_trial_end': [reward_monitor.update]}, network=network,
                       process_im=args['process_im'], args=args)
    # eventually, allow the OAI class to access the robotic agent class
    rl_interface.rl_agent = rl_agent

    # let the agent learn
    training_log = rl_agent.train(total_episodes=args['training_episodes'], episode_time_steps=args['max_step'],
                                  update_length=args['update_length'], checkpoint_path=model_path)

    for i in range(args['batch_envs']):
        modules_list[i]['world'].stop_unity()

    return training_log

def single_test(network, env, reward_callback, model_path, args):
    # the length of the edge in the topology graph
    step_size = 1.0
    # this is the main window for visual output
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = pg.GraphicsLayoutWidget(title="Unity demo")
        main_window.show()
    # a dictionary that contains all employed modules
    modules = dict()
    # we load a maze where all observations have been randomly projected to a 1-D space
    env_path = os.path.abspath('') + '/../unity_envs/offline_unity/%s_ss%s_Infos.pkl' % (env, step_size)
    #modules['world'] = FrontendUnityOfflineInterface(env_path)
    scenario_name = "homecage_redwall"
    env_path = os.path.dirname(os.path.abspath('')) + '/unity_envs/online_unity/{}/unity_envs.x86_64'.format(scenario_name)


    # Asegúrate de que el path del entorno Unity está disponible como una variable de entorno
    os.environ['UNITY_ENVIRONMENT_EXECUTABLE'] = env_path
    modules['world'] = FrontendUnityInterface(scenario_name=scenario_name)

    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, with_GUI=visual_output, image_dims=(84,84,3))
    # extract the number of nodes from the world module
    #num_states = len(list(modules['world'].env.keys())[:-3])
    num_states=100
    num_nodes = num_states // 4
    if args['difficulty'] < 3:
        starts = [num_nodes//2]
    elif args['difficulty'] == 3:
        starts = range(num_nodes)

    if len(args['goals']) == 0:
        # if no goals are given, randomly define a new goal in each episode
        goal_node = np.random.randint(0, num_nodes, 1)
        ## Move the reward in Unity
        world_limits = modules['world'].world_limits# + [1., 1., -1., -1.]
        coordinates = node_to_coordinates(goal_node, num_nodes, world_limits)
        modules['world'].move_object_only(coordinates[0][0], coordinates[1][0])
        orientations = [0, 90, -90, -180]
        if args['with_rotation']:
            start_ori = None  # so then the start orientation is randomly selected
            goal_ori = np.random.choice(orientations, 1)[0]
        else:
            start_ori = 90
            goal_ori = 90
    else:
        start_ori = None
        # otherwise, randomly select from one of the given goals
        g_idx = np.random.randint(0, len(args['goals']))
        goal_node = [args['goals'][g_idx][0]]
        ## Move the reward in Unity
        world_limits = modules['world'].world_limits# + [1., 1., -1., -1.]
        coordinates = node_to_coordinates(goal_node, num_nodes, world_limits)
        modules['world'].move_object_only(coordinates[0][0], coordinates[1][0])
        goal_ori = args['goals'][g_idx][1]

    modules['spatial_representation'] = Discrete2DEnv(modules,
                                                      {'start_nodes': starts, 'start_ori': start_ori,
                                                       'goal_nodes': goal_node,
                                                       'goal_ori': goal_ori,
                                                       'clique_size': 4},
                                                      wheel_speed=1.0,
                                                      track_width=0.5,
                                                      dt=0.2
                                                      )
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)

    #top_path = '../unity_envs/offline_unity/%s_ss1.0_Top.pickle' % args['env']
    #modules['spatial_representation'].store_topology(top_path)

    if args['with_rotation']:
        if args['a_space'] == 'full':
            modules['spatial_representation'].num_actions = 6
        elif args['a_space'] == 'bio':
            modules['spatial_representation'].num_actions = 3
    else:
        modules['spatial_representation'].num_actions = 4
    modules_list = [modules]
    # the rl interface which interact with multiple envs
    rl_interface = InterfaceMultiple(modules_list, visual_output, reward_callback)
    # connect the rl interface back to each module
    modules_list[0]['rl_interface'] = rl_interface
    # initialize reward monitor
    reward_monitor = RewardMonitor(args['test_episodes'], main_window, visual_output, [0, 30])
    # initialize RL agent
    #print(args['process_im'])
    if args['agent'] == 'ppo':
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

    # close the env
    modules['world'].stop_unity()

    # if visual output is required, activate an output window
    if visual_output:
        main_window.close()

    return test_log

def single_run(args, epoch):
    visible=False
    def reward_callback(values: dict) -> (float, bool):
        # maximum number of simulation steps
        # retrieve the number of steps
        elaped_steps = values['modules']['rl_interface'].rl_agent.time_step
        graph = values['modules']['spatial_representation']
        reward = -0.01
        end_trial = False
        # # # if a wall is hit, add small penalty
        #if values['hit_wall']:
        #    reward += -0.005

        # if the agent reaches the goal
        if values['modules']['spatial_representation'].is_agent_in_goal(): # and graph.current_ori==graph.goal_ori: ## Change this for removing the goal orientation
            end_trial = True
            values['modules']['spatial_representation'].goal_reached += 1
            reward = args['reward']
            if visible:
                values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(-1, 0)

        # force trials to end when the maximum number of steps was reached and reset the goal node
        if elaped_steps == (args['max_step'] - 1) or elaped_steps == (args['max_step'] - 1)//2:
            end_trial = True
            if len(args['goals'])==0:
                orientations = [0, 90, -90, -180]
                num_states = 100
                num_nodes = num_states // 4
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
            if visible:
                values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(new_goal_node, new_goal_ori)
            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)

            # reset the number of goal reachings
            values['modules']['spatial_representation'].goal_reached = 0
            # update the goal location in the monitor
            #if values['modules']['spatial_representation'].visual_output:
            #    values['modules']['spatial_representation'].update_visual_elements()
        #print(f"Training Step Reward:{epoch} {reward}")ç

        return reward, end_trial

    def test_reward_callback(values: dict) -> (float, bool):
        # maximum number of simulation steps
        # retrieve the number of steps
        elaped_steps = values['modules']['rl_interface'].rl_agent.time_step
        graph = values['modules']['spatial_representation']
        reward = -0.01
        end_trial = False
        retrieving = False
        # # if a wall is hit, add small penalty
        #if values['hit_wall']:
        #    reward += -0.005
        print(values['modules']['spatial_representation'].agent_pose)
        print(values['modules']['spatial_representation'].goal_node)
        # if the agent reaches the goal
        if values['modules']['spatial_representation'].is_agent_in_goal(): ## and graph.current_ori==graph.goal_ori:
            end_trial = True
            retrieving = True
            values['modules']['spatial_representation'].goal_reached += 1
            reward = args['reward']
            if visible:
                values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(-1, 0)

        # force trials to end when the maximum number of steps was reached and reset the goal node
        if (elaped_steps+1) % 250 == 0:
            retrieving = False
            end_trial = True
            if len(args['goals']) == 0:
                orientations = [0, 90, -90, -180]
                num_states = 100
                num_nodes = num_states // 4
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

            if visible:
                values['modules']['spatial_representation'].reset_goal_nodes_Unity_ONLINE(new_goal_node, new_goal_ori)
            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
            if args['difficulty'] == 2:
                # we also reset the start nodes , if difficulty=3, it will be automatically reset in the topoly function
                new_start_node = np.random.randint(0, num_nodes, 1)
                if args['with_rotation']:
                    new_start_ori = np.random.choice(orientations, 1)[0]
                else:
                    new_start_ori = 90
                while new_start_node == new_goal_node:
                    new_start_node = np.random.randint(0, num_nodes, 1)
                values['modules']['spatial_representation'].reset_start_nodes(new_start_node, new_start_ori)
            # reset the number of goal reachings
            values['modules']['spatial_representation'].goal_reached = 0
            # update the goal location in the monitor
            #if values['modules']['spatial_representation'].visual_output:
            #    values['modules']['spatial_representation'].update_visual_elements()
        values['modules']['spatial_representation'].retrieving = retrieving

        return reward, end_trial

    for network in args['models']:
        # path to store and load models
        m_path = args['save_dir']+'/saved_models/%s_%s_%s_%s_no_visible_goal'%(args['agent'], args['env'], network, epoch)
        # train the agent
        logs = single_train(network, args['env'], reward_callback, model_path=m_path, args=args)
        # store the training data
        pickle.dump(logs, open(args['save_dir']+'/data/training/%s_%s_%s_%s_no_visible_goal.pkl'%(args['agent'],args['env'], network, epoch), 'wb'))
        ## test the agent
        # for phase in range(args['save_freq'], args['training_episodes'] + 1, args['save_freq']):
        phase = args['training_episodes']
        model_path = m_path + '_%s'%phase
        logs = single_test(network, args['env'], test_reward_callback, model_path=model_path, args=args)
        # store testing data
        pickle.dump(logs, open(args['save_dir']+ '/data/test/%s_%s_%s_%s_%s_no_visible_goal.pkl'%(args['agent'], args['env'], network, epoch, phase), 'wb'))


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
    parser.add_argument("--max_step", type=int, help="length of episode", default=750)
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
        args_dict['env'] = 'homecage_redwall_DISCRETETANK'
        args_dict['process_im'] = pro_im
        args_dict['save_dir'] = '%s_%s_diff_%s_%s_run2'%(args_dict['env'], args_dict['agent'],
                                                    diff, args_dict['process_im'])

        #args_dict['training_episodes'] = 2500
        #args_dict['test_episodes'] = 300
        #args_dict['save_freq'] = 100

        args_dict['training_episodes'] = 3000
        args_dict['test_episodes'] = 50
        args_dict['save_freq'] = 1000

        #args_dict['models'] = ['lstm']


        args_dict['batch_envs'] = 2 ##
        args_dict['difficulty'] = diff
        args_dict['with_rotation'] = True
        args_dict['a_space'] = 'bio'
        args_dict['epochs'] = 2 ## CHANGE
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
