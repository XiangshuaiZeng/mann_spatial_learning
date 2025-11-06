# basic imports
import os
import argparse, json
import pickle
import numpy as np
import pyqtgraph as pg
import multiprocessing as mp

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

def single_train(network, env, reward_callback, model_path, args):
    # the length of the edge in the topology graph
    step_size = 1.0
    modules_list = []

    for i in range(args['batch_envs']):
        # this is the main window for visual output
        main_window = None
        visualize = False
        # if visual output is required, activate an output window
        if visual_output and i==0:
            main_window = pg.GraphicsWindow(title="Unity demo")
            visualize = True
            # initialize reward monitor
            reward_monitor = RewardMonitor(args['training_episodes'], main_window, visualize, [0, 30])

        # a dictionary that contains all employed modules
        modules = dict()
        # we load a maze
        env_path = os.path.abspath('') + '/../../unity_envs/offline_unity/%s_ss%s_Infos.pkl' % (env, step_size)
        modules['world'] = FrontendUnityOfflineInterface(env_path)
        modules['observation'] = ImageObservationBaseline(modules['world'], main_window, with_GUI=visualize, image_dims=(84,84,3))

        # extract the number of nodes from the world module
        num_states = len(list(modules['world'].env.keys())[:-3])
        num_nodes = num_states // 4
        starts = range(num_nodes)

        if len(args['goals'])==0:
            # if no goals are given, randomly define a new goal in each episode
            goal_node = np.random.randint(0, num_nodes, 1)
            orientations = [0, 90, -90, -180]
            start_ori = None   # so then the start orientation is randomly selected
            goal_ori = np.random.choice(orientations, 1)[0]
        else:
            start_ori = None
            # otherwise, randomly select from one of the given goals
            g_idx = np.random.randint(0, len(args['goals']))
            goal_node = [args['goals'][g_idx][0]]
            goal_ori = args['goals'][g_idx][1]

        modules['spatial_representation'] = FourConnectedGraphRotation(modules,
                                                                       {'start_nodes': starts,'start_ori': start_ori,
                                                                        'goal_nodes': goal_node, 'goal_ori':goal_ori ,
                                                                        'clique_size': 4}, step_size=step_size,
                                                                       act_type=args_dict['act_type'])
        modules['spatial_representation'].set_visual_debugging(visualize, main_window)

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

    # # load the model
    # rl_agent.load(model_path+'_%s'%args['training_episodes'])

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
        main_window = pg.GraphicsWindow(title="Unity demo")
    # a dictionary that contains all employed modules
    modules = dict()
    # we load a maze where all observations have been randomly projected to a 1-D space
    env_path = os.path.abspath('') + '/../../unity_envs/offline_unity/%s_ss%s_Infos.pkl' % (env, step_size)
    modules['world'] = FrontendUnityOfflineInterface(env_path)
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, with_GUI=visual_output, image_dims=(84,84,3))
    # extract the number of nodes from the world module
    num_states = len(list(modules['world'].env.keys())[:-3])
    num_nodes = num_states // 4
    starts = range(num_nodes)

    if len(args['goals']) == 0:
        # if no goals are given, randomly define a new goal in each episode
        goal_node = np.random.randint(0, num_nodes, 1)
        orientations = [0, 90, -90, -180]
        start_ori = None  # so then the start orientation is randomly selected
        goal_ori = np.random.choice(orientations, 1)[0]
    else:
        start_ori = None
        # otherwise, randomly select from one of the given goals
        g_idx = np.random.randint(0, len(args['goals']))
        goal_node = [args['goals'][g_idx][0]]
        goal_ori = args['goals'][g_idx][1]

    modules['spatial_representation'] = FourConnectedGraphRotation(modules,
                                                                   {'start_nodes': starts,'start_ori': start_ori,
                                                                    'goal_nodes': goal_node, 'goal_ori':goal_ori ,
                                                                    'clique_size': 4}, step_size=step_size,
                                                                   act_type=args['act_type'])
    modules['spatial_representation'].set_visual_debugging(visual_output, main_window)

    # top_path = '../unity_envs/offline_unity/%s_ss1.0_Top.pickle' % args['env']
    # modules['spatial_representation'].store_topology(top_path)

    modules_list = [modules]
    # the rl interface which interact with multiple envs
    rl_interface = InterfaceMultiple(modules_list, visual_output, reward_callback)
    # connect the rl interface back to each module
    modules_list[0]['rl_interface'] = rl_interface
    # initialize reward monitor
    reward_monitor = RewardMonitor(args['test_episodes'], main_window, visual_output, [0, 30])
    # initialize RL agent
    if args['agent'] == 'ppo':
        rl_agent = PPO(rl_interface, custom_callbacks={'on_trial_end': [reward_monitor.update]}, network=network,
                       process_im=args['process_im'], args=args)
    elif args['agent'] == 'a2c':
        rl_agent = A2C(rl_interface, custom_callbacks={'on_trial_end': [reward_monitor.update]}, network=network,
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

    def reward_callback(values: dict) -> (float, bool):
        # maximum number of simulation steps
        # retrieve the number of steps
        elaped_steps = values['modules']['rl_interface'].rl_agent.time_step
        graph = values['modules']['spatial_representation']
        reward = -0.01
        end_trial = False

        # if the agent reaches the goal
        if values['current_node'].goal_node and graph.current_ori==graph.goal_ori:
            end_trial = True
            values['modules']['spatial_representation'].goal_reached += 1
            reward = args['reward']

        # force trials to end when the maximum number of steps was reached and reset the goal node
        if (elaped_steps+1) % args['g_change_freq_train'] == 0:
            # end_trial = True
            if len(args['goals'])==0:
                orientations = [0, 90, -90, -180]
                num_nodes = len(values['modules']['spatial_representation'].nodes)
                new_goal_node = np.random.randint(0, num_nodes, 1)
                new_goal_ori = np.random.choice(orientations, 1)[0]
            else:
                # otherwise, randomly select from one of the given goals
                g_idx = np.random.randint(0, len(args['goals']))
                new_goal_node = [args['goals'][g_idx][0]]
                new_goal_ori = args['goals'][g_idx][1]

            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
            # reset the number of goal reachings
            values['modules']['spatial_representation'].goal_reached = 0
            # update the goal location in the monitor
            if values['modules']['spatial_representation'].visual_output:
                values['modules']['spatial_representation'].update_visual_elements()

        return reward, end_trial

    def test_reward_callback(values: dict) -> (float, bool):
        # maximum number of simulation steps
        # retrieve the number of steps
        elaped_steps = values['modules']['rl_interface'].rl_agent.time_step
        graph = values['modules']['spatial_representation']
        reward = -0.01
        end_trial = False
        # if the agent reaches the goal
        if values['current_node'].goal_node and graph.current_ori==graph.goal_ori:
            end_trial = True
            values['modules']['spatial_representation'].goal_reached += 1
            reward = args['reward']

        # force trials to end when the maximum number of steps was reached and reset the goal node
        if (elaped_steps+1) % args['g_change_freq_test'] == 0:
            if len(args['goals']) == 0:
                orientations = [0, 90, -90, -180]
                num_nodes = len(values['modules']['spatial_representation'].nodes)
                new_goal_node = np.random.randint(0, num_nodes, 1)
                new_goal_ori = np.random.choice(orientations, 1)[0]
            else:
                # otherwise, randomly select from one of the given goals
                g_idx = np.random.randint(0, len(args['goals']))
                new_goal_node = [args['goals'][g_idx][0]]
                new_goal_ori = args['goals'][g_idx][1]

            values['modules']['spatial_representation'].reset_goal_nodes(new_goal_node, new_goal_ori)
            # reset the number of goal reachings
            values['modules']['spatial_representation'].goal_reached = 0
            # update the goal location in the monitor
            if values['modules']['spatial_representation'].visual_output:
                values['modules']['spatial_representation'].update_visual_elements()

        return reward, end_trial

    # path to store and load models
    m_path = args['save_dir']+'/saved_models/%s_%s_%s_%s'%(args['agent'], args['env'], args['network'], epoch)
    # train the agent
    logs = single_train(args['network'], args['env'], reward_callback, model_path=m_path, args=args)
    # # store the training data
    pickle.dump(logs, open(args['save_dir']+'/data/training/%s_%s_%s_%s.pkl'%(args['agent'],args['env'], args['network'], epoch), 'wb'))
    ## test the agent
    phases = np.arange(args['save_freq'], args['training_episodes']+1, args['save_freq'])
    for phase in phases:
    #phase = args['training_episodes']
        model_path = m_path + '_%s'%phase
        logs = single_test(args['network'], args['env'], test_reward_callback, model_path=model_path, args=args)
        ##store testing data
        pickle.dump(logs, open(args['save_dir']+ '/data/test/%s_%s_%s_%s_%s.pkl'%(args['agent'], args['env'], args['network'], epoch, phase), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -- in front is for optional arguments in the ArgumentParser
    # environment
    parser.add_argument("--env", help="name of the environment", default='homecage_5')
    parser.add_argument("--difficulty", type=int, default=1,          ### specify the task difficulty
                        help="3 difficulties, 1: fixed start entire training, 2: different start every episode, 3: different start every trial")
    parser.add_argument("--reward", type=float, default=1.0)
    # experiment
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=2.5e-4)
    parser.add_argument("--batch_envs", type=int, help="number of environments in a batch", default=10)
    parser.add_argument("--max_step", type=int, help="length of episode", default=750)
    parser.add_argument("--test_max_step", type=int, help="length of test episode", default=600)
    parser.add_argument("--g_change_freq_train", type=int, default=750//2)
    parser.add_argument("--g_change_freq_test", type=int, default=600 // 2)
    parser.add_argument("--update_length", type=int, default=250)
    parser.add_argument("--training_episodes", type=int, default=3000)
    parser.add_argument("--save_freq", type=int, default=300)
    parser.add_argument("--test_episodes", type=int, default=250)
    parser.add_argument("--save_dir", help="log dir", default='exp_1')   ###### specify where to store the data #######
    parser.add_argument('--keep_hidden', type=bool, help='keep the hidden state between episodes', default=False)
    parser.add_argument("--epochs", type=int, default = 1)
    parser.add_argument("--auxiliary_task", default=None, help='None | predict_g | predict_dis2g')
    parser.add_argument("--act_type", type=str, default = 'egocentric')

    # model
    parser.add_argument("--agent", help="rl agent: ppo or a2c", default='ppo')
    parser.add_argument("--network", help="lstm | fwm | fwm_v1", default='fwm')
    parser.add_argument("--hidden", type=int, help="size of the rnn hidden layer", default=256)
    parser.add_argument("--k_size", type=int, help="size of fw key tensor", default=128)
    parser.add_argument("--v_size", type=int, help="size of fw value tensor", default=128)
    parser.add_argument("--i_size", type=int, help="size of the input tensor", default=256)
    parser.add_argument("--process_im", help="how to process images: RP | vqvae | one_hot | cnn_train", default='RP')
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
    args_dict = {k: argvars[k] for k in argvars if argvars[k] != None}

    ################ start the training and test  #######################
    ##### specify common conditions
    maze_size = 5
    activation = 'relu'
    args_dict['env'] = 'homecage_Jon_%s'%maze_size
    args_dict['process_im'] = 'cnn'
    args_dict['epochs'] = 3
    args_dict['k_activation'] = activation
    args_dict['v_activation'] = activation
    args_dict['weight_l2'] = 0.0  # apply l2 regulerization to all weights
    args_dict['num_nodes'] = maze_size**2
    args_dict['ds_nonlinearity'] = False # if apply non-linear function in the downstream network
    # the number of row, columns and orientation of all states
    args_dict['row_col_ori'] = (maze_size, maze_size, 4)
    # we can also define goals
    args_dict['goals'] = []
    args_dict['act_type'] = 'egocentric'
    ##### specify varying conditions
    auxiliary_tasks = [None]

    # specify the folder
    args_dict['save_dir'] = '%s_%s_kv%s_newFWM_gview'%(args_dict['env'], args_dict['act_type'], activation)
    args_dict['network'] = 'fwm'
    args_dict['auxiliary_task'] = None
    args_dict['goal_encode'] = 'one_hot' # one_hot | continous
    args_dict['goal_loss_coef'] = 0.4
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
    # with mp.Pool() as pool:
    #     items = [pool.apply_async(single_run, (args_dict, epoch)) for epoch in range(1, args_dict['epochs']+1)]
    for epoch in range(1, args_dict['epochs']+1):
        single_run(args_dict, epoch)
