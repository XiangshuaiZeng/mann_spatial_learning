# basic imports
import PyQt5 as qt
import pyqtgraph as pg
import pyqtgraph.functions
import numpy as np
import pickle
import gym
from math import *
# framework imports
from cobel.spatial_representations.topology_graphs.misc.topology_node import TopologyNode
from cobel.spatial_representations.topology_graphs.misc.cog_arrow import CogArrow
from cobel.spatial_representations.topology_graphs.misc.utils import lineseg_dists, point_in_polygon, doIntersect, Point
from cobel.spatial_representations.spatial_representation import SpatialRepresentation

def if_intersect_multi(single_line, line_group):
    '''
    check if one line has intersection to any of a group of lines

    Parameters
    ----------
    single_line:                       a list/array containing the XY coordinates of one line
    line_group:                        a list/array of lines
    '''
    p1 = Point(single_line[0][0], single_line[0][1])
    q1 = Point(single_line[1][0], single_line[1][1])
    for line in line_group:
        p2 = Point(line[0][0], line[0][1])
        q2 = Point(line[1][0], line[1][1])
        if doIntersect(p1, q1, p2, q2):
            return True
    return False


class FourConnectedGraphRotation(SpatialRepresentation):

    def __init__(self, modules: dict, graph_info: dict, step_size=1.):
        '''
        Manually defined topology graph module without rotation.

        Parameters
        ----------
        modules :                           Dictionary containing the framework modules.
        graph_info :                        Dictionary containing topology graph relevant information.
        step_size :                         The step size.

        Returns
        ----------
        None
        '''
        # call the base class init
        super(FourConnectedGraphRotation, self).__init__()
        # normally, the topology graph is not shown in graphicsWindow
        self.visual_output = False
        self.gui_parent = None
        self.layout = None
        self.step_size=step_size
        # store the graph parameters
        self.graph_info = graph_info
        # extract the world module
        self.modules = modules
        # the world module is required here
        world_module = modules['world']

        self.world = modules['world']
        self.offline = world_module.offline
        # get the limits of the given environment
        self.world_limits = world_module.get_limits()
        # retrieve all the wall info from the environment, here self.wall_limits store the min and max XY of each wall,
        # including the surrounding walls and those inside as obstacles; self.perimeter_nodes are a list of points
        # defining the polygon of the surrounding walls. These info is needed to build a basic topology graph of env.
        self.wall_limits, self.perimeter_nodes = world_module.get_wall_graph()
        # constructed for visulization purpose
        self.world_nodes = self.perimeter_nodes[0:-1]
        self.world_edges = []
        for i in range(len(self.world_nodes)):
            self.world_edges.append([i, (i + 1) % len(self.world_nodes)])
        # this is the node corresponding to the robot's actual position
        self.current_node = -1
        self.current_ori = -1
        # this is the node corresponding to the robot's next position
        self.next_node = -1
        # this records the number of times that a same goal has been reached
        self.goal_reached = 0
        # this list of topologyNode[s] stores all nodes of the graph
        self.nodes = []
        # this list of [int,int]-entries defines all edges (unique) which make up the graph's connectivity
        self.edges = []
        # the starting orientation, default not specified
        self.start_ori = graph_info['start_ori']
        self.clique_size = graph_info['clique_size']
        self.num_actions = 6
        # the list for storing the trajectory of the agent where each entry is the agent's pos as [nodeIdx, orientation]
        self.trajectories = []
        # read topology structure from world module
        # here each node is a XY coordinate and each edge is a list containing 2 integers, indicating a connectivity
        # between 2 nodes
        nodes, edges = self.generate_topology_from_worldInfo(step_size)
        # transfer the node points into the self.nodes list
        for i, n in enumerate(nodes):
            # create the corresponding node, where i is the running index of the mesh_points/corresponding nodes
            self.nodes.append(TopologyNode(i, float(n[0]), float(n[1])))
        # fill in the self.edges list from the edges information
        for e in edges:
            self.edges.append([int(e[0]), int(e[1])])
        # define a dedicated 'noneNode' that acts as a placeholder for neighborhood construction
        none_node = TopologyNode(-1, 0.0, 0.0)
        # construct the neighborhoods of each node in the graph
        for edge in self.edges:
            # first edge node
            a = self.nodes[int(edge[0])]
            # second edge node
            b = self.nodes[int(edge[1])]
            # add node a to the neighborhood of node b, we don't do the other way around here, since both edges [a, b] and
            # [b, a] exist in self.edges.
            a.neighbors.append(b)
        # it is possible that a node does not have the maximum possible number of neighbors, to stay consistent in RL, fill up the neighborhood
        # with noneNode[s]:
        for node in self.nodes:
            while len(node.neighbors) < self.clique_size:
                node.neighbors.append(none_node)
        # assign start nodes. there are two options: 1. randomly chose one node out of all nodes as the starting
        # one every trial, or 2. give a specific group of nodes as the starting ones
        if isinstance(self.graph_info['start_nodes'][0], str):
            try:
                self.graph_info['start_nodes'][0] == 'random'
            except ValueError:
                print("Please either enter 'random' or a series of integer numbers as the starting nodes")
            self.nodes[np.random.randint(len(self.nodes))].start_node = True
        else:
            for node_index in self.graph_info['start_nodes']:
                self.nodes[node_index].start_node = True

        # assign goal nodes
        for node_index in self.graph_info['goal_nodes']:
            self.nodes[node_index].goal_node = True
            self.goal_node = node_index
            #print('GOAL',self.goal_node)

        self.goal_ori = self.graph_info['goal_ori']

        # sum up the topology information of the env
        self.top_info = {'nodes': nodes, 'edges': edges, 'walls': self.wall_limits}
###############################################jon
    def update_environment(self, world_module):
        '''
        Updates the environment's topology based on the new world module information.

        Parameters
        ----------
        world_module :                      The updated world module containing new environment details.

        Returns
        ----------
        None
        '''
        # Update the world module and get new limits and walls
        print("Updating environment...")
        self.world = world_module
        self.world_limits = self.world.get_limits()
        self.wall_limits, self.perimeter_nodes = self.world.get_wall_graph()


        # Reconstruct visualization components if needed
        self.world_nodes = self.perimeter_nodes[:-1]
        self.world_edges = [[i, (i + 1) % len(self.world_nodes)] for i in range(len(self.world_nodes))]

        # Regenerate the topology from the updated world information
        nodes, edges = self.generate_topology_from_worldInfo(self.step_size)
        self.nodes = [TopologyNode(i, float(n[0]), float(n[1])) for i, n in enumerate(nodes)]
        self.edges = [[int(e[0]), int(e[1])] for e in edges]

        # Reset neighborhoods
        none_node = TopologyNode(-1, 0.0, 0.0)  # Placeholder for nodes without the maximum neighbors
        for node in self.nodes:
            node.neighbors = []
            for edge in self.edges:
                if edge[0] == node.index:
                    node.neighbors.append(self.nodes[edge[1]])
                if edge[1] == node.index:
                    node.neighbors.append(self.nodes[edge[0]])
            while len(node.neighbors) < self.clique_size:
                node.neighbors.append(none_node)


        # Re-assign start and goal nodes according to the possibly new topology
        self.assign_start_and_goal_nodes()

        # Update visual elements if visualization is enabled
        if self.visual_output:
            self.update_visual_elements()


    def assign_start_and_goal_nodes(self):
        '''
        Reassigns start and goal nodes in the graph based on current graph_info settings.

        Returns
        ----------
        None
        '''
        # Clear existing start and goal settings
        for node in self.nodes:
            node.start_node = False
            node.goal_node = False

        # Reassign start nodes
        if isinstance(self.graph_info['start_nodes'][0], str) and self.graph_info['start_nodes'][0] == 'random':
            self.nodes[np.random.randint(len(self.nodes))].start_node = True
        else:
            for node_index in self.graph_info['start_nodes']:
                self.nodes[node_index].start_node = True

        # Reassign goal nodes
        for node_index in self.graph_info['goal_nodes']:
            self.nodes[node_index].goal_node = True
            self.goal_node = node_index


#######################################################
    def set_visual_debugging(self, visual_output: bool, gui_parent):
        '''
        This function sets visualization flags.

        Parameters
        ----------
        visual_output :                     If true, the topology graph will be visualized.
        gui_parent :                        The main window used during visualization.

        Returns
        ----------
        None
        '''
        self.gui_parent = gui_parent
        self.visual_output = visual_output
        self.init_visual_elements()

    def store_topology(self, path):
        '''
        This funcion stores the topology information to the given path
        '''
        pickle.dump(self.top_info, open(path, 'wb'))


    def sample_state_space(self):
        '''
        This function samples observations at all topology nodes.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        # not currently used
        pass

    def init_visual_elements(self):
        '''
        This function initializes the visual elements.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        # do basic visualization
        # iff visualOutput is set to True!
        if self.visual_output:
            # add the graph plot to the GUI widget
            self.topology_plot_viewbox = self.gui_parent.addPlot(title='Topology graph')
            # set extension of the plot, lock aspect ratio
            self.topology_plot_viewbox.setXRange(self.world_limits[0, 0], self.world_limits[0, 1])
            self.topology_plot_viewbox.setYRange(self.world_limits[1, 0], self.world_limits[1, 1])
            self.topology_plot_viewbox.setAspectLocked(lock=True)
            # overlay the world's perimeter
            self.perimeter_graph = pg.GraphItem()
            self.topology_plot_viewbox.addItem(self.perimeter_graph)
            self.perimeter_graph.setData(pos=np.array(self.world_nodes), adj=np.array(self.world_edges),
                                         brush=(128, 128, 128))
            # overlay the topology graph
            self.topology_graph = pg.GraphItem()
            self.topology_plot_viewbox.addItem(self.topology_graph)
            # set up a brushes array for visualization of the nodes
            # normal nodes are grey
            symbol_brushes = [pg.mkBrush(color=(128, 128, 128))] * len(self.nodes)
            # set colors of normal and goal nodes
            for node in self.nodes:
                # start nodes are green
                if node.start_node:
                    symbol_brushes[node.index] = pg.mkBrush(color=(0, 255, 0))
                # goal node is red
                if node.goal_node:
                    symbol_brushes[node.index] = pg.mkBrush(color=(255, 0, 0))
            # construct appropriate arrays from the self.nodes and the self.edges information
            temp_nodes, temp_edges = [], []
            for node in self.nodes:
                temp_nodes.append([node.x, node.y])
            for edge in self.edges:
                temp_edges.append([edge[0], edge[1]])
            self.topology_graph.setData(pos=np.array(temp_nodes), adj=np.array(temp_edges), symbolBrush=symbol_brushes)
            # eventually, overlay robot marker
            self.pos_marker = CogArrow(angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 0, 0))
            self.topology_plot_viewbox.addItem(self.pos_marker)
            # initial position to center, this has to be worked over later!
            self.pos_marker.set_data(0.0, 0.0, 90.0)

    def update_robot_pose(self, pose: np.ndarray):
        '''
        This function updates the visual depiction of the agent(robot).

        Parameters
        ----------
        pose :                              The agent's pose.

        Returns
        ----------
        None
        '''
        if self.visual_output:
            self.pos_marker.set_data(pose[0], pose[1], np.rad2deg(np.arctan2(pose[3], pose[2])))

    def reset_start_nodes(self, start_nodes, ori):
        '''
        This function resets the starting nodes and orientations

        Parameters
        ----------
        start_nodes :                       The starting nodes.
        ori :                               The starting orientation.

        Returns
        ----------
        None
        '''
        self.start_ori = None
        for node in self.nodes:
            node.start_node = False
        for node_index in start_nodes:
            self.nodes[node_index].start_node = True
        self.start_ori = ori


    def reset_goal_nodes(self, goal_nodes, ori):
        '''
        This function resets the goal nodes and orientations

        Parameters
        ----------
        start_nodes :                       The starting nodes.
        ori :                               The starting orientation.

        Returns
        ----------
        None
        '''
        for node in self.nodes:
            node.goal_node = False
        for node_index in goal_nodes:
            self.nodes[node_index].goal_node = True
            self.goal_node = node_index

        self.goal_ori = ori

    def reset_goal_nodes_Unity_ONLINE(self, goal_nodes, ori):
        '''
        This function resets the goal nodes and orientations in UNITY online env

        Parameters
        ----------
        start_nodes :                       The starting nodes.
        ori :                               The starting orientation.

        Returns
        ----------
        None
        '''

        if goal_nodes==-1:
            self.world.move_object_only(40, 40)
        else:
            if isinstance(goal_nodes, np.ndarray):
                goal_nodes = goal_nodes.tolist()  # Convertimos el array en una lista
            elif not isinstance(goal_nodes, list):
                goal_nodes = [goal_nodes]  # Si es un número individual, lo ponemos dentro de una lista
            # This function gets the coordinates from the node_id
            def node_to_coordinates(node_id, num_nodes, world_limits):
                # Asumiendo una distribución uniforme en una cuadrícula cuadrada
                grid_size = int(num_nodes ** 0.5)  # Calcula el tamaño de la cuadrícula (5x5 para 25 nodos)
                # Calcular la posición del nodo en la cuadrícula
                row = node_id // grid_size
                col = node_id % grid_size

                # Extraer los límites para x y z del nuevo formato de world_limits
                x_min, x_max = world_limits[0]
                z_min, z_max = world_limits[1]

                # Convertir a coordenadas del mundo real, ajustando para ocupar el nuevo espacio de límites
                z = z_min + (z_max - z_min) / (grid_size - 1) * col
                x = x_min + (x_max - x_min) / (grid_size - 1) * row

                return x, z

            world_limits = self.world_limits + [[1.,-1,],[1.,-1.]]
            #print(goal_nodes)
            coordinates = node_to_coordinates(goal_nodes[0], len(self.nodes), world_limits)
            self.world.move_object_only(coordinates[0], coordinates[1])

    def get_goal(self):
        return [self.goal_node, self.goal_ori]

    def get_node_number(self) -> int:
        '''
        This function returns the number of nodes of the topology graph.

        Parameters
        ----------
        start_nodes :                       The agent's pose.
        ori :                               The agent's pose.

        Returns
        ----------
        None
        '''
        return len(self.nodes)

    def generate_topology_from_worldInfo(self, step_size=1.0) -> (np.ndarray, np.ndarray):
        '''
        This function generates a topolgy graph from the world information.

        Parameters
        ----------
        step_size :                         The minimal distance between nodes.

        Returns
        ----------
        nodes :                             The nodes of the generated topology graph.
        edges :                             The edges of the generated topology graph.
        '''
        # for each wall limit, create the correponding four edges, since each wall is a square on 2D plane
        wall_edges = []
        for item in self.wall_limits:
            # edge: (minx, minz), (minx, maxz)
            wall_edges.append([[item[0], item[1]], [item[0], item[3]]])
            # edge: (minx, maxz), (maxx, maxz)
            wall_edges.append([[item[0], item[3]], [item[2], item[3]]])
            # edge: (maxx, maxz), (maxx, minz)
            wall_edges.append([[item[2], item[3]], [item[2], item[1]]])
            # edge: (maxx, minz), (minx, minz)
            wall_edges.append([[item[2], item[1]], [item[0], item[1]]])
        wall_edges = np.asarray(wall_edges)

        # generate 2d grid based on a square defined by the world nodes (minx, miny) and (maxx, maxy)
        minx = ceil(min(self.world_nodes[:, 0]))
        miny = ceil(min(self.world_nodes[:, 1]))
        maxx = floor(max(self.world_nodes[:, 0]))
        maxy = floor(max(self.world_nodes[:, 1]))
        x = np.arange(minx, maxx + step_size, step_size)
        y = np.arange(miny, maxy + step_size, step_size)
        xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')

        # store each nodes which is not too close to the wall
        nodes, edges, nodes_index = [], [], []
        index = 0
        # from min x to max x; from min y to max y
        for xi in xv[0, :]:
            for yi in yv[:, 0]:
                distances = lineseg_dists(np.array([xi, yi]), wall_edges[:, 0], wall_edges[:, 1])
                # if there is a node too close to any of the wall, its index is None
                distances = distances < (step_size / 5)
                if distances.nonzero()[0].size > 0:
                    nodes_index.append(None)
                else:
                    # if a node is outside the perimeter polygon, its index is None
                    if not point_in_polygon(self.perimeter_nodes, np.array([xi, yi])):
                        nodes_index.append(None)
                    else:
                        nodes_index.append(index)
                        nodes.append(np.array([xi, yi]))
                        index += 1
        # Construct the 4-coneected edges
        num_xgrid = len(xv[0, :])
        num_ygrid = len(yv[:, 0])
        # the first node is at the left bottom corner, then fill the space in y direction
        for i in range(len(nodes_index)):
            row = int(i % num_ygrid)
            col = int(i / num_ygrid)
            if nodes_index[i] is None:
                continue
            # add the left connectivity
            if col - 1 < 0:
                pass
            else:
                if nodes_index[(col - 1) * num_ygrid + row] is not None:
                    edge_line = [nodes[nodes_index[i]], nodes[nodes_index[(col - 1) * num_ygrid + row]]]
                    if not if_intersect_multi(edge_line, wall_edges):
                        edges.append([nodes_index[i], nodes_index[(col - 1) * num_ygrid + row]])
            # add the right connectivity
            if col + 1 > num_xgrid - 1:
                pass
            else:
                if nodes_index[(col + 1) * num_ygrid + row] is not None:
                    edge_line = [nodes[nodes_index[i]], nodes[nodes_index[(col + 1) * num_ygrid + row]]]
                    if not if_intersect_multi(edge_line, wall_edges):
                        edges.append([nodes_index[i], nodes_index[(col + 1) * num_ygrid + row]])
            # add the down connectivity
            if row - 1 < 0:
                pass
            else:
                if nodes_index[col * num_ygrid + row - 1] is not None:
                    edge_line = [nodes[nodes_index[i]], nodes[nodes_index[col * num_ygrid + row - 1]]]
                    if not if_intersect_multi(edge_line, wall_edges):
                        edges.append([nodes_index[i], nodes_index[col * num_ygrid + row - 1]])
            # add the up connectivity
            if row + 1 > num_ygrid - 1:
                pass
            else:
                if nodes_index[col * num_ygrid + row + 1] is not None:
                    edge_line = [nodes[nodes_index[i]], nodes[nodes_index[col * num_ygrid + row + 1]]]
                    if not if_intersect_multi(edge_line, wall_edges):
                        edges.append([nodes_index[i], nodes_index[col * num_ygrid + row + 1]])

        return np.asarray(nodes), np.asarray(edges)

    def normalize_angle(self, angle: float) -> float:
        '''
        In this topology, there are only 3 options for the absolute value of the agent's angle
        [0, 90, 180]. Here we force the real angle of the robot to be one of them because there was
        always a numerical error caused by computation inaccuracy.

        Parameters
        ----------
        angle :                             The angle to be normalized.

        Returns
        ----------
        angle :                             The normalized angle.
        '''
        possible_angles = np.array([0, 90, 180])
        angle = np.rad2deg(np.arctan2(np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))))
        deviations = np.abs(possible_angles - np.abs(angle))
        ideal_angle = possible_angles[np.argmin(deviations)]

        if ideal_angle == 90:
            if angle < 0:
                angle = -ideal_angle
            else:
                angle = ideal_angle
        elif ideal_angle == 0:
            angle = ideal_angle
        else:
            angle = -ideal_angle  # 180 and -180 are the same, here we always take -180

        return angle




    def generate_behavior_from_action(self, action: int) -> dict:
        '''
        This function executes the action selected by the agent.

        Parameters
        ----------
        action :                            The action to be executed.

        Returns
        ----------
        callback_value :                    Dictionary containing information about the new environmental state.
        '''
        # the world module is required here
        world_module = self.modules['world']
        next_node_pos = np.array([0.0, 0.0])
        callback_value = dict()
        callback_value['hit_wall'] = False
        # if a standard action is performed, NOT a reset
        if action != 'reset':
            # get current heading
            heading_x = np.cos((world_module.env_data['pose'][2]) / 180 * np.pi)
            heading_y = np.sin((world_module.env_data['pose'][2]) / 180 * np.pi)
            heading = np.array([heading_x, heading_y])
            heading = heading / np.linalg.norm(heading)
            # get directions of all edges
            # here, self is this class itself
            actual_node = self.nodes[self.current_node]
            neighbors = self.nodes[self.current_node].neighbors
            # lists for forward edges
            forward_edge, right_edge, left_edge, backward_edge = [], [], [], []
            # find possible movement directions. Note: when a left edge is found, it is simultaneously stored as a right edge with huge turning angle, and vice versa. That way,
            # the agent does not get stuck in situations where there is only a forward edge, and say, a left edge, and the action is 'right'. In such a situation, the agent will just turn
            # right using the huge 'right' turning angle.
            for n in neighbors:
                if n.index != -1:
                    actual_node_position = np.array([actual_node.x, actual_node.y])
                    neighbor_position = np.array([n.x, n.y])
                    vec_edge = neighbor_position - actual_node_position
                    vec_edge = vec_edge / np.linalg.norm(vec_edge)
                    angle = np.arctan2(heading[0] * vec_edge[1] - heading[1] * vec_edge[0],
                                       heading[0] * vec_edge[0] + heading[1] * vec_edge[1])
                    angle = angle / np.pi * 180.0
                    angle = self.normalize_angle(angle)
                    if angle == 0:
                        forward_edge = [n.index, vec_edge, angle]
                    elif angle == 90:
                        left_edge = [n.index, vec_edge, angle]
                    elif angle == -90:
                        right_edge = [n.index, vec_edge, angle]
                    elif angle == 180 or angle == -180:
                        backward_edge = [n.index, vec_edge, angle]
            # store the current node as previous node
            previous_node = self.current_node
            angle = world_module.env_data['pose'][2]

            if action == 0:
                # this is a forward movement
                if len(forward_edge) != 0:
                    # there is a forward edge that the agent can use
                    self.next_node = forward_edge[0]
                    next_node_pos = np.array([self.nodes[
                                                  self.next_node].x,
                                              self.nodes[
                                                  self.next_node].y])
                else:
                    # no forward edge found, the agent remains still
                    self.next_node = self.modules[
                        'spatial_representation'].current_node
                    next_node_pos = np.array([self.nodes[
                                                  self.next_node].x,
                                              self.nodes[
                                                  self.next_node].y])
                self.update_robot_pose(
                    [next_node_pos[0], next_node_pos[1], heading[0], heading[1]])
                callback_value['hit_wall'] = True
            elif action == 1:
                # this is a left turn movement
                self.next_node = self.current_node
                next_node_pos = np.array(
                    [self.nodes[self.next_node].x,
                     self.nodes[self.next_node].y])
                angle += 90  # turn 90 degrees to the left
                self.update_robot_pose(
                    [next_node_pos[0], next_node_pos[1], np.cos(angle / 180 * np.pi), np.sin(angle / 180 * np.pi)])
            elif action == 2:
                # this is a right turn movement
                self.next_node = self.current_node
                next_node_pos = np.array(
                    [self.nodes[self.next_node].x,
                     self.nodes[self.next_node].y])
                angle -= 90
                self.update_robot_pose(
                    [next_node_pos[0], next_node_pos[1], np.cos(angle / 180 * np.pi), np.sin(angle / 180 * np.pi)])
            elif action == 3:
                # do nothing
                self.next_node = self.modules[
                    'spatial_representation'].current_node
                next_node_pos = np.array([self.nodes[
                                              self.next_node].x,
                                          self.nodes[
                                              self.next_node].y])
                self.update_robot_pose(
                    [next_node_pos[0], next_node_pos[1], heading[0], heading[1]])
                # no left edge found, the agent remains still
                # this is a left movement
                #if len(left_edge) != 0:
                #    # there is a left edge that the agent can use
                #    self.next_node = left_edge[0]
                #    next_node_pos = np.array([self.nodes[
                #                                  self.next_node].x,
                #                              self.nodes[
                #                                  self.next_node].y])
                #else:
                #    # no left edge found, the agent remains still
                #    self.next_node = self.modules[
                #        'spatial_representation'].current_node
                #    next_node_pos = np.array([self.nodes[
                #                                  self.next_node].x,
                #                              self.nodes[
                #                                  self.next_node].y])
                #self.update_robot_pose(
                #    [next_node_pos[0], next_node_pos[1], heading[0], heading[1]])
                #callback_value['hit_wall'] = True
            elif action == 4:
                # this is a right movement
                if len(right_edge) != 0:
                    # there is a right edge that the agent can use
                    self.next_node = right_edge[0]
                    next_node_pos = np.array([self.nodes[
                                                  self.next_node].x,
                                              self.nodes[
                                                  self.next_node].y])
                else:
                    # no right edge found, the agent remains still
                    self.next_node = self.modules[
                        'spatial_representation'].current_node
                    next_node_pos = np.array([self.nodes[
                                                  self.next_node].x,
                                              self.nodes[
                                                  self.next_node].y])
                self.update_robot_pose(
                    [next_node_pos[0], next_node_pos[1], heading[0], heading[1]])
                callback_value['hit_wall'] = True
            elif action == 5:
                # this is a backward movement
                if len(backward_edge) != 0:
                    # there is a backward edge that the agent can use
                    self.next_node = backward_edge[0]
                    next_node_pos = np.array([self.nodes[
                                                  self.next_node].x,
                                              self.nodes[
                                                  self.next_node].y])
                else:
                    # no backward edge found, the agent remains still
                    self.next_node = self.modules[
                        'spatial_representation'].current_node
                    next_node_pos = np.array([self.nodes[
                                                  self.next_node].x,
                                              self.nodes[
                                                  self.next_node].y])
                self.update_robot_pose(
                    [next_node_pos[0], next_node_pos[1], heading[0], heading[1]])
                callback_value['hit_wall'] = True

            # CAMBIAR
            ###########UPDATE UNITY REWARD TO MATCH WITH THE LOCATION WHEN IS REACHED
            if (self.next_node ==self.goal_node) and (action==3):  # and self.discarted==False:
                self.reset_goal_nodes_Unity_ONLINE([self.next_node], 0)
            ###########UPDATE UNITY REWARD TO MATCH WITH THE LOCATION
            angle = self.normalize_angle(angle)
            # For online simulation, the control commands are next node's XY postion and angle
            # For offline simulation, the control commands are next node's index and angle
            if not self.offline:
                self.modules['world'].actuate_robot(np.array([next_node_pos[0], next_node_pos[1], angle]))
            else:
                self.modules['world'].actuate_robot(np.array([self.next_node, angle]))
            # update the observation
            self.modules['observation'].update()
            # make the current node the one the agent travelled to
            self.current_node = self.next_node
            # here, next node is already set and the current node is set to this next node.
            callback_value['current_node'] = self.nodes[self.next_node]

            # record the current node idx and angle into the last trajectory history
            self.trajectories[-1].append([self.current_node, angle])
            # put the trajectories up till now of this episode in the call back value
            callback_value['episode_traj'] = self.trajectories[-1]

        # if a reset is performed
        else:
            # a random node is chosen to place the agent at (this node MUST NOT be the global goal node!)
            nodes = self.nodes
            nodes_selection = [n for n in nodes if n.start_node == True]
            # store the current node as previous node
            #previous_node = self.current_node
            self.next_node = np.random.choice(nodes_selection)
            # NOT RESET IN THE SAME NODE as GOAL
            while  self.get_goal()[0]==self.next_node.index:
                self.next_node = np.random.choice(nodes_selection)


            next_node_pos = np.array(
                [self.nodes[self.next_node.index].x,
                 self.nodes[
                     self.next_node.index].y])
            self.current_node = self.next_node.index
            neighbors = self.next_node.neighbors
            # list for available neighbor directions
            directions = []
            for n in neighbors:
                if n.index != -1:
                    # only parse valid neighbors
                    next_node_position = np.array([self.next_node.x,
                                                   self.next_node.y])
                    neighbor_position = np.array([n.x, n.y])
                    vec_edge = neighbor_position - next_node_position
                    vec_edge = vec_edge / np.linalg.norm(vec_edge)
                    world_angle = np.arctan2(vec_edge[1], vec_edge[0])
                    directions.append([n.index, vec_edge, world_angle])
            # select new heading randomly
            new_heading_selection = directions[np.random.randint(len(directions))]
            new_heading_angle = new_heading_selection[2]
            new_heading_angle = np.rad2deg(new_heading_angle)
            new_heading_vector = new_heading_selection[1]
            new_heading_angle = self.normalize_angle(new_heading_angle)
            if self.start_ori is not None:
                new_heading_angle = np.random.choice(self.start_ori, 1)[0]
                new_heading_vector[0] = np.cos(np.deg2rad(new_heading_angle))
                new_heading_vector[1] = np.sin(np.deg2rad(new_heading_angle))
            # update the agents position and orientation (heading)
            if not self.offline:
                self.modules['world'].actuate_robot(np.array([next_node_pos[0], next_node_pos[1], new_heading_angle]))
            else:
                self.modules['world'].actuate_robot(
                    np.array([self.next_node.index, new_heading_angle]))
            self.update_robot_pose(
                [next_node_pos[0], next_node_pos[1], new_heading_vector[0], new_heading_vector[1]])
            # update the observation
            self.modules['observation'].update()
            # whenever there is a reset, start recording a new trajectory
            self.trajectories.append([[self.current_node, new_heading_angle]])

        # we also update the current orientation
        self.current_ori = int(self.modules['world'].env_data['pose'][2])

        callback_value['current_node'] = self.nodes[self.current_node]

        # if possible try to update the visual debugging display
        if hasattr(qt.QtGui, 'QApplication'):
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
        else:
            if qt.QtWidgets.QApplication.instance() is not None:
                qt.QtWidgets.QApplication.instance().processEvents()

        return callback_value

    def update_visual_elements(self):
        '''
        This function updates the visual elements.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        # set up a brushes array for visualization of the nodes
        # normal nodes are grey
        symbol_brushes = [pg.mkBrush(color=(128, 128, 128))] * len(self.nodes)
        # set colors of normal and goal nodes
        for node in self.nodes:
            # start nodes are green
            if node.start_node:
                symbol_brushes[node.index] = pg.mkBrush(color=(0, 255, 0))
            # goal node is red
            if node.goal_node:
                symbol_brushes[node.index] = pg.mkBrush(color=(255, 0, 0))
        # construct appropriate arrays from the self.nodes and the self.edges information
        temp_nodes, temp_edges = [], []
        for node in self.nodes:
            temp_nodes.append([node.x, node.y])
        for edge in self.edges:
            temp_edges.append([edge[0], edge[1]])
        self.topology_graph.setData(pos=np.array(temp_nodes), adj=np.array(temp_edges), symbolBrush=symbol_brushes)

    def get_action_space(self) -> gym.spaces.Discrete:
        '''
        This function returns the action space.

        Parameters
        ----------
        None

        Returns
        ----------
        action_space :                      The (gym) action space.
        '''
        return gym.spaces.Discrete(self.num_actions)

    def clear_trajectories(self):
        '''
        This function clears the trajectory history if needed.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        # clear the trajectory history if needed
        self.trajectories = []
