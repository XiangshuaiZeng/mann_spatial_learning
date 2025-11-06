import numpy as np
import gym
from gym import spaces
import PyQt5 as qt
import pyqtgraph as pg
import pyqtgraph.functions
from math import ceil, floor
import pickle

# Import functions and classes from your CoBeL framework
from cobel.spatial_representations.topology_graphs.misc.topology_node import TopologyNode
from cobel.spatial_representations.topology_graphs.misc.cog_arrow import CogArrow
from cobel.spatial_representations.spatial_representation import SpatialRepresentation
from cobel.spatial_representations.topology_graphs.misc.utils import (
    lineseg_dists,
    point_in_polygon,
    doIntersect,
    Point
)

def if_intersect_multi(single_line, line_group):
    """
    Checks if `single_line` (a segment) intersects with any line inside `line_group`.
    """
    p1 = Point(single_line[0][0], single_line[0][1])
    q1 = Point(single_line[1][0], single_line[1][1])
    for line in line_group:
        p2 = Point(line[0][0], line[0][1])
        q2 = Point(line[1][0], line[1][1])
        if doIntersect(p1, q1, p2, q2):
            return True
    return False


class Discrete2DEnv(SpatialRepresentation):
    """
    Discrete 2D navigation environment with two wheels (tank type).
    At each step, the action is an integer that indicates which wheels are activated:
        0 -> no wheel
        1 -> left wheel only
        2 -> right wheel only
        3 -> both wheels

    Each activated wheel moves at a fixed linear speed (wheel_speed), and the distance
    between the wheels (track_width) determines the rotation rate.
    """

    def __init__(
        self,
        modules: dict,
        graph_info: dict,
        wheel_speed=1.0,
        track_width=0.5,
        dt=1.0
    ):
        """
        Parameters
        ----------
        modules : dict
            Framework modules (e.g., 'world', 'observation', etc.).
        graph_info : dict
            Dictionary containing info about start_nodes, start_ori, goal_nodes,
            goal_ori, clique_size, etc.
        wheel_speed : float
            Linear speed of each wheel when active.
        track_width : float
            Distance between the robot's two wheels.
        dt : float
            Fictive time step for each action.
        """
        super().__init__()  # Call SpatialRepresentation constructor

        # 1) Store references and configuration
        self.modules = modules
        self.graph_info = graph_info
        self.wheel_speed = wheel_speed
        self.track_width = track_width
        self.dt = dt

        # 2) World data
        self.world = self.modules['world']
        self.offline = self.world.offline
        self.world_limits = self.world.get_limits()  # -> [[xMin,xMax],[yMin,yMax]]
        self.wall_limits, self.perimeter_nodes = self.world.get_wall_graph()

        # 3) Visualization
        self.visual_output = False
        self.gui_parent = None
        self.pos_marker = CogArrow(angle=0.0, headLen=20.0, tipAngle=25.0, tailLen=0.0, brush=(255, 0, 0))

        # 4) Agent pose [x, y, theta_in_degrees]
        self.agent_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_pose = -1
        self.current_ori = -1

        # 5) Goal and orientation info
        self.start_ori = graph_info.get('start_ori', 0.0)
        self.start_nodes = graph_info.get('start_nodes', ['random'])

        self.goal_reached = 0
        self.goal_nodes = graph_info.get('goal_nodes', [])
        self.goal_node = self.goal_nodes[0] if len(self.goal_nodes) > 0 else None
        self.goal_marker = None

        # 6) Trajectories
        self.trajectories = []

    # =========================================================================
    # ====================    Visual / Debugging Methods    ====================
    # =========================================================================

    def set_visual_debugging(self, visual_output: bool, gui_parent):
        """
        Enables/disables visualization. Creates the window if necessary.
        """
        self.visual_output = visual_output
        self.gui_parent = gui_parent
        self.init_visual_elements()

    def init_visual_elements(self):
        """
        Initializes graphic elements (pyqtgraph) if visualization is required.
        """
        if self.visual_output and self.gui_parent is not None:
            # Create a plot to display the environment
            self.topology_plot_viewbox = self.gui_parent.addPlot(title='Discrete 2D Env (Tank)')
            # Adjust axes to match world limits
            x_min, x_max = self.world_limits[0]
            y_min, y_max = self.world_limits[1]

            self.topology_plot_viewbox.setXRange(x_min, x_max)
            self.topology_plot_viewbox.setYRange(y_min, y_max)
            self.topology_plot_viewbox.setAspectLocked(lock=True)

            # Draw perimeter
            self.perimeter_graph = pg.GraphItem()
            self.topology_plot_viewbox.addItem(self.perimeter_graph)

            # Perimeter nodes: if the last closes the polygon, draw it
            self.world_nodes = self.perimeter_nodes[:-1]
            self.world_edges = []
            for i in range(len(self.world_nodes)):
                self.world_edges.append([i, (i + 1) % len(self.world_nodes)])

            self.perimeter_graph.setData(
                pos=np.array(self.world_nodes),
                adj=np.array(self.world_edges),
                brush=(128, 128, 128)
            )

            # Robot marker
            self.topology_plot_viewbox.addItem(self.pos_marker)
            # Initialize at (0,0) with angle 0
            self.pos_marker.set_data(0.0, 0.0, 0.0)

            # GOAL marker
            if self.goal_node is not None:
                self.goal_marker = pg.ScatterPlotItem(
                    size=10,
                    brush=(0, 255, 0),  # GREEN color for the goal
                    pen=None
                )
                coordinates = self.node_to_coordinates(self.goal_node)
                self.goal_marker.setData(pos=np.array([[coordinates[0], coordinates[1]]]))

                self.topology_plot_viewbox.addItem(self.goal_marker)

    def update_robot_pose(self, pose: np.ndarray):
        """
        Updates position/orientation in the visualization.
        pose: [x, y, theta_degs]
        """
        if self.visual_output:
            x, y, theta_deg = pose
            self.pos_marker.set_data(x, y, theta_deg)

    def update_goal_pose(self, goal_node):
        """
        Updates goal position in the visualization.
        """
        if self.visual_output and self.goal_marker is not None:
            coordinates = self.node_to_coordinates(goal_node)
            self.goal_marker.setData(pos=np.array([[coordinates[0], coordinates[1]]]))

    # =========================================================================
    # ====================    Action / Movement Methods    =====================
    # =========================================================================

    def get_action_space(self) -> spaces.Discrete:
        """
        Returns a discrete action space with 4 actions:
          0 -> no wheel
          1 -> left wheel only
          2 -> right wheel only
          3 -> both wheels
        """
        return spaces.Discrete(3)

    def generate_behavior_from_action(self, action) -> dict:
        """
        Executes the discrete action and updates the robot's pose using two-wheel
        differential (tank-like) kinematics.

        Actions:
            0 -> v_l=0,   v_r=0
            1 -> v_l=+1,  v_r=0
            2 -> v_l=0,   v_r=+1
            3 -> v_l=+1,  v_r=+1

        * wheel_speed scales velocity (1 -> wheel_speed).
        * track_width determines rotational velocity.
        """
        callback_value = dict()
        callback_value['hit_wall'] = False

        # -- Check if "reset" was passed instead of a discrete action
        if isinstance(action, str) and action == 'reset':
            return self._reset_agent_pose()

        # -- Assign wheel speeds based on action
        if action == 0:
            v_left, v_right = self.wheel_speed, 0.0
        elif action == 1:
            v_left, v_right = 0.0, self.wheel_speed
        elif action == 2:
            v_left, v_right = self.wheel_speed, self.wheel_speed
        else:
            # In case of unexpected value
            v_left, v_right = 0.0, 0.0

        x, y, theta_deg = self.agent_pose
        theta_rad = np.deg2rad(theta_deg)

        # Differential drive kinematics
        v_lin = (v_left + v_right) / 2.0  # Linear velocity of the center
        omega = (v_right - v_left) / self.track_width  # Angular velocity (rad/step)

        # Integrate pose
        new_x = x + v_lin * self.dt * np.cos(theta_rad)
        new_y = y + v_lin * self.dt * np.sin(theta_rad)

        new_theta_rad = theta_rad + omega * self.dt
        new_theta_deg = np.rad2deg(new_theta_rad) % 360

        # -- Collision check
        if self._check_collision(x, y, new_x, new_y):
            new_x = x
            new_y = y
            new_theta_deg = new_theta_deg
            callback_value['hit_wall'] = True

        # -- Update internal pose
        self.agent_pose = np.array([new_x, new_y, new_theta_deg], dtype=np.float32)

        # -- Send pose to Unity (or world sim)
        if not self.offline:
            self.modules['world'].actuate_robot(self.agent_pose)
        else:
            self.modules['world'].actuate_robot(self.agent_pose)

        # -- Update observation
        self.modules['observation'].update()

        # -- Store in trajectory
        if len(self.trajectories) == 0:
            self.trajectories.append([])
        self.trajectories[-1].append([new_x, new_y, new_theta_deg])

        # -- Return callback info
        callback_value['episode_traj'] = self.trajectories[-1]
        callback_value['current_pose'] = self.agent_pose.copy()

        # -- Visualization
        self.update_robot_pose(self.agent_pose)
        if hasattr(qt.QtGui, 'QApplication'):
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
        else:
            if qt.QtWidgets.QApplication.instance() is not None:
                qt.QtWidgets.QApplication.instance().processEvents()

        return callback_value

    def _reset_agent_pose(self):
        """
        Resets the agent's pose to a random (x, y) position and random orientation (0-360).
        """
        callback_value = dict()
        callback_value['hit_wall'] = False

        # Choose initial angle
        theta0 = np.random.uniform(0, 360)

        # Choose random position within safe bounds
        x_min, x_max = self.world_limits[0]
        y_min, y_max = self.world_limits[1]
        x_min += 0.2
        x_max -= 0.2
        y_min += 0.2
        y_max -= 0.2

        x0 = np.random.uniform(x_min, x_max)
        y0 = np.random.uniform(y_min, y_max)

        self.agent_pose = np.array([x0, y0, theta0], dtype=np.float32)

        # Send pose to Unity (or offline world)
        if not self.offline:
            self.modules['world'].actuate_robot(self.agent_pose)
        else:
            self.modules['world'].actuate_robot(self.agent_pose)

        self.modules['observation'].update()

        # New trajectory
        self.trajectories.append([[x0, y0, theta0]])

        # Visualization
        self.update_robot_pose(self.agent_pose)

        callback_value['episode_traj'] = self.trajectories[-1]
        callback_value['current_pose'] = self.agent_pose.copy()
        return callback_value

    # =========================================================================
    # ====================    Collision / Reset / Others    ===================
    # =========================================================================

    def _check_collision(self, x, y, new_x, new_y):
        """
        Checks for collisions with walls or if agent leaves world limits.
        """
        # -- 1) Global boundaries
        x_min, x_max = self.world_limits[0]
        y_min, y_max = self.world_limits[1]
        if not (x_min + 0.2 <= new_x <= x_max - 0.2 and y_min + 0.2 <= new_y <= y_max - 0.2):
            return True

        # -- 2) Internal walls
        segment = [[x, y], [new_x, new_y]]
        for wall in self.wall_limits:
            # wall = [xmin, zmin, xmax, zmax]
            corners = [
                [[wall[0], wall[1]], [wall[0], wall[3]]],
                [[wall[0], wall[3]], [wall[2], wall[3]]],
                [[wall[2], wall[3]], [wall[2], wall[1]]],
                [[wall[2], wall[1]], [wall[0], wall[1]]]
            ]
            if if_intersect_multi(segment, corners):
                return True
        return False

    def clear_trajectories(self):
        """
        Clears trajectory history.
        """
        self.trajectories = []

    def sample_state_space(self):
        """
        Abstract method not used in this environment.
        """
        pass

    # =========================================================================
    # ====================    Goal & Various Utilities    =====================
    # =========================================================================

    def reset_goal_nodes(self, goal_nodes, ori):
        """
        Resets goal nodes in visualization and internal logic.
        """
        for node_index in goal_nodes:
            self.goal_node = node_index
            # Update visualization
            self.update_goal_pose(node_index)

    def reset_goal_nodes_Unity_ONLINE(self, goal_nodes, ori):
        """
        Resets goal nodes for Unity (online environment).
        """
        if goal_nodes == -1:
            self.world.move_object_only(40, 40)
        else:
            coordinates = self.node_to_coordinates(goal_nodes[0])
            self.world.move_object_only(coordinates[0], coordinates[1])

    def is_agent_in_goal(self):
        """
        Checks if the agent's position is within a certain radius of the goal node.
        """
        if self.goal_node is None:
            return False

        goal_x, goal_z = self.node_to_coordinates(self.goal_node)
        agent_x, agent_z = self.agent_pose[:2]
        distance = np.sqrt((goal_x - agent_x) ** 2 + (goal_z - agent_z) ** 2)
        tolerance = 0.8
        return distance <= tolerance

    def get_goal(self):
        """
        Returns the current goal node.
        """
        return [self.goal_node] if self.goal_node is not None else []

    def node_to_coordinates(self, node_id):
        """
        Simple example mapping a node ID to (x, z) coordinates in a 5x5 grid.
        Adjust according to your specific node and world_limit definitions.
        """
        num_nodes = 25
        world_limits = self.world_limits + [[1., -1.], [1., -1.]]
        grid_size = int(num_nodes ** 0.5)  # 5 if num_nodes=25

        row = node_id // grid_size
        col = node_id % grid_size

        x_min, x_max = world_limits[0]
        z_min, z_max = world_limits[1]

        z = z_min + (z_max - z_min) / (grid_size - 1) * col
        x = x_min + (x_max - x_min) / (grid_size - 1) * row

        return x, z


