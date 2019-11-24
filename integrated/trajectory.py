import numpy as np


def vel_from_world_to_object(object_state_current, object_state_before):
    delta_position_x, delta_position_y, delta_theta = (i for i in (object_state_current - object_state_before))
    theta = object_state_current[2]
    vel_relative_to_object_x = delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    vel_relative_to_object_y = delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)

    return np.array([vel_relative_to_object_x, vel_relative_to_object_y, delta_theta])

def robot_pos_from_world_to_relative(object_pos, robot_pos):

    object_x, object_y, theta = [object_pos[i] for i in range(3)]
    robot_x, robot_y = [robot_pos[i] for i in range(2)]

    delta_position_x, delta_position_y = robot_x - object_x, robot_y - object_y
    robot_position_relative_to_object_x = delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    robot_position_relative_to_object_y = delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)

    return np.array([robot_position_relative_to_object_x, robot_position_relative_to_object_y])

class Trajectory():
    """collect trajectory history and preprocess the data making it more suitable for the input of predictor"""
    def __init__(self, env):
        self.env = env
        self.goal = np.zeros([7])
        self.obstacle_pos = np.zeros([3])
        self.relative_state = [] # list of numpy.array(delta_x,delta_y,delta_theta,x_robot, y_robot, action_x, action_y)
        self.absolute_state = [] # list of numpy.array(x_object,y_object,theta_object,x_robot,y_robot)
        self.object_absolute_state = [] # list of numpy.array(pos_x, pos_y, theta)
        self.robot_absolute_state = [] # list of numpy.array(pos_x, pos_y)
        self.count = 0


    def get_relative_state(self):
        return np.asarray(self.relative_state)

    def get_absolute_state(self):
        return np.asarray(self.absolute_state)

    def get_goal_position(self):
        goal = self.goal
        orientation = 2*np.arccos(goal[3])
        return np.array([goal[0], goal[1], orientation])

    def get_obstacle_position(self):
        return np.array([self.obstacle_pos[0], self.obstacle_pos[1]])

    def reset(self):
        # reset the count num
        self.count = 0
        self.relative_state.clear()
        self.absolute_state.clear()
        self.object_absolute_state.clear()
        self.robot_absolute_state.clear()

        self.goal = np.zeros([7])
        self.obstacle_pos = np.zeros([3])

    def update(self, obs):
        self.goal = obs['desired_goal']
        self.obstacle_pos = self.env.unwrapped.sim.data.get_site_xpos('obstacle')
        observation = obs['observation']
        robot_pos = observation[:2] # absolute
        object_pos = observation[6:13] # absolute
        self.object_absolute_state.append(np.array([object_pos[0], object_pos[1], 2*np.arccos(object_pos[3])]))
        self.robot_absolute_state.append(robot_pos)

        # updata relative_state
        # object_vel = np.zeros([3])
        if self.count == 0:
            object_vel = np.zeros([3])
        else:
            object_vel = vel_from_world_to_object(self.object_absolute_state[self.count], self.object_absolute_state[self.count - 1])

        robot_pos_relative = robot_pos_from_world_to_relative(object_pos, robot_pos)
        robot_action_relative = np.zeros([2])
        self.relative_state.append(np.concatenate([object_vel, robot_pos_relative, robot_action_relative]))

        # update absolute state
        self.absolute_state.append(np.concatenate([self.object_absolute_state[self.count], self.robot_absolute_state[self.count]]))

        # update count
        self.count += 1
