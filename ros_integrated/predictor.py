from __future__ import print_function
from __future__ import division

import numpy as np
import os
import sys
import copy

BASE_DIR=(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from model import Model
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights/60_steps')
class Predictor():
    def __init__(self, c_o, c_g, c_d, time_steps):
        self.T = time_steps # time_steps to predict
        self.model = Model(3, 2, 25, 64, 64, 4, load_data = False)
        self.model.load_weights(WEIGHTS_PATH)
        # self.model = model
        self.trajectory_length = self.model.env_time_step
        self.input_sequence_len = self.model.time_steps
        self.relative_state = np.zeros([self.trajectory_length + self.T, 7]) # all relative to object coordinate, (delta_x,delta_y,delta_theta,x_robot, y_robot, action_x, action_y)
        self.absolute_state = np.zeros([self.trajectory_length + self.T, 5]) # all absolute positions (x_object,y_object,theta_object,x_robot,y_robot)

        self.object_absolute_trajectory_imagine = []# the shooting trajectory
        self.robot_absolute_trajectory_imagine = []


        self.obstacle_position = np.zeros([2]) # obstacle absolute position
        self.goal_position = np.zeros([3]) # goal_position and orientation absolute data(relative to world coordinate)
        self.count = 0 # reset to zero every catch_up, count how many states have been changed

        self.c_o = c_o # cost coefficient for obstacle distance
        self.c_g = c_g # cost coefficient for goal distance
        self.c_d = c_d # cost coefficient for distance between object and robot


    def catch_up(self, relative_state, absolute_state, obstacle_position, goal_position, step):
        """update the current state and trajectory history of this episode for this sample agent
        Args:
            relative_state: np.array(step + 1, 7)
            absolute_state: np.array(step + 1, 5)
            obstacle_position: np.array(2,)
            step: (int) timestep
        """
        assert relative_state.shape == (step + 1, 7)
        assert absolute_state.shape == (step + 1, 5)
        assert obstacle_position.shape == (2,)
        assert goal_position.shape == (3,)

        # relative state(for input of the model)
        self.relative_state[:(step + 1)] = relative_state[:] # robot action of the relative_state[step] is zeros(fake)
        self.relative_state[(step + 1):] = np.zeros([7])

        # absolute state(for cost_fun)
        self.absolute_state[:(step + 1)] = absolute_state[:]
        self.absolute_state[(step + 1):] = np.zeros([5])

        # obstacle position and goal position
        self.goal_position[:] = goal_position[:]
        self.obstacle_position[:] = obstacle_position[:]

        object_data = copy.copy(absolute_state[-1][:3])
        robot_data = copy.copy(absolute_state[-1][3:])
        # how many states it has predicted
        self.count = 0 #reset count

    def get_relative_action(self, action, step):
        object_pos = copy.copy(self.absolute_state[step + self.count][:3])
        theta = object_pos[2]
        action_x = action[0]
        action_y = action[1]
        action_relative_to_object_x = action_x*np.cos(theta) + action_y*np.sin(theta)
        action_relative_to_object_y = action_y*np.cos(theta) - action_x*np.sin(theta)
        return np.array([action_relative_to_object_x, action_relative_to_object_y])

    def predict(self, action, step):
        """action: relative to world coordinate"""
        assert action.shape == (2,)
        # action_ = 0.03*np.clip(action, -1, 1)
        action_ = copy.copy(action)
        input = np.zeros([self.input_sequence_len, 7])

        # transfer the action to object coordinate
        relative_action = self.get_relative_action(action_, step)

        # update the action data for current state which is set as (0. , 0.)
        self.relative_state[step + self.count][5:] = relative_action[:]

        # get input sequence for model, the model need self.input_sequence_len steps sequence as input
        for i in range(self.input_sequence_len):
            idx = i + step + self.count + 1 - self.input_sequence_len
            if idx < 0:
                input[i] = np.zeros([7])
                input[i][3:5] = copy.copy(self.relative_state[0][3:5])
            else:
                input[i] = copy.copy(self.relative_state[idx])

        input = input[np.newaxis, :]
        state_increment = self.model.predict(input) # [delta_x, delta_y, delta_theta]

        # update self.relative_state and self.count
        self.relative_state[step + self.count + 1][:3] = state_increment[:]
        self.relative_state[step + self.count + 1][3:5] = self.get_prediction_relative_position(state_increment, step)[:]

        # update self.absolute_state
        self.absolute_state[step + self.count + 1][:3] = self.get_prediction_absolute_position(state_increment, step)[:]
        robot_absolute = copy.copy(self.absolute_state[step + self.count][3:])
        self.absolute_state[step + self.count + 1][3:] = robot_absolute + action_

        # update self.object_absolute_trajectory_imagine and self.robot_absolute_trajectory_imagine
        object_data = copy.copy(self.absolute_state[step + self.count + 1][:3])
        robot_data = copy.copy(self.absolute_state[step + self.count + 1][3:])

        # compute the cost
        cost = self.cost_fun(object_data, robot_data, self.obstacle_position, self.goal_position)

        # update count
        self.count += 1
        return cost


    def cost_fun(self, object_position, robot_position, obstacle_position, goal_position):
        c_o = self.c_o
        c_g = self.c_g
        c_d = self.c_d
        object_position_ = object_position[:2]
        object_rotation_ = object_position[2]
        # fixed_pos = np.array([1.35, 0.65])
        goal_rotation_ = goal_position[2]
        goal_position_ = goal_position[:2]
        delta_theta = np.abs(object_rotation_ - goal_rotation_)

        # cost = c_g*np.squeeze(np.sum(np.square(object_position_ - goal_position[:2]))) + 0.5*c_d*np.squeeze(np.sum(np.square(object_position_ - robot_position)))
        # cost = c_d*np.squeeze(np.sum(np.square(fixed_pos - robot_position)))
        cost = c_g*np.squeeze(np.sum(np.square(object_position - goal_position)))
        return cost

    def get_prediction_relative_position(self, state_increment, step):
        """robot position relative to object"""
        x_original = copy.copy(self.relative_state[step + self.count][3])
        y_original = copy.copy(self.relative_state[step + self.count][4])
        action_x = copy.copy(self.relative_state[step + self.count][5])
        action_y = copy.copy(self.relative_state[step + self.count][6])

        coordinate_increment_x = state_increment[0]
        coordinate_increment_y = state_increment[1]
        coordinate_increment_theta = state_increment[2]
        # trasition without rotation
        x = x_original + action_x - coordinate_increment_x
        y = y_original + action_y - coordinate_increment_y
        # rotation
        x_relative_update = x*np.cos(coordinate_increment_theta) + y*np.sin(coordinate_increment_theta)
        y_relative_update = y*np.cos(coordinate_increment_theta) - x*np.sin(coordinate_increment_theta)

        return np.array([x_relative_update, y_relative_update])

    def get_prediction_absolute_position(self, state_increment, step):
        object_x_original = copy.copy(self.absolute_state[step + self.count][0])
        object_y_original = copy.copy(self.absolute_state[step + self.count][1])
        object_theta_original = copy.copy(self.absolute_state[step + self.count][2])

        increment_x = state_increment[0]
        increment_y = state_increment[1]
        increment_theta = state_increment[2]

        delta_x = increment_x*np.cos(object_theta_original) - increment_y*np.sin(object_theta_original)
        delta_y = increment_x*np.sin(object_theta_original) + increment_y*np.cos(object_theta_original)

        x_absolute = delta_x + object_x_original
        y_absolute = delta_y + object_y_original
        theta_absolute = object_theta_original + increment_theta

        return np.array([x_absolute, y_absolute, theta_absolute])


    def get_imagine_trajectory(self):
        object_traj = copy.copy(np.array(self.object_absolute_trajectory_imagine))
        robot_traj = copy.copy(np.array(self.robot_absolute_trajectory_imagine))
        return object_traj, robot_traj

def main():
    predictor = Predictor()
    predictor.test()

if __name__ == '__main__':
    main()
