from __future__ import print_function
from __future__ import division
import numpy as np
import os
import copy
from dwa_predictor import DWA_Predictor
from trajectory import Trajectory


SAVE_PATH = '/home/lyn/HitLyn/Push/imagine_trajectory'
class DWA():
    """DWA algorithm for pushing """
    def __init__(self, K, T, A):
        self.A = A # action scale
        self.T = T # time steps per sequence
        self.predictor = DWA_Predictor(1, 2, 1, self.T)
        self.trajectory = Trajectory()
        self.trajectory.reset()
        self.K = K # K sample action sequences
        self.lambd = 1

        # self.u_init = np.array([0.0])
        self.cost = np.zeros([self.K])
        # self.noise = np.random.uniform(-3.14, 3.14, size = (self.K, self.T, self.dim_u))
        # self.noise = np.zeros([self.K, self.T, self.dim_u])
        self.random_theta = np.linspace(-np.pi/2, np.pi/2, num = self.K)

    def _compute_best_action_(self, k, step):
        # self.noise[k] = np.concatenate([np.random.normal(loc = 0, scale = 1, size = (self.T, 1)), np.random.rand(self.T, 1)], axis = 1)
        # self.noise[k] = np.clip(np.random.normal(loc = 0, scale = 2, size = (self.T, 1)), -1.57, 1.57)
        random_theta = self.random_theta[k]
        # eps = self.noise[k]
        self.predictor.catch_up(self.trajectory.get_relative_state(), self.trajectory.get_absolute_state(), self.trajectory.get_obstacle_position(), self.trajectory.get_goal_position(), step) # make the shadow state the same as the actual robot and object state
        for t in range(self.T):
            action = copy.copy(self.A * np.power(0.5, t) * np.array([np.sin(random_theta), np.cos(random_theta)]))
            cost = self.predictor.predict(action, step) # there will be shadow states in predictor
            self.cost[k] += cost

    def compute_best_action(self, step):
        for k in range(self.K):
            self._compute_best_action_(k, step)

        idx = np.argmin(self.cost)
        print('theta_chosen: ', idx)
        theta = self.random_theta[idx]
        action = self.A * np.array([np.sin(theta), np.cos(theta)])
        return action

    def trajectory_clear(self):
        self.trajectory.reset()

    def trajectory_set_goal(self, pos_x, pos_y, theta):
        self.trajectory.set_goal(pos_x, pos_y, theta)

    def trajectory_update_state(self, pose_object, pose_tool):
        self.trajectory.update_state(pose_object, pose_tool)

    def trajectory_update_action(self, action):
        self.trajectory.update_action(action)


    def get_K(self):
        return self.K


    def cost_clear(self):
        self.cost = np.zeros([self.K])
