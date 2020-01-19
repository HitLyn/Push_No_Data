from __future__ import print_function
from __future__ import division
import numpy as np
import os
import copy
from predictor import Predictor
from trajectory import Trajectory


# WEIGHTS_PATH = '/home/lyn/HitLyn/Push/saved_model/epoch150/60_steps'
SAVE_PATH = '/home/lyn/HitLyn/Push/imagine_trajectory'
class MPPI():
    """MPPI algorithm for pushing """
    def __init__(self, K, T, A):
        self.A = A # action scale
        self.T = T # time steps per sequence
        self.predictor = Predictor(1, 2, 1, self.T)
        self.trajectory = Trajectory()
        self.trajectory.reset()
        self.K = K # K sample action sequences
        self.lambd = 1

        self.dim_u = 1 # U is theta
        # action init
        self.U_reset()

        self.u_init = np.array([0.0])
        self.cost = np.zeros([self.K])
        self.noise = np.random.normal(loc = 0, scale = 1, size = (self.K, self.T, self.dim_u))
        # self.noise = np.zeros([self.K, self.T, self.dim_u])

    def _compute_cost(self, k, step):
        # self.noise[k] = np.concatenate([np.random.normal(loc = 0, scale = 1, size = (self.T, 1)), np.random.rand(self.T, 1)], axis = 1)
        # self.noise[k] = np.clip(np.random.normal(loc = 0, scale = 2, size = (self.T, 1)), -1.57, 1.57)
        self.noise[k] = np.clip(np.random.normal(loc = 0, scale = 2, size = (self.T, 1)), -3.14, 3.14)
        eps = self.noise[k]
        self.predictor.catch_up(self.trajectory.get_relative_state(), self.trajectory.get_absolute_state(), self.trajectory.get_obstacle_position(), self.trajectory.get_goal_position(), step) # make the shadow state the same as the actual robot and object state
        for t in range(self.T):
            if t > 0:
                eps[t] = 0.4*eps[t - 1] + 0.6*eps[t]
            self.noise[k][t] = eps[t]
            theta = self.U[t] + eps[t]
            action = copy.copy(self.A * np.concatenate([np.sin(theta), np.cos(theta)]))
            cost = self.predictor.predict(action, step) # there will be shadow states in predictor
            self.cost[k] += cost

    def compute_cost(self, step):
        for k in range(self.K):
            self._compute_cost(k, step)

    def trajectory_clear(self):
        self.trajectory.reset()

    def trajectory_set_goal(self, pos_x, pos_y, theta):
        self.trajectory.set_goal(pos_x, pos_y, theta)

    def trajectory_update_state(self, pose_object, pose_tool):
        self.trajectory.update_state(pose_object, pose_tool)

    def trajectory_update_action(self, action):
        self.trajectory.update_action(action)

    def compute_noise_action(self):
        beta = np.min(self.cost)
        eta = np.sum(np.exp((-1/self.lambd) * (self.cost - beta))) + 1e-6
        w = (1/eta) * np.exp((-1/self.lambd) * (self.cost - beta))

        self.U += [np.dot(w, self.noise[:, t]) for t in range(self.T)]
        print(self.U)
        theta = self.U[0]
        action = copy.copy(self.A * np.concatenate([np.sin(theta), np.cos(theta)]))

        return action

    def U_reset(self):
        self.U = np.zeros([self.T, self.dim_u])
        # self.U[:, 1] = 0.8

    def get_K(self):
        return self.K

    def U_update(self):
        self.U = np.roll(self.U, -1, axis = 0)
        self.U[-1] = self.u_init

    def cost_clear(self):
        self.cost = np.zeros([self.K])
