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
    def __init__(self, K, T):
        self.T = T # time steps per sequence
        self.predictor = Predictor(1, 2, 1, self.T)
        self.trajectory = Trajectory()
        self.trajectory.reset()
        self.K = K # K sample action sequences
        self.lambd = 1

        self.dim_u = 2
        # action init
        self.U_reset()

        self.u_init = np.zeros([self.dim_u])
        self.cost = np.zeros([self.K])
        self.noise = np.random.normal(loc = 0, scale = 1, size = (self.K, self.T, self.dim_u))

    def _compute_cost(self, k, step):
        self.noise[k] = np.random.normal(loc = 0, scale = 1, size = (self.T, self.dim_u))
        eps = self.noise[k]
        self.predictor.catch_up(self.trajectory.get_relative_state(), self.trajectory.get_absolute_state(), self.trajectory.get_obstacle_position(), self.trajectory.get_goal_position(), step) # make the shadow state the same as the actual robot and object state
        for t in range(self.T):
            if t > 0:
                eps[t] = 0.8*eps[t - 1] + 0.2*eps[t]
            self.noise[k][t] = eps[t]
            action = copy.copy(self.U[t] + eps[t])
            cost = self.predictor.predict(action, step) # there will be shadow states in predictor
            self.cost[k] += cost

    def compute_cost(self, step):
        for k in range(self.K):
            self._compute_cost(k, step)

    def trajectory_clear(self):
        self.trajectory.reset()

    def trajectory_update_state(self, pose_object, pose_tool):
        self.trajectory.update_state(pose_object, pose_tool)

    def trajectory_update_action(self, action):
        self.trajectory.update_action(action)

    def compute_noise_action(self):
        beta = np.min(self.cost)
        eta = np.sum(np.exp((-1/self.lambd) * (self.cost - beta))) + 1e-6
        w = (1/eta) * np.exp((-1/self.lambd) * (self.cost - beta))

        self.U += [np.dot(w, self.noise[:, t]) for t in range(self.T)]
        action = self.U[0]

        return action

    def U_reset(self):
        self.U = np.zeros([self.T, self.dim_u])
        self.U[:, 0] = 1

    def get_K(self):
        return self.K

    def U_update(self):
        self.U = np.roll(self.U, -1, axis = 0)
        self.U[-1] = self.u_init

    def cost_clear(self):
        self.cost = np.zeros([self.K])
