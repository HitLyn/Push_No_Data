import numpy as np
import os
import copy
from predictor_trotest import Predictor
# from real_predictor import Predictor
from trajectory_trotest import Trajectory

# WEIGHTS_PATH = '/home/lyn/HitLyn/Push/saved_model/epoch150/60_steps'
SAVE_PATH = '/home/lyn/HitLyn/Push/imagine_trajectory'
class MPPI():
    """MPPI algorithm for pushing """
    def __init__(self, env, K, T):
        self.env = env
        self.T = T # time steps per sequence
        self.predictor = Predictor(1, 2, 1, self.T)
        # self.predictor = Predictor(1, 1, 2, self.T)
        self.trajectory = Trajectory(self.env)
        self.trajectory.reset()
        self.K = K # K sample action sequences
        # self.T = T # time steps per sequence
        self.lambd = 1

        # self.dim_u = self.env.action_space.sample().shape[0]
        self.dim_u = 2
        self.U = np.zeros([self.T, self.dim_u])
        self.time_limit = self.env._max_episode_steps

        self.u_init = np.zeros([self.dim_u])
        self.cost = np.zeros([self.K])
        self.noise = np.random.normal(loc = 0, scale = 1, size = (self.K, self.T, self.dim_u))

    def compute_cost(self, k, step):
        self.noise[k] = np.random.normal(loc = 0, scale = 1, size = (self.T, self.dim_u))
        eps = self.noise[k]
        # print('noiseK: ', self.noise[k])
        self.predictor.catch_up(self.trajectory.get_relative_state(), self.trajectory.get_absolute_state(), self.trajectory.get_obstacle_position(), self.trajectory.get_goal_position(), step) # make the shadow state the same as the actual robot and object state
        for t in range(self.T):
            if t > 0:
                eps[t] = 0.8*eps[t - 1] + 0.2*eps[t]
            self.noise[k][t] = eps[t]
            action = copy.copy(self.U[t] + eps[t])
            # print('action: ', action)
            cost = self.predictor.predict(action, step) # there will be shadow states in predictor
            # print('predict step: ', t + 1, 'robot predict position: ', self.predictor.absolute_state[step + t + 1][3:])
            # print('robot actual position current: ', self.predictor.absolute_state[step][3:])
            self.cost[k] += cost


    def rollout(self, episode):
        for episode in range(episode):
            print('episode: {}'.format(episode))
            obs = self.env.reset()
            self.trajectory.reset()
            # self.predictor.reset()
            self.trajectory.update_state(obs)
            self.U = np.zeros([self.T, self.dim_u])
            for step in range(self.time_limit):
                print('step: ', step)
                # self.env.render()
                for k in range(self.K):
                    self.compute_cost(k, step)

                beta = np.min(self.cost)
                eta = np.sum(np.exp((-1/self.lambd) * (self.cost - beta))) + 1e-6
                w = (1/eta) * np.exp((-1/self.lambd) * (self.cost - beta))

                self.U += [np.dot(w, self.noise[:, t]) for t in range(self.T)]
                obs, r, _, _ = self.env.step(np.concatenate([self.U[0], np.zeros([2])]))
                self.trajectory.update_action(self.U[0])
                self.trajectory.update_state(obs)
                # self.trajectory.update_action(self.U[0])
                self.env.render()
                # print('step: ', step)
                self.U = np.roll(self.U, -1, axis = 0) #shift left
                self.U[-1] = self.u_init
                self.cost = np.zeros([self.K]) # reset cost

        # object_trajectory_imagine, robot_trajectory_imagine = self.predictor.get_imagine_trajectory()
        # # print(object_trajectory_imagine, robot_trajectory_imagine)
        # # for i in range(len(object_trajectory_imagine)):
        # #     print(object_trajectory_imagine[i])
        # # print(object_trajectory_imagine[i] for i in range(len(object_trajectory_imagine)))
        # np.save(os.path.join(SAVE_PATH, 'object'), object_trajectory_imagine)
        # np.save(os.path.join(SAVE_PATH, 'robot'), robot_trajectory_imagine)
