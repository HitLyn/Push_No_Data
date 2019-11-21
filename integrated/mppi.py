import numpy as np
from predictor import Predictor
from trajectory import Trajectory


class MPPI():
    """MPPI algorithm for pushing """
    def __init__(self, env, K, T):
        self.env = env
        self.predictor = Predictor()
        self.trajectory = Trajectory(self.env)
        self.K = K # K sample action sequences
        self.T = T # time steps per sequence
        self.lambda = 1

        self.dim_u = self.env.action_space.sample().shape[0]
        self.U = np.zeros([self.T, self.dim_u])
        self.time_limit = self.env._max_episode_steps

        self.u_init = np.zeros([self.dim_u])
        self.cost = np.zeros([self.K])
        self.noise = np.random.normal(loc = 0, scale = 1, size = (self.K, self.T, self.dim_u))

    def compute_cost(self, k, step):
        self.noise[k] = np.random.normal(loc = 0, scale = 1, size = (self.T, self.dim_u))
        self.predictor.catch_up(self.trajectory.get_relative_state(), self.trajectory.get_absolute_state(), self.trajectory.get_goal_position(), step) # make the shadow state the same as the actual robot and object state
        for t in range(self.T):
            action = self.U[t] + self.noise[k][t]
            cost = self.predictor.predict(action, step) # there will be shadow states in predictor
            self.cost[k] += cost


    def rollout(self, episode):
        for episode in range(episode):
            print('episode: {}'.format(episode))
            obs = self.env.reset()
            self.trajectory.reset()
            self.trajectory.update(obs)
            self.U = np.zeros([self.T, self.dim_u])
            for step in range(self.time_limit):
                for k in range(self.K):
                    self.compute_cost(k, step)

                beta = np.min(self.cost)
                eta = np.sum(np.exp((-1/self.lambda) * (self.cost - beta))) + 1e-6
                w = (1/eta) * np.exp((-1/self.lambda) * (self.cost - beta))

                self.U += [np.dot(w, self.noise[:, t]) for t in range(self.T)]
                obs, r, _, _ = self.env.step(self.U[0])
                self.trajectory.update(obs)
                self.env.render()
                self.U = np.roll(self.U, -1, axis = 0) #shift left
                self.U[-1] = self.u_init
                self.cost = np.zeros([self.K]) # reset cost
