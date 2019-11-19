import numpy as np

class MPPI():
    """MPPI algorithm for pushing """
    def __init__(self, env, K, T):
        self.env = env
        self.K = K # K sample action sequences
        self.T = T # time steps per sequence
        self.lambda = 1

        self.dim_u = self.env.action_space.sample().shape[0]
        self.U = np.zeros([self.T, self.dim_u])
        self.time_limit = self.env._max_episode_steps

        self.u_init = np.zeros([self.dim_u])
        self.cost = np.zeros([self.K])
        self.noise = np.random.normal(loc = 0, scale = 1, size = (self.K, self.T, self.dim_u))

    def compute_cost(self, k):
        self.noise[k] = np.random.normal(loc = 0, scale = 1, size = (self.T, self.dim_u))
        for t in range(self.T):
            action = self.U[t] + self.noise[k][t]
            cost = self.env.predict(action) # there will be shadow states in gym.env
            self.cost[k] += cost


    def rollout(self, episode):
        for episode in range(episode):
            print('episode: {}'.format(episode))
            self.env.reset()
            self.U = np.zeros([self.T, self.dim_u])
            for step in range(self.time_limit):
                for k in range(self.K):
                    self.compute_cost(k)

                beta = np.min(self.cost)
                eta = np.sum(np.exp((-1/self.lambda) * (self.cost - beta))) + 1e-6
                w = (1/eta) * np.exp((-1/self.lambda) * (self.cost - beta))

                self.U += [np.dot(w, self.noise[:, t]) for t in range(self.T)]
                s, r, _, _ = self.env.step(self.U[0])
                self.env.render()
                self.U = np.roll(self.U, -1, axis = 0) #shift lest
                self.U[-1] = self.u_init
                self.cost = np.zeros([self.K]) # reset cost
