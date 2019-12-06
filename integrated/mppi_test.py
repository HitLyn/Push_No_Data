import numpy as np
from predictor_test import Predictor
from trajectory_test import Trajectory
# import ray
import multiprocessing as mp


def compute_cost_parallel(predictor_args, initial_state, rollout_num_per_cpu, base_act, step):
    num_cpu = mp.cpu_count()
    args_list = []
    for i in range(num_cpu):
        args_cpu = [predictor_args, initial_state, rollout_num_per_cpu, base_act, step]
        args_list.append(args_cpu)

    results = _try_multiprocess(args_list, num_cpu, max_process_time = 3000, max_timeout = 4)
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    return paths

def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeout):
    if max_timeout == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
    parallel_runs = [pool.apply_async(cpu_compute_cost_, args = (args_list[i],)) for i in range(num_cpu)]
    try:
        results = [p.get(timeout = max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("timeout error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(args_list, num_cpu, max_process_time, max_timeout - 1)

    pool.close()
    pool.terminate()
    pool.join()

    return results

def cpu_compute_cost_(args_list):
    return cpu_compute_cost(*args_list)

def cpu_compute_cost(predictor_args, initial_state, rollout_num_per_cpu, base_act, step):
    act_list = []
    noise_list = []
    for i in range(rollout_num_per_cpu):
        act, noise = generate_random_actions(base_act)
        act_list.append(act)
        noise_list.append(noise)
    paths = do_predictor_rollout(predictor_args, initial_state, act_list, noise_list, step)

    return paths

def generate_random_actions(base_act):

    eps = np.random.normal(loc=0, scale=1.0, size=base_act.shape)
    return eps + base_act, eps

def do_predictor_rollout(predictor_args, initial_state, act_list, noise_list, step):
    """act_list: list with num_rollout_per_cpu elements, each element is np.array with size (H, dim_u)"""
    predictor = Predictor(*predictor_args)
    paths = []
    N = len(act_list)
    H = act_list[0].shape[0]
    for i in range(N):
        predictor.catch_up(*initial_state)
        act = []
        noise = []
        obs = []
        cost = []
        for k in range(H):
            obs.append(predictor._get_obs())
            act.append(act_list[i][k])
            noise.append(noise_list[i][k])
            c = predictor.predict(act[-1], step)
            cost.append(c)

        path = dict(observations = np.array(obs), actions = np.array(act), costs = np.array(cost), noise = np.array(noise))
        paths.append(path)

    return paths

def score_trajectory(paths):
    costs = np.zeros(len(paths))
    for i in range(len(paths)):
        costs[i] = 0
        for t in range(paths[i]['costs'].shape[0]):
            costs[i] += paths[i]['costs'][t]
    return costs

def stack_noise(paths):
    K = len(paths)
    T = paths[0]['noise'].shape[0]
    dim_u = paths[0]['noise'].shape[1]
    noise = np.random.normal(loc = 0, scale = 1, size = (K, T, dim_u))
    for i in range(len(paths)):
        for t in range(paths[i]['noise'].shape[0]):
            noise[i, t] = paths[i]['noise'][t]
    return noise

class MPPI():
    """MPPI algorithm for pushing """
    def __init__(self, env, K, T):
        self.env = env
        self.T = T # time steps per sequence

        self.predictor_args = [self.env, 1, 3, 1, self.T]

        self.trajectory = Trajectory(self.env)
        self.K = K # K sample action sequences
        # self.T = T # time steps per sequence
        self.lambd = 1

        # self.dim_u = self.env.action_space.sample().shape[0]
        self.dim_u = 2
        self.U = np.zeros([self.T, self.dim_u])
        self.base_act = np.zeros([self.T, self.dim_u])
        self.time_limit = self.env._max_episode_steps

        self.u_init = np.zeros([self.dim_u])
        self.cost = np.zeros([self.K])
        self.noise = np.random.normal(loc = 0, scale = 1, size = (self.K, self.T, self.dim_u))


    # @ray.remote
    def compute_cost(self, k, step):
        self.noise[k] = np.random.normal(loc = 0, scale = 1, size = (self.T, self.dim_u))
        self.predictor.catch_up(self.trajectory.get_goal_state(), self.trajectory.get_initial_state(), self.trajectory.get_action_history(), step) # make the shadow state the same as the actual robot and object state
        for t in range(self.T):
            action = self.U[t] + self.noise[k][t]
            cost = self.predictor.predict(action, step) # there will be shadow states in predictor
            self.cost[k] += cost
        return self.cost[k]

    def mppi_compute_cost_parallel(self, predictor_args, initial_state, rollout_num_per_cpu, base_act, step):
        return compute_cost_parallel(predictor_args, initial_state, rollout_num_per_cpu, base_act, step)

    def rollout(self, episode):
        for episode in range(episode):
            print('episode: {}'.format(episode))
            obs = self.env.reset()
            self.trajectory.reset()
            self.trajectory.state_update(obs)
            self.U = np.zeros([self.T, self.dim_u])
            rollout_num_per_cpu = self.K//mp.cpu_count()

            for step in np.arange(self.time_limit):
                # update current_state
                current_state = [self.trajectory.get_goal_state(), self.trajectory.get_initial_state(), self.trajectory.get_action_history(), step]
                # compute costs for K sample in multiprocess
                paths = self.mppi_compute_cost_parallel(self.predictor_args, current_state, rollout_num_per_cpu, self.U, step) # list of K elements, each element is a dict{observation: (self.T,), actions: (self.T, 2), costs: (self.T,)}
                # get cost from paths
                self.cost = score_trajectory(paths)
                # get actions from paths
                self.noise = stack_noise(paths)

                beta = np.min(self.cost)
                eta = np.sum(np.exp((-1/self.lambd) * (self.cost - beta))) + 1e-6
                w = (1/eta) * np.exp((-1/self.lambd) * (self.cost - beta))

                self.U += [np.dot(w, self.noise[:, t]) for t in range(self.T)]
                obs, r, _, _ = self.env.step(np.concatenate([self.U[0], np.zeros([2])]))
                self.trajectory.action_update(np.concatenate([self.U[0], np.zeros([2])]))
                # self.trajectory.state_update(obs)
                self.env.render()
                print('step: ', step)
                self.U = np.roll(self.U, -1, axis = 0) #shift left
                self.U[-1] = self.u_init
                self.cost = np.zeros([self.K]) # reset cost
