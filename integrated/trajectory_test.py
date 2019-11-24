import numpy as np


class Trajectory():
    """collect trajectory history and preprocess the data making it more suitable for the input of predictor"""
    def __init__(self, env):
        self.env = env
        self.goal = np.zeros([7])
        self.object_initial_state = np.zeros([7])
        self.action_history = []

    def get_initial_state(self):
        return self.object_initial_state

    def get_goal_state(self):
        return self.goal

    def get_action_history(self):
        return np.asarray(self.action_history)

    def reset(self):
        # reset the count num
        self.action_history.clear()

        self.goal = np.zeros([7])
        self.object_initial_state = np.zeros([7])

    def state_update(self, obs):
        self.goal = obs['desired_goal']
        self.object_initial_state = obs['achieved_goal']

    def action_update(self, action):
        self.action_history.append(action)
