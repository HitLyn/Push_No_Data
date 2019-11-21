import numpy as np

class Trajectory():
    """collect trajectory history and preprocess the data making it more suitable for the input of predictor"""
    def __init__(self, env):
        self.env = env

    def get_relative_state(self):
        pass

    def get_absolute_state(self):
        pass

    def get_goal_position(self):
        pass

    def reset(self):
        pass

    def update(self, obs):
        pass
