import numpy as np
class Model():
    def __init__(self, x, t, y, c, h, u, load_data = False):
        self.env_time_step = 50
        self.time_steps = 10
    def predict(self, input):
        return np.random.normal(loc=0, scale=1.0, size=(3))
