from mppi_test import MPPI
import numpy as np
import os
import gym



ENV = 'FetchPush-v4'
TIMESTEPS = 10
N_SAMPLES = 400
EPISODE = 5



def main():
    env = gym.make(ENV)
    mppi = MPPI(env = env, K = N_SAMPLES, T = TIMESTEPS)
    mppi.rollout(episode = EPISODE)




if __name__ == '__main__':
    main()
