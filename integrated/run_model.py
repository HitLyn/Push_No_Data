from mppi import MPPI
import numpy as np
import os
import gym



ENV = 'FetchPush-mppi'
TIMESTEPS = 20
N_SAMPLES = 1000
EPISODE = 10



def main():
    env = gym.make(ENV)
    mppi = MPPI(env = env, K = N_SAMPLES, T = TIMESTEPS)
    mppi.rollout(episode = EPISODE)




if __name__ == '__main__':
    main()
