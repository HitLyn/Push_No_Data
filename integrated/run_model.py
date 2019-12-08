from mppi_trotest import MPPI
import numpy as np
import os
import gym



ENV = 'FetchPush-v5'
TIMESTEPS = 10
N_SAMPLES = 80
EPISODE = 50



def main():
    env = gym.make(ENV)
    env.reset()
    # print('env built')
    mppi = MPPI(env = env, K = N_SAMPLES, T = TIMESTEPS)
    # print('mppi built')
    mppi.rollout(episode = EPISODE)




if __name__ == '__main__':
    main()
