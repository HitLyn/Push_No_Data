from multi_process_mppi import MPPI
import numpy as np
import os
import gym



ENV = 'FetchPush-v4'
TIMESTEPS = 10
N_SAMPLES = 400
EPISODE = 5



def main():
    env = gym.make(ENV)
    # print('env built')
    mppi = MPPI(env = env, K = N_SAMPLES, T = TIMESTEPS)
    # print('mppi built')
    mppi.rollout(episode = EPISODE)




if __name__ == '__main__':
    main()
