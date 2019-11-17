import numpy as np
import matplotlib.pyplot as plt
import os

def main():

    object_data = np.load('/home/lyn/HitLyn/Push/original_data/object_data_60.npy')
    robot_data = np.load('/home/lyn/HitLyn/Push/original_data/robot_data_60.npy')


    i = 104
    object_episode = object_data[i]
    robot_episode = robot_data[i]



    fig = plt.figure()
    fig.add_subplot(2,1,1)
    # plt.plot(object_episode[:, 0], object_episode[:, 1])
    plt.plot(object_episode[:, 2])
    # plt.plot(object_episode[:, 7])
    # plt.plot(object_episode[:, 8])

    fig.add_subplot(2,1,2)
    plt.plot(object_episode[:, 9])
    # plt.plot(object_episode[:, 10])
    # plt.plot(object_episode[:, 11])
    # plt.plot(object_episode[:, 12])


    # fig.add_subplot(2,2,2)
    # plt.plot(error[:, 0])
    #
    # fig.add_subplot(2,2,3)
    # plt.plot(generated_trajectory[:, 1], 'b')
    # plt.plot(original_trajectory[:, 1], 'r')
    #
    # fig.add_subplot(2,2,4)
    # plt.plot(error[:, 1])

    plt.show()

if __name__ == '__main__':
    main()
