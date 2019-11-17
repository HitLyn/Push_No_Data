import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation





def main():
    #load data
    object_data = np.load('/home/lyn/policies/push-v9/object_data.npy')
    robot_data = np.load('/home/lyn/policies/push-v9/robot_data.npy')

    #data process
    object_position_center = object_data[:100, :2] # get the left bottom conrner of the object
    object_rotation_d = np.degrees(np.arccos(object_data[:100, 3]) * 2).reshape(-1, 1) # rotation degrees
    object_rotation_r = (np.arccos(object_data[:100, 3]) * 2).reshape(-1, 1)
    object_position_left_corner = object_position_center - 0.0707*np.concatenate((np.sin(np.pi/4 - object_rotation_r), np.cos(np.pi/4 - object_rotation_r)), axis = 1)

    robot_position = robot_data[:, :2]


    #plot
    fig = plt.figure()
    ax = plt.axes(xlim = (1, 2), ylim = (0, 1))
    # patch_object = patches.Rectangle((1, 1), 0.1, 0.1, fc = 'y')
    # patch_robot = patches.Circle((1, 1), 0.003, fc = 'b')

    patch_object_list = []
    patch_robot_list = []
    for i in range(len(object_rotation_d)):
        if i % 5 == 0:
            object_patch = patches.Rectangle((object_position_left_corner[i, 0], object_position_left_corner[i, 1]), 0.1, 0.1, fc = 'w', ec = 'r')
            object_patch.angle = object_rotation_d[i, 0]
            patch_object_list.append(object_patch)
            patch_robot_list.append(patches.Circle((robot_position[i, 0], robot_position[i, 1]), 0.003, fc = 'b'))

    for patch in patch_object_list:
        ax.add_patch(patch)

    for patch in patch_robot_list:
        ax.add_patch(patch)




    plt.show()




if __name__ == '__main__':
    main()
