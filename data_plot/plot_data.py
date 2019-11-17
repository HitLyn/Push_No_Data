import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation





def main():
    #load data
    # object_data = np.load('/home/lyn/policies/push-v9/object_data.npy')
    # robot_data = np.load('/home/lyn/policies/push-v9/robot_data.npy')


    object_data = np.load('/home/lyn/HitLyn/Push/new_data/object_pure_test.npy').reshape(-1, 13)
    # object_data = np.load('/home/lyn/HitLyn/Push/generated_trajectory/new_no_weights_loaded.npy').reshape(-1, 3)
    # object_data = np.load('/home/lyn/HitLyn/Push/generated_trajectory/pure_test.npy').reshape(-1, 3)
    robot_data = np.load('/home/lyn/HitLyn/Push/new_data/robot_pure_test.npy').reshape(-1, 6)

    #data process
    object_position_center = object_data[:, :2] # get the left bottom conrner of the object

    object_rotation_d = np.degrees(np.arccos(object_data[:, 3]) * 2).reshape(-1, 1) # rotation degrees
    object_rotation_r = (np.arccos(object_data[:, 3]) * 2).reshape(-1, 1)

    # object_rotation_d = np.degrees(object_data[:, 2]).reshape(-1, 1) # rotation degrees
    # object_rotation_r = (object_data[:, 2]).reshape(-1, 1)

    object_position_left_corner = object_position_center - 0.0707*np.concatenate((np.sin(np.pi/4 - object_rotation_r), np.cos(np.pi/4 - object_rotation_r)), axis = 1)

    robot_position = robot_data[:, :2]


    #plot
    fig = plt.figure()
    ax = plt.axes(xlim = (0.8, 1.8), ylim = (0.2, 1.2))
    patch_object = patches.Rectangle((1, 1), 0.1, 0.1, fc = 'y')
    patch_robot = patches.Circle((1, 1), 0.003, fc = 'b')

    def init_object():
        patch_object.set_xy([object_position_left_corner[0, 0], object_position_left_corner[0, 1]])
        patch_object.angle = object_rotation_d[0, 0]
        ax.add_patch(patch_object)

        patch_robot.center = (robot_position[0, 0], robot_position[0, 1])
        ax.add_patch(patch_robot)

        return patch_object, patch_robot,

    def init_robot():
        patch_robot.center = (robot_position[0, 0], robot_position[0, 1])
        ax.add_patch(patch_robot)
        return patch_robot,



    def animate_object(i):
        x = object_position_left_corner[i, 0]
        y = object_position_left_corner[i, 1]
        patch_object.set_xy([x, y])
        patch_object.angle = object_rotation_d[i, 0]

        x = robot_position[i, 0]
        y = robot_position[i, 1]
        patch_robot.center = (x, y)

        # patch_list = []

        # ax.add_patch(patch_object)

        # patch_list.append(ax.add_patch(patch_robot))
        # ax.add_patch(patch_object)

        return patch_object, patch_robot,
        # return tuple()

    def animate_robot(i):
        x = robot_position[i, 0]
        y = robot_position[i, 1]
        patch_robot.center = (x, y)

        return patch_robot,


    # anim_object = animation.FuncAnimation(fig, animate_object, init_func = init_object, frames = len(object_rotation_d), interval = 200, blit = True)
    # anim_robot = animation.FuncAnimation(fig, animate_robot, init_func = init_robot, frames = len(object_rotation_d), interval = 100, blit = True)
    anim_object = animation.FuncAnimation(fig, animate_object, init_func = init_object, frames = len(object_rotation_d), interval = 200, blit = True)
    # anim_object.save('example.mp4')

    plt.show()




if __name__ == '__main__':
    main()
