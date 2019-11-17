import gym
import time
import numpy as np

"""Generate training data for model"""


height_offset = 0.1# offset between object and mocap
data = []


def main():
    # env setup
    env = gym.make('FetchPush-v1')
    env = env.unwrapped
    env.render()
    time.sleep(1)
    env.reset()
    env.render()
    time.sleep(1)
    # env.render()

    # generate start positions around box for stick
    start_position = generate_position(200, env)
    for position in start_position:
        # print('goal position: ' ,position)
        # print('object position: ' ,sim.data.get_site_xpos('object0'))
        move_stick_to(position, env)

    #     trajectory = move_around()
    #     data.append(trajectory)
    #
    # save(data)



def save(data):
    """output the data as a csv file"""
    pass





def move_around():
    """move the stick around the box to generate trajectory
        Return:
            trajectory(np.array)
    """
    pass




def move_stick_to(position, env):
    """move the stick to the goal position
        Args:
            position(np.array):a starting position

    """
    # env.reset()
    env.sim.data.mocap_pos[:] = position
    while(np.sum(np.square(position[:2] - env.sim.data.get_geom_xpos('robot0:stick')[:2])) > 1e-7):
        env.render()
        env.sim.step()
        print('move to the position.')

    print('ready!')
    time.sleep(0.1)




def generate_position(n, env):
    """generate start position according to the initial position of the box list for stick
        Args:
            n(int): how many positions to generate
            env(gym env object): the env to imply

        Return:
            position_list(list(np.array))
    """
    object_pos = env.sim.data.get_site_xpos('object0')
    stick_pos_x = object_pos[0] - 0.08
    stick_pos_y_start = object_pos[1] -0.07
    stick_pos_y_end = object_pos[1] + 0.07
    stick_pos_z = object_pos[2] + height_offset

    stick_pos_y = np.linspace(stick_pos_y_start, stick_pos_y_end, num = n)
    positions = np.zeros([n, 3])
    positions[:, 0] = stick_pos_x
    positions[:, 1] = stick_pos_y
    positions[:, 2] = stick_pos_z

    return positions





if __name__ == '__main__':
    main()
