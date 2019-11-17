import numpy as np
import matplotlib.pyplot as plt
import os


PATH = '/home/lyn/policies/push-v9'

def data_preprocess_normal(object_data, robot_data):

    object_position = object_data[:, :2]
    object_rotation = (np.arccos(object_data[:, 3]) * 2).reshape(-1, 1)
    object_state = np.concatenate((object_position, object_rotation), axis = 1)

    assert object_state.shape == (len(object_position), 3)

    robot_state = robot_data[:, :2]
    assert robot_state.shape == (len(robot_state), 2)

    episode_num = len(object_state)//101 # how many episodes in the dataset: 101 timestep each
    sequences_num_per_episode = 101 - 1 # how many sequences to generate per episode

    sample_inputs = np.zeros((episode_num * sequences_num_per_episode, 1, 4))
    sample_targets = np.zeros((episode_num * sequences_num_per_episode, 1, 3))

    # mget inputs and targets
    for episode in range(episode_num):
        for i in range(sequences_num_per_episode):
            idx = episode * sequences_num_per_episode + i
            sample_inputs[idx] = get_sample_input_relative(object_state, robot_state, 1, episode, i)
            sample_targets[idx] = get_sample_target_relative(object_state, 1, episode, i)

    data_set = {}
    data_set['input'] = np.squeeze(sample_inputs)
    data_set['target'] = np.squeeze(sample_targets)

    return data_set

def get_sample_input_relative(object_state, robot_state, time_steps, episode, i):
    """
    Returns: np.array, size(time_steps, 7)
        """
    sample_inputs = np.zeros((time_steps, 4))

    for step in range(time_steps):
        idx = episode * 101 + i + step
        state = np.concatenate((object_state[idx], robot_state[idx]))
        action = robot_state[idx + 1]
        sample_inputs[step] = input_from_world_to_object(np.concatenate((state, action)))

    return sample_inputs

def get_sample_target_relative(object_state, time_steps, episode, i):
    """
    Returns: np.array, size(time_steps, 3)
        """
    sample_targets = np.zeros((time_steps, 3))
    for step in range(time_steps):
        idx = episode * 101 + i +step
        # state = object_state[idx + 1] - object_state[idx]
        sample_targets[step] = target_from_world_to_object(object_state[idx + 1], object_state[idx])

    return sample_targets

def input_from_world_to_object(state):
    """turn a vector which is relative to world coordinate to object coordinate, change the input of the network!!!!
    Args:
        object_x: position x of object (in world coordinate)
        object_y: position y of object (in world coordinate)
        theta: object's rotation in anticlockwise (radians)
        robot_x: position x of robot (in world coordinate)
        robot_y: position y of robot (in world coordinate)
        robot_x_s: nextstep position x of robot (in world coordinate)
        robot_y_s: nextstep position y of robot (in world coordinate)
    Returns:
        (robot_x, robot_y, action_x, action_y): relative position and action of robot to object(in object coordinate)"""
    assert state.shape == (7,)
    object_x, object_y, theta, robot_x, robot_y, robot_x_s, robot_y_s = (i for i in state)
    delta_position_x, delta_position_y = robot_x - object_x, robot_y - object_y
    delta_action_x, delta_action_y = robot_x_s - robot_x, robot_y_s - robot_y
    robot_position_relative_to_object_x = delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    robot_position_relative_to_object_y = delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)
    action_relative_to_object_x = delta_action_x*np.cos(theta) + delta_action_y*np.sin(theta)
    action_relative_to_object_y = delta_action_y*np.cos(theta) - delta_action_x*np.sin(theta)

    return np.array([robot_position_relative_to_object_x, robot_position_relative_to_object_y, action_relative_to_object_x, action_relative_to_object_y])

def target_from_world_to_object(object_state_s, object_state):
    delta_position_x, delta_position_y, delta_theta = (i for i in (object_state_s - object_state))
    theta = object_state[2]
    delta_position_relative_to_object_x =  delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    delta_position_relative_to_object_y =  delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)

    return np.array([delta_position_relative_to_object_x, delta_position_relative_to_object_y, delta_theta])





def main():
    # load data
    robot_data = np.load(os.path.join(PATH, 'robot_data.npy'))
    object_data = np.load(os.path.join(PATH, 'object_data.npy'))

    # preprocess the data to input state for network
    data = data_preprocess_normal(object_data, robot_data)
    data_input = data['input']

    position_input = data_input[:, :2]
    action_input = data_input[:, 2:]

    ##################### plot ########################
    position_x = position_input[:, 0]
    position_y = position_input[:, 1]

    action_x = action_input[:, 0]
    action_y = action_input[:, 1]

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.scatter(position_x, position_y)

    fig.add_subplot(1,2,2)
    plt.scatter(action_x, action_y)

    plt.show()
























if __name__ == '__main__':
    main()
