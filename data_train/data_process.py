import numpy as np
import os


DATA_PATH = '/home/lyn/HitLyn/Push/original_data'
time_steps = 3

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


def get_sample_input(object_state, robot_state, episode, i):
    """
    Returns: np.array, size(time_steps, 4)
        """

    idx = episode * 101 + i
    state = np.concatenate((object_state[idx], robot_state[idx]))
    action = robot_state[idx + 1]
    sample_inputs = input_from_world_to_object(np.concatenate((state, action)))

    return sample_inputs

def get_sample_target(object_state, episode, i):
    """object_state
    Returns: np.array, size(time_steps, 3)
        """

    idx = episode * 101 + i
        # state = (object_state[idx + 1] - object_state[idx])
    sample_targets = target_from_world_to_object(object_state[idx + 1], object_state[idx])

    return sample_targets

def stack_time_step(state, timestep, episode, i):
     sample_inputs = []

     for step in range(timestep):
         idx = episode * 100 + i + step

         sample_inputs.append(state[idx])

     return np.array(sample_inputs)



def data_preprocess(object_data, robot_data):

    object_position = object_data[:, :2]
    object_rotation = (np.arccos(object_data[:, 3]) * 2).reshape(-1, 1)
    object_state = np.concatenate((object_position, object_rotation), axis = 1)

    assert object_state.shape == (len(object_position), 3)

    robot_state = robot_data[:, :2]
    assert robot_state.shape == (len(robot_state), 2)

    episode_num = len(object_state)//101 # how many episodes in the dataset: 101 timestep each
    sequences_num_per_episode = 101 - time_steps # how many sequences to generate per episode

    sample_inputs = np.zeros((episode_num * sequences_num_per_episode, time_steps, 4))
    sample_targets = np.zeros((episode_num * sequences_num_per_episode, time_steps, 3))

    state_inputs = np.zeros((episode_num * 100, 4))
    state_targets = np.zeros((episode_num * 100, 3))

    # get state
    for episode in range(episode_num):
        for i in range(100):
            idx = episode*100 + i
            state_inputs[idx] = get_sample_input(object_state, robot_state, episode, i)
            state_targets[idx] = get_sample_target(object_state, episode, i)

    # scale and normalization
    state_inputs[:, 2:] = state_inputs[:, 2:] * 5
    state_inputs = np.clip(state_inputs, -0.2, 0.2)
    mean = state_inputs.mean()
    std = state_inputs.std()
    state_inputs = (state_inputs - mean)/std

    # get inputs and targets
    for episode in range(episode_num):
        for i in range(sequences_num_per_episode):
            idx = episode * sequences_num_per_episode + i
            sample_inputs[idx] = stack_time_step(state_inputs, time_steps, episode, i)
            sample_targets[idx] = stack_time_step(state_targets, time_steps, episode, i)

    data_set = {}
    data_set['input'] = sample_inputs
    data_set['target'] = sample_targets


    return data_set

def main():
    object_data = np.load(os.path.join(DATA_PATH, 'object_data_half.npy')).reshape([-1, 13])
    robot_data = np.load(os.path.join(DATA_PATH, 'robot_data_half.npy')).reshape([-1, 6])
    # data_set = data_preprocess(object_data, robot_data)
    print(object_data.shape)
    print(robot_data.shape)

    # print(data_set['input'].shape, data_set['target'].shape, data_set_relative['input'].shape, data_set_relative['target'].shape)
    # print(data_set['input'][50], '\n', data_set['target'][50], '\n', data_set_relative['input'][50],'\n', data_set_relative['target'][50])
    # print(data_set['input'].shape, data_set['target'].shape)





if __name__ == '__main__':
    main()
