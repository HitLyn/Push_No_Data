import numpy as np
import os
import sys

BASE_DIR=(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from model import Model

WEIGHTS_PATH = '/home/lyn/HitLyn/Push/saved_model_randomized/epoch150/60_steps'
DATA_PATH = '/home/lyn/HitLyn/Push/new_data_randomized/real'
SAVED_DATA_PATH = '/home/lyn/HitLyn/Push/generated_trajectory_randomized/real'


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

def vel_from_world_to_object(object_state_current, object_state_before):
    delta_position_x, delta_position_y, delta_theta = (i for i in (object_state_current - object_state_before))
    theta = object_state_current[2]
    vel_relative_to_object_x = delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    vel_relative_to_object_y = delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)

    return np.array([vel_relative_to_object_x, vel_relative_to_object_y, delta_theta])

def get_object_state_increment_world_coordinate(model, step, time_sequence, trajectory, robot_state):
    """get the incremental state of the object by utlizing the LSTM model we trained
    Args:
        step(int): the step num
        trajectory(array[101, 3]): trajectory of the object of this episode, updated once after each step, step by step(world coordinate)
        robot_state(array[101, 2]): the robot state of this episode(world coordinate)
    Returns:
        incremental state(array[3,])
    """
    # initialize inputs for model to predict
    input = np.zeros([time_sequence, 7])
    for i in range(time_sequence):
        object_state_current = trajectory[step - time_sequence + i]
        robot_position = robot_state[step - time_sequence + i]
        robot_action = robot_state[step - time_sequence + 1 + i]
        state = np.concatenate([object_state_current, robot_position, robot_action])
        robot_input = input_from_world_to_object(state)
        input[i][3:] = robot_input
        # get object velocity
        if step == time_sequence:
            vel = np.zeros([3])
        else:
            object_state_before = trajectory[step - time_sequence + i - 1]
            vel = vel_from_world_to_object(object_state_current, object_state_before)
        input[i][:3] = vel


    incremental_state = model.predict(input[np.newaxis, :])
    # relative coordinate
    x = incremental_state[0]
    y = incremental_state[1]
    delta_theta = incremental_state[2]

    theta = trajectory[step - 1][2]

    # turn it into world coordinate
    incremental_x_world_coordinate = x*np.cos(theta) - y*np.sin(theta)
    incremental_y_world_coordinate = x*np.sin(theta) + y*np.cos(theta)

    return np.array([incremental_x_world_coordinate, incremental_y_world_coordinate, delta_theta])

def get_object_state_increment_world_coordinate_original(step, time_sequence, episode, original_output, trajectory):
    """get the incremental state of the object original
    Args:
        step(int): the step num
        trajectory(array[101, 3]): trajectory of the object of this episode, updated once after each step, step by step(world coordinate)
        robot_state(array[101, 2]): the robot state of this episode(world coordinate)
    Returns:
        incremental state(array[3,])
    """
    # initialize inputs for model to predict
    incremental_state = original_output[episode][step - time_sequence]

    # relative coordinate
    x = incremental_state[0]
    y = incremental_state[1]
    delta_theta = incremental_state[2]

    theta = trajectory[step - 1][2]

    # turn it into world coordinate
    incremental_x_world_coordinate = x*np.cos(theta) - y*np.sin(theta)
    incremental_y_world_coordinate = x*np.sin(theta) + y*np.cos(theta)

    return np.array([incremental_x_world_coordinate, incremental_y_world_coordinate, delta_theta])


def main():

    test_robot_data_raw = np.load(os.path.join(DATA_PATH, 'robot_data_.npy'))
    test_object_data_raw = np.load(os.path.join(DATA_PATH, 'object_data_.npy'))
    # test_robot_data_raw = np.load(os.path.join(DATA_PATH, 'robot_data.npy')).reshape(-1, 3)
    # test_object_data_raw = np.load(os.path.join(DATA_PATH, 'object_data.npy')).reshape(-1, 7)

    model = Model(3, 2, len(test_robot_data_raw), 64, 64, 4, load_data = False)
    env_step = model.env_time_step
    time_sequence = model.time_steps
    model.model.load_weights(WEIGHTS_PATH)

    test_object_position = test_object_data_raw[:, :2]
    test_object_rotation = (np.arccos(test_object_data_raw[:, 6]) * 2).reshape(-1, 1)
    test_object_state = np.concatenate((test_object_position, test_object_rotation), axis = 1).reshape(-1, env_step, 3) # [episode_num, 101, 3]

    test_robot_state = test_robot_data_raw[:, :2].reshape(-1, env_step, 2) # [episode_num, 101, 2]

    # test_data_set = model.data_preprocess(test_object_data_raw, test_robot_data_raw)

    # predict = model.model.predict(test_data_set['input'])
    # error = predict - target

    ### generate predict trajectories ####
    episode_num = int(len(test_robot_state))
    ### initialize ###
    predict_trajectory = np.zeros([episode_num, env_step, 3])
    original_trajectory = np.zeros([episode_num, env_step, 3])

    original_data_set = model.data_preprocess(test_object_data_raw, test_robot_data_raw)
    original_output = original_data_set['target']
    original_output = original_output.reshape(episode_num, -1, time_sequence, 3)
    assert original_output.shape == (episode_num, (env_step - 1)//time_sequence, time_sequence, 3)

    #### loop ####
    for episode in range(episode_num):
        trajectory_p = np.zeros([env_step, 3]) # the first time_sequence points are regarded as the same as targets because we need these steps to predict
        # trajectory_o = np.zeros([env_step, 3])
        trajectory_p[:time_sequence] = test_object_state[episode][:time_sequence]
        # trajectory_o[:time_sequence] = test_object_state[episode][:time_sequence]

        robot_state = test_robot_state[episode]
        # print(robot_state.shape)
        for step in range(time_sequence, env_step):
            trajectory_p[step] = get_object_state_increment_world_coordinate(model, step, time_sequence, trajectory_p, robot_state) + trajectory_p[step - 1]
            # trajectory_o[step] = get_object_state_increment_world_coordinate_original(step, time_sequence, episode, original_output, trajectory_o) + trajectory_o[step - 1]

        predict_trajectory[episode] = trajectory_p
        # original_trajectory[episode] = trajectory_o

        template = 'Generate {} episodes.'
        print(template.format(episode + 1))

    # print(predict_trajectory.shape)
    print('finished')
    print('predict trajectory: ', predict_trajectory.shape)
    np.save(os.path.join(SAVED_DATA_PATH, 'real'), predict_trajectory)
    # np.save(os.path.join(SAVED_DATA_PATH, 'original'), original_trajectory)



if __name__ == '__main__':
    main()
