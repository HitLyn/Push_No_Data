import numpy as np
import os
import sys

BASE_DIR=(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from data_train.model_2 import Model

WEIGHTS_PATH = '/home/lyn/HitLyn/Push/saved_model/epoch150/log'
DATA_PATH = '/home/lyn/HitLyn/Push/original_data'
SAVED_DATA_PATH = '/home/lyn/HitLyn/Push/generated_trajectory'


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

def get_object_state_increment_world_coordinate(model, step, trajectory, robot_state):
    """get the incremental state of the object by utlizing the LSTM model we trained
    Args:
        step(int): the step num
        trajectory(array[101, 3]): trajectory of the object of this episode, updated once after each step, step by step(world coordinate)
        robot_state(array[101, 2]): the robot state of this episode(world coordinate)
    Returns:
        incremental state(array[3,])
    """
    # initialize inputs for model to predict
    input = np.zeros([3, 4])
    for i in range(3):
        object_state = trajectory[step - 3 + i]
        robot_position = robot_state[step - 3 + i]
        robot_action = robot_state[step - 2 + i]
        state = np.concatenate([object_state, robot_position, robot_action])
        input_not_normalization = input_from_world_to_object(state) ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! normalization!!!!!!!!!!!!

        input_not_normalization[2:] = input_not_normalization[2:]*5
        input_not_normalization = np.clip(input_not_normalization, -0.2, 0.2)
        input[i] = (input_not_normalization - model.mean)/model.std


    incremental_state = np.squeeze(model.predict(input[np.newaxis, :]))
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
    model = Model(3, 2, 16, 256, 3)
    # model.model.load_weights(WEIGHTS_PATH)

    test_robot_data_raw = np.load(os.path.join(DATA_PATH, 'test_robot_data.npy'))
    test_object_data_raw = np.load(os.path.join(DATA_PATH, 'test_object_data.npy'))

    test_object_position = test_object_data_raw[:, :2]
    test_object_rotation = (np.arccos(test_object_data_raw[:, 3]) * 2).reshape(-1, 1)
    test_object_state = np.concatenate((test_object_position, test_object_rotation), axis = 1).reshape(-1, 101, 3) # [episode_num, 101, 3]

    test_robot_state = test_robot_data_raw[:, :2].reshape(-1, 101, 2) # [episode_num, 101, 2]

    # test_data_set = model.data_preprocess(test_object_data_raw, test_robot_data_raw)

    # predict = model.model.predict(test_data_set['input'])
    # error = predict - target

    ### generate predict trajectories ####
    episode_num = int(len(test_robot_state))
    ### initialize ###
    predict_trajectory = np.zeros([episode_num, 101, 3])

    #### loop ####
    for episode in range(episode_num):
        trajectory = np.zeros([101, 3]) # the first 3 points are regarded as the same as targets because we need 3 steps to predict
        trajectory[:3] = test_object_state[episode][:3]

        robot_state = test_robot_state[episode]
        # print(robot_state.shape)
        for step in range(3, 101):
            trajectory[step] = get_object_state_increment_world_coordinate(model, step, trajectory, robot_state) + trajectory[step - 1]

        predict_trajectory[episode] = trajectory
        template = 'Generate {} episodes.'
        print(template.format(episode + 1))

    # print(predict_trajectory.shape)
    np.save(os.path.join(SAVED_DATA_PATH, 'LSTM_150'), predict_trajectory)



if __name__ == '__main__':
    main()
