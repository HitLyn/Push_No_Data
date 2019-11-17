import numpy as np
import os
import argparse

from model import Model



def robot_data_process(test_robot_data):

    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type = str, default = '/home/lyn/HitLyn/Push/original_data')
    parser.add_argument('--weights_load_path', type = str, default = '/home/lyn/HitLyn/Push/saved_model')
    parser.add_argument('--generate_data_path', type = str, default = '/home/lyn/HitLyn/Push/data_generation')
    args = parser.parse_args()

    model = Model()
    weights_path = args.weights_load_path
    model.model.load_weights(weights_path)

    test_robot_data = np.load(os.path.join(args.test_data_path, 'robot_robot_data.npy'))
    test_input = robot_data_process(test_robot_data)











if __name__ == '__main__':
    main()
