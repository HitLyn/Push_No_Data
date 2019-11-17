import numpy as np
import os
import argparse

from model_2 import Model



def robot_data_process(test_robot_data):

    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type = str, default = '/home/lyn/HitLyn/Push/original_data')
    parser.add_argument('--weights_load_path', type = str, default = '/home/lyn/HitLyn/Push/saved_model/epoch150/log')
    parser.add_argument('--generate_data_path', type = str, default = '/home/lyn/HitLyn/Push/data_generation')
    parser.add_argument('--batch_size', type = int, default = 128)
    args = parser.parse_args()

    model = Model(3, 2, 16, 256, 3)
    weights_path = args.weights_load_path
    model.model.load_weights(weights_path)

    model.model.summary()

    test_robot_data = np.load(os.path.join(args.test_data_path, 'test_robot_data.npy'))
    test_object_data = np.load(os.path.join(args.test_data_path, 'test_object_data.npy'))
    test_data_set = model.data_preprocess(test_object_data, test_robot_data)

    predict = model.model.predict(test_data_set['input'])
    target = test_data_set['target']
    error = predict - target

    np.save('/home/lyn/HitLyn/Push/saved_model/epoch150/error/error', error)

    # print(np.mean(np.absolute(error), axis = 0), np.amax(np.absolute(error), axis = 0))
    # print(predict.shape)
    predict_position = predict[:, :2]
    target_position = target[:, :2]

    np.save('/home/lyn/HitLyn/Push/saved_model/epoch150/error/predict', predict_position)
    np.save('/home/lyn/HitLyn/Push/saved_model/epoch150/error/target', target_position)











if __name__ == '__main__':
    main()
