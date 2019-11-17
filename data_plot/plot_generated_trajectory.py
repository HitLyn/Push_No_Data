import numpy as np
import matplotlib.pyplot as plt
import os

def main():

    predict_data = np.load('/home/lyn/HitLyn/Push/saved_model/epoch150/error/predict.npy').reshape(-1, 98, 2)
    target_data = np.load('/home/lyn/HitLyn/Push/saved_model/epoch150/error/target.npy').reshape(-1, 98, 2)
    error_data = np.load('/home/lyn/HitLyn/Push/saved_model/epoch150/error/error.npy').reshape(-1, 98, 3)

    # predict_data = np.load('/home/lyn/HitLyn/Push/saved_model/epoch150/error/predict.npy')
    # target_data = np.load('/home/lyn/HitLyn/Push/saved_model/epoch150/error/target.npy')
    # error_data = np.load('/home/lyn/HitLyn/Push/saved_model/epoch150/error/error.npy')
    #
    # print (predict_data.shape)
    # print (error_data.shape)

    i = 32
    generated_trajectory = predict_data[i]
    original_trajectory = target_data[i]
    error = error_data[i]
    # print(error)
    # print(np.concatenate([generated_trajectory - original_trajectory, error], axis = 1))



    fig = plt.figure()
    fig.add_subplot(2,2,1)
    plt.plot(generated_trajectory[:, 0], 'b')
    plt.plot(original_trajectory[:, 0], 'r')

    fig.add_subplot(2,2,2)
    plt.plot(error[:, 0])

    fig.add_subplot(2,2,3)
    plt.plot(generated_trajectory[:, 1], 'b')
    plt.plot(original_trajectory[:, 1], 'r')

    fig.add_subplot(2,2,4)
    plt.plot(error[:, 1])

    plt.show()






















if __name__ == '__main__':
    main()
