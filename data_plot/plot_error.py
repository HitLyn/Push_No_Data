import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import seaborn as sns
import pandas as pd


def main():
    error = np.load('/home/lyn/HitLyn/Push/saved_model/epoch150/error/error.npy')
    error_x = error[:, 0]
    error_y = error[:, 1]
    error_theta = error[:, 2]
    N = np.arange(len(error))

    # fig = plt.figure()
    # # fig.add_subplot(1,3,1)
    # # plt.scatter(N, error_x, s = 0.05)
    # #
    # # fig.add_subplot(1,3,2)
    # # plt.scatter(N, error_y, s = 0.05)
    # fid.add_subplot(1,2,1)
    # plt.scatter(error_x, error_y)
    #
    # fig.add_subplot(1,2,3)
    # plt.scatter(N, error_theta, s = 0.05)
    #
    # plt.show()
    df = pd.DataFrame(np.concatenate((error_x[:, np.newaxis], error_y[:, np.newaxis]), axis = 1), columns = ["error_x", "error_y"])
    sns.jointplot(x = "error_x", y = "error_y", xlim = (-0.015, 0.015), ylim = (-0.015, 0.015), data = df, kind = "kde")
    # fig = plt.figure()
    # ax = fig.add_subplot(1,2,1)
    # ax.set_xlim((-0.015, 0.015))
    # ax.set_ylim((-0.015, 0.015))
    # plt.scatter(error_x, error_y)
    plt.show()

if __name__ == '__main__':
    main()
