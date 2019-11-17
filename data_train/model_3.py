import tensorflow as tf
import numpy as np
import os

DATA_PATH = '/home/lyn/HitLyn/Push/original_data'
SAVE_PATH = '/home/lyn/HitLyn/Push/saved_model'


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

def get_sample_input(env_time_step, object_state, robot_state, episode, i):
    """
    Returns: np.array, size(time_steps, 4)
        """

    idx = episode * env_time_step + i
    state = np.concatenate((object_state[idx], robot_state[idx]))
    action = robot_state[idx + 1]
    sample_inputs = input_from_world_to_object(np.concatenate((state, action)))

    return sample_inputs

def get_sample_target(env_time_step, object_state, episode, i):
    """object_state
    Returns: np.array, size(time_steps, 3)
        """

    idx = episode * env_time_step + i
        # state = (object_state[idx + 1] - object_state[idx])
    sample_targets = target_from_world_to_object(object_state[idx + 1], object_state[idx])

    return sample_targets


class Model():
    def __init__(self, object_state_dim, robot_action_dim, env_time_step, batch_size):
        """
        Args:
            object_state_dim(int):
            robot_action_dim(int):
            env_time_step(int): the time steps we cut from the env(50)
            rnn_units(int):
            batch_size(int):
        """
        # init #
        self.object_state_dim = object_state_dim
        self.robot_action_dim = robot_action_dim
        self.env_time_step = env_time_step
        self.batch_size = batch_size
        self.input_feature_size =2*self.robot_action_dim # 4 in our env

        self.epochs = 150


        # model build #
        self.model = tf.keras.Sequential([
            # tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3)
        ])

        # self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.losses.MeanAbsoluteError())
        self.model.build((self.batch_size, self.input_feature_size))

        # data load and preprocess #
        self.object_data = np.load(os.path.join(DATA_PATH, 'object_data_60.npy')).reshape(-1, 13)
        self.robot_data = np.load(os.path.join(DATA_PATH, 'robot_data_60.npy')).reshape(-1, 6)
        self.test_object_data = np.load(os.path.join(DATA_PATH, 'test_object_data_60.npy')).reshape(-1, 13)
        self.test_robot_data = np.load(os.path.join(DATA_PATH, 'test_robot_data_60.npy')).reshape(-1, 6)
        # self.evaluate_object_data = np.load(os.path.join(DATA_PATH, 'object_evaluate_data_60.npy')).reshape(-1, 13)
        # self.evaluate_robot_data = np.load(os.path.join(DATA_PATH, 'robot_evaluate_data_60.npy')).reshape(-1, 6)
        self.data_set = self.data_preprocess(self.object_data, self.robot_data)
        self.test_data_set = self.data_preprocess(self.test_object_data, self.test_object_data)
        # self.evaluate_data_set = self.data_preprocess(self.evaluate_object_data, self.evaluate_robot_data)


        # optimizer and loss function
        self.loss = tf.losses.MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam()
        # metrics
        self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        # self.train_error = tf.keras.metrics.MeanAbsolutePercentageError(name = 'train_error')
        self.train_absolute_error = tf.keras.metrics.MeanAbsoluteError(name = 'absolute_error')

        self.test_loss = tf.keras.metrics.Mean(name = 'test_loss')
        # self.test_error = tf.keras.metrics.MeanAbsolutePercentageError(name = 'test_error')
        self.test_absolute_error = tf.keras.metrics.MeanAbsoluteError(name = 'test_error')

        # self.evaluate_loss = tf.keras.metrics.Mean(name = 'evaluate_loss')
        # # self.evaluate_error = tf.keras.metrics.MeanAbsolutePercentageError(name = 'evaluate_error')
        # self.evaluate_absolute_error = tf.keras.metrics.MeanAbsoluteError(name = 'evaluate_error')


    def train_step(self, input, target):   # for normal
        with tf.GradientTape() as tape:
            predictions = self.model(input)
            loss = self.loss(target, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        # self.train_error(target, predictions)
        # self.train_absolute_error(target, predictions)

    def test_step(self, input, target):
        predictions = self.model(input)
        loss = self.loss(target, predictions)
        self.test_loss(loss)
        # self.test_error(target, predictions)
        # self.test_absolute_error(target, predictions)

    def evaluate_step(self, input, target):
        predictions = self.model(input)
        loss = self.loss(target, predictions)
        self.evaluate_loss(loss)
        # self.test_error(target, predictions)
        # self.evaluate_absolute_error(target, predictions)


    def train(self, data_set, test_data_set):
        """
        Args:
            data_set(dict): data_set['input']: np.array, data_set['target']: np.array
        """

        dataset = tf.data.Dataset.from_tensor_slices((data_set['input'], data_set['target']))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data_set['input'], test_data_set['target'])).shuffle(1000000).batch(self.batch_size, drop_remainder = True)
        # evaluate_dataset = tf.data.Dataset.from_tensor_slices((evaluate_data_set['input'], evaluate_data_set['target'])).shuffle(1000000).batch(self.batch_size, drop_remainder = True)

        for epoch in range(self.epochs):
            dataset_epoch = dataset.shuffle(10000000).batch(self.batch_size, drop_remainder = True)

            for input, target in dataset_epoch:
                self.train_step(input, target)

            for test_input, test_target in test_dataset:
                self.test_step(test_input, test_target)
            #
            # for evaluate_input, evaluate_target in evaluate_dataset:
            #     self.evaluate_step(evaluate_input, evaluate_target)

            template = 'Epoch {}, Loss: {}, Test Loss: {}'
            print(template.format(epoch + 1,
                                self.train_loss.result(),
                                # self.train_absolute_error.result(),
                                # self.evaluate_loss.result(),
                                # self.evaluate_absolute_error.result(),
                                # self.test_absolute_error.result(),
                                self.test_loss.result(),
                                ))

            # reset the metrics for next epoch
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            # self.evaluate_loss.reset_states()


            #save weights
            if (epoch + 1) % 10 == 0:
                dirpath = os.path.join(SAVE_PATH, 'epoch' + str(epoch + 1), 'log')
                namepath = os.path.join(SAVE_PATH, 'epoch' + str(epoch + 1), '60_steps')
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                    print('saving weights to', namepath)
                    self.model.save_weights(namepath)
                else:
                    self.model.save_weights(namepath)



    def data_preprocess(self, object_data, robot_data):

        object_position = object_data[:, :2]
        object_rotation = (np.arccos(object_data[:, 3]) * 2).reshape(-1, 1)
        object_state = np.concatenate((object_position, object_rotation), axis = 1)

        assert object_state.shape == (len(object_position), 3)

        robot_state = robot_data[:, :2]
        assert robot_state.shape == (len(robot_state), 2)

        episode_num = len(object_state)//self.env_time_step # how many episodes in the dataset: 101 timestep each
        sequences_num_per_episode = self.env_time_step - 1 # how many sequences to generate per episode

        sample_inputs = np.zeros((episode_num * sequences_num_per_episode, 4))
        sample_targets = np.zeros((episode_num * sequences_num_per_episode, 3))

        # get state
        for episode in range(episode_num):
            for i in range(self.env_time_step - 1):
                idx = episode*(self.env_time_step - 1) + i
                sample_inputs[idx] = get_sample_input(self.env_time_step, object_state, robot_state, episode, i)
                sample_targets[idx] = get_sample_target(self.env_time_step, object_state, episode, i)


        data_set = {}
        data_set['input'] = sample_inputs
        data_set['target'] = sample_targets


        return data_set


    def predict(self, state):
        """
        Because of the way the RNN state is passed from timestep to timestep,
        the model only accepts a fixed batch size once built.
        To run the model with a different batch_size, we need to rebuild the model
        Args:
            state: np.array, size = (1, self.time_steps, 4)
        """
        self.model.build(tf.TensorShape([1, None, self.input_feature_size]))
        output = self.model.predict(state)
        return output


    def evaluate(self, inputs, targets):
        pass

def main():
    model = Model(3, 2, 60, 128)
    data_set = model.data_set
    test_data_set = model.test_data_set
    # evaluate_data_set = model.evaluate_data_set
    # print(data_set['input'][50], data_set['target'][50])
    # print(model.model_normal.summary())
    model.train(data_set, test_data_set)

if __name__ == '__main__':
    main()
