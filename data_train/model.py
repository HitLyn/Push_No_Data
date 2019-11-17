import tensorflow as tf
import numpy as np
import os

DATA_PATH = '/home/lyn/HitLyn/Push/original_data'
SAVE_PATH = '/home/lyn/HitLyn/Push/saved_model'
SAVE_PATH_NORMAL = '/home/lyn/HitLyn/Push/saved_model_normal'
scale = 100 # scale to enlarge state error

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

def get_sample_input(object_state, robot_state, time_steps, episode, i):
    """
    Returns: np.array, size(time_steps, 4)
        """
    sample_inputs = np.zeros((time_steps, 4))

    for step in range(time_steps):
        idx = episode * 101 + i + step
        state = np.concatenate((object_state[idx], robot_state[idx]))
        action = robot_state[idx + 1]
        sample_inputs[step] = input_from_world_to_object(np.concatenate((state, action)))

    return sample_inputs

def get_sample_target(object_state, time_steps, episode, i):
    """object_state
    Returns: np.array, size(time_steps, 3)
        """
    sample_targets = np.zeros((time_steps, 3))
    for step in range(time_steps):
        idx = episode * 101 + i +step
        # state = (object_state[idx + 1] - object_state[idx])
        sample_targets[step] = target_from_world_to_object(object_state[idx + 1], object_state[idx])

    return sample_targets


class Model():
    def __init__(self, object_state_dim, robot_action_dim, rnn_units, batch_size, time_steps):
        """
        Args:
            object_state_dim(int):
            robot_action_dim(int):
            rnn_units(int):
            batch_size(int):
            time_steps(int): how many time steps we need to integrate as a sequence
        """
        # init #
        self.object_state_dim = object_state_dim
        self.robot_action_dim = robot_action_dim
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.input_feature_size =2*self.robot_action_dim # 4 in our env

        self.epochs = 150


        # model build #
        self.model = tf.keras.Sequential([
            # tf.keras.layers.LayerNormalization(epsilon = 1e-4),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units, return_sequences = True, stateful = True, recurrent_initializer='glorot_uniform')),
            tf.keras.layers.LSTM(rnn_units, return_sequences = True, stateful = True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(object_state_dim)
        ])

        # self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = tf.losses.MeanSquaredError())
        # self.model.build((batch_size, None, self.input_feature_size))

        # data load and preprocess #
        self.object_data = np.load(os.path.join(DATA_PATH, 'object_data_partial.npy'))
        self.robot_data = np.load(os.path.join(DATA_PATH, 'robot_data_partial.npy'))
        self.test_object_data = np.load(os.path.join(DATA_PATH, 'test_object_data.npy'))
        self.test_robot_data = np.load(os.path.join(DATA_PATH, 'test_object_data.npy'))
        self.data_set = self.data_preprocess(self.object_data, self.robot_data)
        self.test_data_set = self.data_preprocess(self.test_object_data, self.test_object_data)

        # try normal neural network
        self.data_set_normal = self.data_preprocess_normal(self.object_data, self.robot_data)
        self.model_normal = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(3)
        ])
        # optimizer and loss function
        self.loss = tf.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        # metrics
        self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        self.train_error = tf.keras.metrics.MeanAbsolutePercentageError(name = 'train_error')
        self.train_absolute_error = tf.keras.metrics.MeanAbsoluteError(name = 'absolute_error')

        self.test_loss = tf.keras.metrics.Mean(name = 'test_loss')
        self.test_error = tf.keras.metrics.MeanAbsolutePercentageError(name = 'test_error')
        self.test_absolute_error = tf.keras.metrics.MeanAbsoluteError(name = 'test_error')



    def train_step(self, input, target):   # for normal
        with tf.GradientTape() as tape:
            predictions = self.model_normal(input)
            loss = self.loss(target, predictions)
        gradients = tape.gradient(loss, self.model_normal.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_normal.trainable_variables))

        self.train_loss(loss)
        # self.train_error(target, predictions)
        self.train_absolute_error(target, predictions)

    def test_step(self, input, target):
        predictions = self.model_normal(input)
        loss = self.loss(target, predictions)
        self.test_loss(loss)
        # self.test_error(target, predictions)
        self.test_absolute_error(target, predictions)

    def LSTM_train_step(self, input, target):
        with tf.GradientTape() as tape:
            predictions = self.model(input)
            loss = self.loss(target, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        # self.train_error(target, predictions)
        self.train_absolute_error(target, predictions)

    def LSTM_test_step(self, input, target):
        predictions = self.model(input)
        loss = self.loss(target, predictions)
        self.test_loss(loss)
        # self.test_error(target, predictions)
        self.test_absolute_error(target, predictions)


    def train(self, data_set, test_data_set):
        """
        Args:
            data_set(dict): data_set['input']: np.array, data_set['target']: np.array
        """

        dataset = tf.data.Dataset.from_tensor_slices((data_set['input'], data_set['target'])).shuffle(10000000).batch(32, drop_remainder = True)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data_set['input'], test_data_set['target'])).shuffle(1000000).batch(32, drop_remainder = True)

        for epoch in range(self.epochs):
            if epoch + 1 % 50 == 0:
                dirpath = os.path.join(SAVE_PATH, 'epoch' + str(echo))
                if not os.path.exists(dirpath):
                    os.mkdir(dirpath)
                else:
                    self.model.save_weights(dirpath)


            dataset.shuffle(10000000)
            # i = 0
            # for input, target in dataset:
            #     self.LSTM_train_step(input, target)

            for input, target in dataset:
                self.LSTM_train_step(input, target)
                # if i % 100 == 0:
                #     print('training batch: ', i)
                # i += 1

            for test_input, test_target in test_dataset:
                self.LSTM_test_step(test_input, test_target)

            template = 'Epoch {}, Loss: {}, Mean Error: {}, Test Loss: {}, Test Mean Error: {}'
            print(template.format(epoch + 1,
                                self.train_loss.result(),
                                self.train_absolute_error.result(),
                                self.test_loss.result(),
                                self.test_absolute_error.result()))

            # reset the metrics for next epoch
            self.train_loss.reset_states()
            self.train_error.reset_states()
            self.train_absolute_error.reset_states()

            #save weights
            if epoch + 1 % 50 == 0:
                dirpath = os.path.join(SAVE_PATH, 'epoch' + str(echo + 1))
                if not os.path.exists(dirpath):
                    os.mkdir(dirpath)
                else:
                    self.model.save_weights(dirpath)



    def data_preprocess(self, object_data, robot_data):

        object_position = object_data[:, :2]
        object_rotation = (np.arccos(object_data[:, 3]) * 2).reshape(-1, 1)
        object_state = np.concatenate((object_position, object_rotation), axis = 1)

        assert object_state.shape == (len(object_position), 3)

        robot_state = robot_data[:, :2]
        assert robot_state.shape == (len(robot_state), 2)

        episode_num = len(object_state)//101 # how many episodes in the dataset: 101 timestep each
        sequences_num_per_episode = 101 - self.time_steps # how many sequences to generate per episode

        sample_inputs = np.zeros((episode_num * sequences_num_per_episode, self.time_steps, 4))
        sample_targets = np.zeros((episode_num * sequences_num_per_episode, self.time_steps, 3))

        # get inputs and targets
        for episode in range(episode_num):
            for i in range(sequences_num_per_episode):
                idx = episode * sequences_num_per_episode + i
                sample_inputs[idx] = get_sample_input(object_state, robot_state, self.time_steps, episode, i)
                sample_targets[idx] = get_sample_target(object_state, self.time_steps, episode, i)
        data_set = {}
        data_set['input'] = sample_inputs
        data_set['target'] = sample_targets
        data_set['shuffled_input'] = data_set['input']
        data_set['shuffled_target'] = data_set['target']

        # shuffle
        idx = np.arange(len(sample_inputs))
        np.random.shuffle(idx)
        for i in range(len(idx)):
            data_set['shuffled_input'][i] = data_set['input'][idx[i]]
            data_set['shuffled_target'][i] = data_set['target'][idx[i]]

        return data_set

    def data_preprocess_normal(self, object_data, robot_data):

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
                sample_inputs[idx] = get_sample_input(object_state, robot_state, 1, episode, i)
                sample_targets[idx] = get_sample_target(object_state, 1, episode, i)
        data_set = {}
        data_set['input'] = np.squeeze(sample_inputs)
        data_set['target'] = np.squeeze(sample_targets)

        data_set['shuffled_input'] = data_set['input']
        data_set['shuffled_target'] = data_set['target']

        # shuffle
        idx = np.arange(len(sample_inputs))
        np.random.shuffle(idx)
        for i in range(len(idx)):
            data_set['shuffled_input'][i] = data_set['input'][idx[i]]
            data_set['shuffled_target'][i] = data_set['target'][idx[i]]

        return data_set

    def predict(self, state):
        """
        Because of the way the RNN state is passed from timestep to timestep,
        the model only accepts a fixed batch size once built.
        To run the model with a different batch_size, we need to rebuild the model
        Args:
            state: np.array, size = (1, self.time_steps, 7)
        """
        self.model.build(tf.TensorShape([1, None, self.input_feature_size]))
        output = self.model.predict(state)
        return output


    def evaluate(self, inputs, targets):
        pass

def main():
    model = Model(3, 2, 128, 64, 3)
    data_set = model.data_set
    test_data_set = model.test_data_set
    # print(data_set['input'][50], data_set['target'][50])
    # print(model.model_normal.summary())
    model.train(data_set, test_data_set)

if __name__ == '__main__':
    main()
