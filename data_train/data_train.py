import tensorflow as tf
import numpy as np


def dynamic_model(object_state_dim, rnn_units, batch_size, time_steps, input_feature_size):
    model = tf.keras.Sequential([
        tf.keras.layers.LayerNormalization(epsilon = 1e-4, batch_input_shape = (batch_size, None, input_feature_size)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units, return_sequences = True, stateful = True, recurrent_initializer='glorot_uniform')),
        # tf.keras.layers.LSTM(rnn_units, return_sequences = True, stateful = True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(object_state_dim)
    ])
    return model





def main():
    model = dynamic_model(3, 128, 64, 5, 7)
    model.compile(optimizer = tf.keras.optimizers.Adam, loss = tf.losses.MeanSquaredError)
    # model.build()
    model.summary()

if __name__ == '__main__':
    main()
