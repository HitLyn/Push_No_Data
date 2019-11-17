import tensorflow as tf
import numpy as np

def main():

    a = np.random.randint(5, size = (98, 3, 7))
    b = np.random.randint(5, size = (98, 3, 3))

    input_set = tf.data.Dataset.from_tensor_slices(a)
    target_set = tf.data.Dataset.from_tensor_slices(b)
    dataset = tf.data.Dataset.zip((input_set,target_set))
    batched_dataset = dataset.batch(4, drop_remainder = True)
    batched_dataset.map(lambda x,y: x,y)

if __name__ =='__main__':
    main()
