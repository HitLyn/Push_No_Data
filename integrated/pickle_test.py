import pickle
import os
import sys
import tensorflow as tf

BASE_DIR=(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from data_train.model_5 import Model
from model_test import Model as M




def main():
    # model = Model(3, 2, 100, 64, 64, 10, load_data = False)
    # tf.saved_model.save(model.model, "/home/lyn/HitLyn/Push/saved_model/model/model_5")
    # model.model.save('/home/lyn/HitLyn/Push/saved_model/model/model.h5')
    model = tf.keras.models.load_model('/home/lyn/HitLyn/Push/saved_model/model/model.h5')
    model.summary()



if __name__ == '__main__':
    main()
