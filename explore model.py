
import numpy as np
import tensorflow as tf
import re
import time
import pickle
from sklearn.model_selection import train_test_split

X = np.load('requirements to use model/X1581151233.npy')
Y = np.load('requirements to use model/Y1581151233.npy')
embedding_matrix = np.load('requirements to use model/embedding_matrix1581151233.npy')
def create_model():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(27849,50,weights=[embedding_matrix],input_shape=(50,)))
    model.add(tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(256)))
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    return model



model = create_model()
model.load_weights('requirements to use model\model_checkpoint')
score = model.evaluate(X,Y,batch_size=32)
print(score)