
import numpy as np
import tensorflow as tf
import re
import time
import pickle
from sklearn.model_selection import train_test_split

X = np.load('X1581151233.npy')
Y = np.load('Y1581151233.npy')
embedding_matrix = np.load('embedding_matrix1581151233.npy')
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
tb = tf.keras.callbacks.TensorBoard()
mchpoint = tf.keras.callbacks.ModelCheckpoint("model_checkpoint",verbose=1,monitor='val_acc',save_weights_only=True)
X_train,X_test,y_train,y_test = train_test_split(X,Y)
history = model.fit(X_train,y_train,32 ,epochs=10,callbacks=[tb,mchpoint],validation_data=(X_test,y_test))
model.save_weights('m_w')
with open("history") as f:
    pickle.dump(history,f)