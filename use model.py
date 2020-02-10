import tensorflow as tf
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

TOKENIZER = np.load(r'requirements to use model\tokenizer_pickle',allow_pickle=True)

stopwords=set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clear_sentence(s):
    s = re.sub("[^a-zA-Z]"," ",s)
    s = word_tokenize(s)
    s = [lemmatizer.lemmatize(i).lower() for i in s if i not in stopwords]
    return s

def tokenize_and_padding(text,tokenizer,max_len=50):
    text = tokenizer.texts_to_sequences(text)
    text = tf.keras.preprocessing.sequence.pad_sequences(text,maxlen=max_len,padding='post',value=0)
    return text

def create_model():

    embedding_matrix = np.load(r'requirements to use model\embedding_matrix1581151233.npy')
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
model.load_weights(r"requirements to use model\model_checkpoint")

while True:
    s = input()

    x= [clear_sentence(s)]
    x = tokenize_and_padding(x,TOKENIZER)

    print(model.predict(x))