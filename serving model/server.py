import numpy as np
import flask
from flask import request, render_template
import io
import tensorflow as tf
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import re


app = flask.Flask(__name__)
model = None

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

def load_model():

    embedding_matrix = np.load(r'requirements to use model\embedding_matrix1581151233.npy')
    _model = tf.keras.models.Sequential()
    _model.add(tf.keras.layers.Embedding(27849,50,weights=[embedding_matrix],input_shape=(50,)))
    _model.add(tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(256)))
    _model.add(tf.keras.layers.Dense(32,activation='relu'))
    _model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

    _model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    _model.load_weights(r"requirements to use model\model_checkpoint")

    global model
    model = _model


@app.route("/predict", methods=["POST"])
def predict():

    text = request.data

    original_text = text

    text = text.decode("utf-8") 

    text= [clear_sentence(text)]
    text = tokenize_and_padding(text,TOKENIZER)

    result = model.predict(text)
    r = str(result[0][0])

    return render_template('index.html',r=r,original_text=original_text)


@app.route("/")
def home():
    return render_template('index.html')

if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1",port=5000,debug=True,use_reloader=False)
