import warnings
warnings.simplefilter('ignore') # to dismiss tensorflow future warnings
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

max_len = 70
num_words = 20000

stopwords=set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clear_sentence(s):
    s = re.sub("[^a-zA-Z]"," ",s)
    s = word_tokenize(s)
    s = [lemmatizer.lemmatize(i).lower() for i in s if i not in stopwords]

    return s

def create_train_data_and_tokenizer(num_samples=None,occurence_bound=2):
    text_path = 'text_data_train.txt'
    labels_path = 'label_data_train.txt'
    text = []
    labels = []
    with open(text_path,'r',encoding="iso-8859-1") as f:
        for index,line in enumerate(f):
            text.append(line.strip())
            if num_samples:
                if index >num_samples:
                    break
    with open(labels_path,'r',encoding="iso-8859-1") as f:
        for index,line in enumerate(f):
            labels.append(line.strip())
            if num_samples:
                if index >num_samples:
                    break

    text = [clear_sentence(i) for i in text]

    # keep only words which occur more than 2 times
    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
    tokenizer.fit_on_texts(text)
    high_frequency = 0
    for i in tokenizer.word_counts:
        if tokenizer.word_counts[i]>occurence_bound:
            high_frequency+=1
    # print(high_frequency," words to keep")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=high_frequency,lower=True,oov_token=0)
    tokenizer.fit_on_texts(text)

    X = np.array(text)
    Y = np.array(labels)

    return X,Y,tokenizer


def tokenize_and_padding(text,tokenizer,max_len=max_len):
    text = tokenizer.texts_to_sequences(text)
    text = tf.keras.preprocessing.sequence.pad_sequences(text,maxlen=max_len,padding='post',value=0)
    return text


X,Y,tokenizer = create_train_data_and_tokenizer(num_samples=20000)
X = tokenize_and_padding(X,tokenizer)
