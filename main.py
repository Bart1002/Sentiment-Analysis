import warnings
warnings.simplefilter('ignore') # to dismiss tensorflow future warnings
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import time
import pickle
from sklearn.model_selection import train_test_split

max_len = 50
num_words = 50000

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
            labels.append(int(line.strip()))
            if num_samples:
                if index >num_samples:
                    break

    text = [clear_sentence(i) for i in text]

    # keep only words which occur more than 2 times
    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
    tokenizer.fit_on_texts(text)
    of_higher_frequency = 0
    for i in tokenizer.word_counts:
        if tokenizer.word_counts[i]>occurence_bound:
            of_higher_frequency+=1
    # print(high_frequency," words to keep")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=of_higher_frequency,lower=True,oov_token=0)
    tokenizer.fit_on_texts(text)

    X = np.array(text)
    Y = np.array(labels)

    return X,Y,tokenizer, of_higher_frequency


def tokenize_and_padding(text,tokenizer,max_len=max_len):
    text = tokenizer.texts_to_sequences(text)
    text = tf.keras.preprocessing.sequence.pad_sequences(text,maxlen=max_len,padding='post',value=0)
    return text


X,Y,tokenizer,of_higher_frequency = create_train_data_and_tokenizer(num_samples=20000)
X = tokenize_and_padding(X,tokenizer)


def get_word_vec(word, *arr):
    return word, np.asarray(arr,dtype="float32")

def get_words_embeddings(tokenizer,of_higher_frequency):
    glove_path = r'C:\Users\barte\Downloads\glove.6B\glove.6B.50d.txt'
    
    embedding_dict = dict(get_word_vec(*o.strip().split()) for o in open(glove_path,'r',encoding='utf-8'))

    all_vecs = np.stack((embedding_dict.values()))
    all_mean,all_std = np.mean(all_vecs), np.std(all_vecs)

    del all_vecs

    word_index = tokenizer.word_index
    matrix_size = min(of_higher_frequency,len(word_index))


    matrix = np.random.normal(all_mean,all_std,(len(word_index)+1,50))

    for word,i in word_index.items():
        if i>matrix_size:
            continue
        vec = embedding_dict.get(word)

        if vec is not None:
            matrix[i] = vec

    return matrix

print(of_higher_frequency,"------------------------------------------------------------------")
embedding_matrix = get_words_embeddings(tokenizer,of_higher_frequency)
t = int(time.time())
np.save(f"requirements to use model/embedding_matrix{t}",embedding_matrix)
np.save(f"requirements to use model/tokenizer{t}",tokenizer)
np.save(f"requirements to use model/X{t}",X)
np.save(f"requirements to use model/Y{t}",Y)
def create_model():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(27849,50,weights=[embedding_matrix],input_shape=(max_len,)))
    model.add(tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(256)))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    return model

# for i in range(10):
#     print(X[i])
# print("--------------------------")
model = create_model()
tb = tf.keras.callbacks.TensorBoard()

X_train,X_test,y_train,y_test = train_test_split(X,Y)
history = model.fit(X_train,y_train,32 ,epochs=10,callbacks=[tb],validation_data=(X_test,y_test))

with open("history") as f:
    pickle.dump(history,f)


