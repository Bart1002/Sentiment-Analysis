import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# data = pd.read_csv('training.1600000.processed.noemoticon.csv',index_col=['id'],encoding="iso-8859-1")
# data = data.reindex(np.random.permutation(data.to_del))

# data = pd.read_csv('training.1600000.processed.noemoticon.csv',usecols=['sentiment','text'],encoding="iso-8859-1")

# stopwords=set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def clear_text(text):
#     text = re.sub('[^a-zA-Z]',' ',text)
#     text = word_tokenize(text)
#     text = [lemmatizer.lemmatize(i) for i in text]
#     return text

# def sentiment(s):
#     if s==0:
#         return 0
#     else:
#         return 1

# data['sentiment'] =  data['sentiment'].apply(sentiment,convert_dtype=True)
# data['text']=data['text'].apply(clear_text,convert_dtype=True)
# data.to_csv('training.1600000.processed.noemoticon.csv')

# def parse_text(text):
#     text = re.sub("[^a-zA-Z]"," ",text).split()
#     print(text)
#     return text

# data = pd.read_csv('training.1600000.processed.noemoticon.csv',usecols=['sentiment','text'],encoding="iso-8859-1",nrows=10000)


# print(data.head())

# data['text'] = data['text'].apply(parse_text,convert_dtype=True)
