import pandas as pd
import numpy as np
import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer



data = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding="iso-8859-1",header=None)
print(data.head())
data.drop([1,2,3,4],axis=1,inplace=True)

def change_sentiment(s):
    if s==0:
        return 0
    else:
        return 1
data[0] = data[0].apply(change_sentiment)
data.to_csv('training data ALL',header=['sentiment','text'])

# stopwords=set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

