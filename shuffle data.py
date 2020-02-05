import pandas as pd
import numpy as np

data = pd.read_csv('training data ALL',encoding="iso-8859-1")

print(data.head())

data = data.sample(frac=1)

print(data.head(30))

sentiment = data['sentiment'].values
text = data['text'].values

with open('text_data_train.txt','w',encoding="iso-8859-1") as f:
    for i in text:
        f.write(str(i)+'\n')

with open('label_data_train.txt','w',encoding="iso-8859-1") as f:
    for i in sentiment:
        f.write(str(i)+'\n')

