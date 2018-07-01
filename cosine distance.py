import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PATH_TO_DATA = ('D:/Py/DataFrames/Avito_Demand_Prediction_Challenge(KAGGLE)/')

used_cols = ['item_id', 'title', 'description']
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train.csv'), usecols=used_cols)
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test.csv'), usecols=used_cols)

# preprocessing
text_cols = used_cols
text_cols.remove('item_id')
for col in text_cols:
    for df in [train_df, test_df]:
        df[col] = df[col].str.replace('/\n', ' ').replace('\xa0', ' ').replace('.', ' . ').replace(',', ' , ')
        df[col].fillna("NA", inplace=True)
        df[col] = df[col].str.lower()


# CountVectorizer for ALL
cv = CountVectorizer(ngram_range=(1, 1))
full_cv = cv.fit_transform(
    train_df['title'].values.tolist() + test_df['title'].values.tolist() + \
    train_df['description'].values.tolist() +test_df['description'].values.tolist())

train_title = cv.transform(train_df['title'].values.tolist())
train_description = cv.transform(train_df['description'].values.tolist())

test_title = cv.transform(test_df['title'].values.tolist())
test_description = cv.transform(test_df['description'].values.tolist())

# [TRAIN COSINE DISTANCE]
dists = []
for i in range(train_df.shape[0]):
    dist = cosine_similarity(train_description[i], train_title[i])
    dists.append(dist)

train_dists = np.append(dists)
train_df['cosine_distance'] = train_dists

# [TEST COSINE DISTANCE]
dists = []
for i in range(train_df.shape[0]):
    dist = cosine_similarity(train_description[i], train_title[i])
    dists.append(dist)

test_dists = np.append(dists)
test_df['cosine_distance'] = test_dists