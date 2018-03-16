#!/usr/bin/python

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle

# label need to be 0 to num_class -1
data = pd.read_csv('../data/Training Data/data_with_nlp.csv', error_bad_lines=False)
data = data.dropna()
data = shuffle(data).reset_index(drop=True)
data['happiness_index'] = data['happiness_index'] + 2
data = data[['user_friends_count', 'user_followers_count', 'retweet_count', 'exclamation_number', 'length', 'question_number', 'uppercase_ratio', 'nlppred', 'happiness_index']].values

sz = data.shape
max_idx = sz[1] - 1

train = data[:int(sz[0] * 0.7), :]
test = data[int(sz[0] * 0.7):, :]

train_X = train[:, :max_idx - 1]
train_Y = train[:, max_idx]

test_X = test[:, :max_idx - 1]
test_Y = test[:, max_idx]

xg_data = xgb.DMatrix(data[:, :max_idx - 1], label = data[:, max_idx])
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.03
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 5
param['min_child_weight'] = 11
param['subsample'] = 0.8
param['colsample_bytree'] = 0.7

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 150
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 5)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
print('Test error using softprob = {}'.format(error_rate))

# do cross validation
res = xgb.cv(param, xg_data, num_round, nfold=5)
print(res)
