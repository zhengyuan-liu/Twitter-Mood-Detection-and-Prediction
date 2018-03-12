import pandas as pd
import pickle
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

data = pd.read_csv('../data/Training Data/data_with_nlp.csv', error_bad_lines=False)
data = data.dropna()
data['happiness_index'] = data['happiness_index'] + 2
data = data[['user_friends_count', 'user_followers_count', 'retweet_count', 'exclamation_number', 'length', 'question_number', 'uppercase_ratio', 'nlppred', 'happiness_index']]
data['retweet_count'] = pd.to_numeric(data['retweet_count'])

xgb_model = xgb.XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
param_grid = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['multi:softprob'],
              'max_depth': [6],
              'silent': [1],
              'seed': [1337],
              'learning_rate': [x * 0.01 for x in range(1,20)], #so called `eta` value
              'min_child_weight': range(1,13,2),
              'subsample': [x * 0.1 for x in range(5,10)],
              'colsample_bytree': [x * 0.1 for x in range(5,10)],
              'n_estimators': range(50,400,50)} #so called `num_round` value, number of trees

clf = GridSearchCV(xgb_model, param_grid, n_jobs=5, 
                   cv=StratifiedKFold(data['happiness_index'], n_folds=5, shuffle=True), 
                   scoring='accuracy',
                   verbose=2, refit=True)

clf.fit(data.drop('happiness_index', axis=1), data['happiness_index'])

#trust your CV!
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Accuracy:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

print(clf.best_estimator_)
with open('./Model/XGB_new.model', "w") as fp:
    pickle.dump(clf.best_estimator_, fp)

