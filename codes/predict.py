import glob
import pandas as pd
import pickle
import xgboost as xgb

with open('./Model/XGB.model', "r") as fp:
    clf = pickle.load(fp)

for filename in glob.iglob('../data/Test Data/*.csv'):
    data = pd.read_csv(filename, error_bad_lines=False)
    data = data[['user_friends_count', 'user_followers_count', 'retweet_count', 'exclamation_number', 'length', 'question_number', 'uppercase_ratio', 'nlppred', 'created_at']]
    data = data.dropna()
    data['retweet_count'] = pd.to_numeric(data['retweet_count'])
    result = data[['created_at']]
    result.rename(columns = {'created_at':'dateTime'}, inplace=True)

    predictions = clf.predict(data.drop('created_at', axis=1)) - 2

    result['prediction'] = predictions
    result['prediction'] = result['prediction'].astype(int)
    filename = filename.replace("../data/Test Data/", "../prediction/").replace("_nlp", "_result")
    result.to_csv(filename, index=False)
