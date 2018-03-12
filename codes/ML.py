from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Suppress warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

# Load the data from csv
df = pd.read_csv('../data/Training Data/data_with_nlp.csv', error_bad_lines=False)
df = df.dropna()
df = shuffle(df).reset_index(drop=True)

# print(df.head())
# >>> created_at, favorite_count, geo, happiness_index, hashtags,
# 	  place, retweet_count, tweets, user_followers_count, user_friends_count,
# 	  user_id, user_location, user_name, user_tweet_count, exclamation_number,
# 	  length, question_number, uppercase_ratio, nlppred

# Select features
# df = df.replace('?', np.nan)
# data_X = df.drop('happiness_index', axis=1).values
data_X = df[['user_friends_count', 'user_followers_count', 'user_tweet_count', 'retweet_count', 'favorite_count', 'exclamation_number', 'length', 'question_number', 'uppercase_ratio', 'nlppred']].values
data_X = np.delete(data_X, np.s_[0], axis=1)
data_y = df['happiness_index'].values

# Over sampling
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(data_X, data_y)

# # Check over-sampling result
# from collections import Counter
# print(sorted(Counter(data_y).items()))
# print(sorted(Counter(y_resampled).items()))

# Standardizatoin
scaler = StandardScaler()
scaler.fit(X_resampled)
X_resampled = scaler.transform(X_resampled)
# np.savetxt('check.out', X_resampled, delimiter=',')

# Create linear regression object
regr = linear_model.LinearRegression()
# X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
# regr = linear_model.LinearRegression(normalize=True)

# Create logistic regression object
classifier = linear_model.LogisticRegression()

# Cross Validation
k_fold = KFold(n_splits=10)
mses = []
accuracies = []
for k, (train, test) in enumerate(k_fold.split(X_resampled, y_resampled)):
    
    # Train the model using the training sets
    regr.fit(X_resampled[train], y_resampled[train])
    classifier.fit(X_resampled[train], y_resampled[train])

    # Make predictions using the testing set
    data_y_pred = regr.predict(X_resampled[test])

    # The coefficients
    coef = regr.coef_

    # The mean squared error
    mse = mean_squared_error(y_resampled[test], data_y_pred)
    mses.append(mse)

    # The mean accuracy
    accuracy = classifier.score(X_resampled[test], y_resampled[test])
    accuracies.append(accuracy)

    # Explained variance score: 1 is perfect prediction
    r2 = r2_score(y_resampled[test], data_y_pred)

    # print("\n[fold {0}] \nCoefficients: {1} \nMean squared error: {2:.2f} \nVariance score: {3:.2f} \nAccuracy: {4:.2f}".
    #       format(k, coef, mse, r2, accuracy))

print('\n10-CV MSE = {0:.4f} Accuracy = {1:.4f}\n'.format(np.mean(mses), np.mean(accuracies)))

# # Use the whole dataset to train
# data_X_train = X_resampled
# data_y_train = y_resampled

# # Load test data from csv, do standardization
# df = pd.read_csv('test.csv', error_bad_lines=False)
# df = df.dropna()
# data_X_test = df[['user_friends_count', 'user_followers_count', 'retweet_count', 'exclamation_number', 'length', 'question_number', 'uppercase_ratio', 'nlppred']].values
# scaler = StandardScaler()
# scaler.fit(data_X_test)
# data_X_test = scaler.transform(data_X_test)

# # Train the model using the training sets
# regr.fit(data_X_train, data_y_train)
# classifier.fit(data_X_train, data_y_train)

# # Make predictions using the testing set
# data_y_pred = regr.predict(data_X_test)

# # The coefficients
# print('Coefficients(Linear Regression): ')
# print(regr.coef_)
# print('Coefficients(Logistic Regression): ')
# print(classifier.coef_)

# # Output predictions to csv file
# df['predictions'] = data_y_pred
# df.to_csv('predictions.csv')
