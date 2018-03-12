import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from svmutil import *


def get_weak_accuracy(p_label, test_label):
    count = 0.0
    sum = len(p_label)
    for i in range(sum):
        if p_label[i] == test_label[i] or p_label[i] * test_label[i] > 0:
            count += 1.0
    return count/sum

if __name__ == '__main__':

    # Read Corpus
    df = pd.read_csv('../data/Training Data/data_with_nlp.csv', error_bad_lines=False)
    df = df.dropna()
    df = shuffle(df, random_state=97).reset_index(drop=True)
    df['retweet_count'] = df['retweet_count'].astype('float64')

    # Select Features
    data_X = df[['user_friends_count', 'user_followers_count', 'retweet_count', 'exclamation_number', 'length',
                 'question_number', 'uppercase_ratio', 'nlppred']].values
    data_Y = df['happiness_index'].values

    # Standardization
    scaler = StandardScaler()
    scaler.fit(data_X)
    data_X = scaler.transform(data_X)

    # 10-Fold cross validation to compute the accuracy of the model
    hold = 10
    kf = KFold(n_splits=hold)
    accuracy = []
    weak_accuracy = []
    for train_keys, test_keys in kf.split(range(len(data_X))):
        train_feature = data_X[train_keys]
        train_label = data_Y[train_keys]
        test_feature = data_X[test_keys]
        test_label = data_Y[test_keys]

        # Random oversampling
        ros = RandomOverSampler()
        X_resampled, Y_resampled = ros.fit_sample(train_feature, train_label)

        model = svm_train(Y_resampled.tolist(), X_resampled.tolist(), '-c 4 -q')  # train the svm
        p_label, p_acc, p_val = svm_predict(test_label.tolist(), test_feature.tolist(), model)  # predict on test set
        accuracy.append(p_acc[0])
        weak_accuracy.append(get_weak_accuracy(p_label, test_label))
        print('New Accuracy: ', weak_accuracy[-1])

    print('10-CV Accuracy = {}'.format(np.mean(accuracy)))
    print('10-CV Weak Accuracy = {}'.format(np.mean(weak_accuracy)))

    # train and save the final svm model
    ros = RandomOverSampler()
    X_resampled, Y_resampled = ros.fit_sample(data_X, data_Y)
    final_model = svm_train(Y_resampled.tolist(), X_resampled.tolist(), '-c 4 -q')  # train the final svm
    svm_save_model('happiness_index_svm.model', final_model)

