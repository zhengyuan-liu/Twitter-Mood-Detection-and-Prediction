import csv
from itertools import islice
import numpy as np
import random
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


# Apply Z-score Standardization to the data and return the standardized data
def z_score(data):
    dim = data.shape[1]  # dimension of features
    for i in range(dim-1):  # for every feature of the data
        col = data[:, i]
        col_mean = np.mean(col)  # mean of the feature
        col_std = np.std(col)  # standard deviation of the feature
        if col_std!=0:
            data[:, i] = (data[:, i] - col_mean) / col_std  # Apply Z-score Standardization
        else:
            data[:, i] = 0
    return data

# Oversample training data to sovle the imbalanced data problem
def oversample(data, count):
    max_count = max(count)
    oversampled_data = data
    for i in range(len(count)):
        for j in range(max_count - count[i]):
            rand = random.randint(0, count[i] - 1)     
            oversampled_data[i].append(data[i][rand])
    return oversampled_data

#train data and get weights of Neural Network
def get_parameters(X,y,syn0,syn1):
    
    fun=0.1
    for j in range(100000):
        l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
        l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
        l2_delta = (y - l2)*(l2*(1-l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        syn1 += fun*l1.T.dot(l2_delta)
        syn0 += fun*X.T.dot(l1_delta)   
    return syn0,syn1

if __name__ == '__main__':
    df = pd.read_csv('../data/Training Data/data_with_nlp.csv', error_bad_lines=False)
    df = df.dropna()
    df = shuffle(df).reset_index(drop=True)
    feature_vectors = df[['user_friends_count', 'user_followers_count', 'retweet_count',  'exclamation_number', 'length', 'question_number', 'uppercase_ratio', 'nlppred','happiness_index']].values
    labels = df[['happiness_index']].values
    feature_vectors=feature_vectors.astype(np.float)
    feature_vectors=feature_vectors[0:2700:15]
    labels=labels[0:2700:15]
    '''print(feature_vectors)
    print("--------------")
    print(labels)'''
    hidden=20
    syn0 = 2*np.random.random((8,hidden)) - 1
    syn1 = 2*np.random.random((hidden,1)) - 1
    # 10-Fold cross validation
    hold = 10
    kf = KFold(n_splits=hold)
    accuracy=0
    accuracy_2=0
    z_score(feature_vectors)
    
    
    for train_keys, test_keys in kf.split(range(len(labels))):
        train_feature = feature_vectors[train_keys]
        train_label = labels[train_keys]
        test_feature = feature_vectors[test_keys]
        test_label = labels[test_keys]
        #oversample data
        class_num = 5
        count = [0] * class_num
        row_data=[[]for i in range(class_num)]
        train_feature=np.ndarray.tolist(train_feature)
        for i in range(len(train_feature)):
            count[int(train_feature[i][8]+2)]+=1
            row_data[int(train_feature[i][8]+2)].append(train_feature[i])
        
        print ('**************************  Begin 10-CV **************************')
        print ('the number of data for each label before oversampling:')
        print(count)
        oversampled_data=oversample(row_data,count)
        print ('the number of data for each label after oversampling:')
        print (len(oversampled_data[0]))
        print('-----------------------------------')
        
        train_data = []
        for i in range(class_num):
            train_data +=oversampled_data[i]
        random.shuffle(train_data)
        train_data = np.array(train_data)
        final_data = np.array(train_data[:, 0:-1])
        test_feature=np.array(test_feature[:, 0:-1])
        final_labels = [np.array(train_data[:, -1])]
        final_labels=np.array(final_labels).T
        test_label=np.array(test_label)
        #the range of function sigmoid is(0,1),so we have to deal the labels
        for i in range(len(final_labels)):    
            final_labels[i]=final_labels[i]/4+0.5
        for i in range(len(test_label)):    
            test_label[i]=float(test_label[i]/4)+0.5

        syn0,syn1=get_parameters(final_data,final_labels,syn0,syn1)
        hidden_value = 1/(1+np.exp(-(np.dot(test_feature,syn0))))
        output_value= 1/(1+np.exp(-(np.dot(hidden_value,syn1))))
        count=0
        count_1=0
        #calculate accuracy
        for i in range(len(test_label)):
            if abs(output_value[i][0]-test_label[i][0])<0.125:
                count+=1
            if output_value[i][0]>0.625 and test_label[i][0]>0.625:
                count_1+=1
            if output_value[i][0]<0.375 and test_label[i][0]<0.375:
                count_1+=1
            if 0.375<=output_value[i][0]<=0.625 and 0.375<=test_label[i][0]<=0.625:
                count_1+=1
        float_count=float(count)
        float_count_1=float(count_1)
        accuracy+=float_count/len(test_label)
        accuracy_2+=float_count_1/len(test_label)
        print ('accuracy of predicted happiness_index:')
        print(float_count/len(test_label))
        print ('accuracy of predicted positive/negative mood:')
        print(float_count_1 / len(test_label))
        print('-----------------------------------')
    accuracy_final=accuracy/hold
    accuracy_final_2=accuracy_2/hold
    print('******************************************')
    print('10-CV accuracy:', accuracy_final)
    print ('Mean accuracy of predicted positive/negative mood:', accuracy_final_2)
    
    
    with open('input_layer.csv','wb') as input_file:
        writer=csv.writer(input_file)
        for line in syn0:
            writer.writerow(line)
    with open('hidden_layer.csv','wb') as hidden_file:
        writer=csv.writer(hidden_file)
        for line in syn1:
            writer.writerow(line)

