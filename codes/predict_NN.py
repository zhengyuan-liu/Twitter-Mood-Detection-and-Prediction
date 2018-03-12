# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 18:59:11 2017

@author: x1310
"""
import csv
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
def sigmoid(x):
    return 1/(1+np.exp(-x))
 

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

if __name__ == '__main__':
    #obtain weights in neural network        
    input_layer=[] 
    hidden_layer=[]
    with open('model/input_layer.csv','rb') as input_file:
        reader=csv.reader(input_file)
        for row in reader:
            input_layer.append(map(eval,row))
    with open('model/hidden_layer.csv','rb') as hidden_file:
        reader=csv.reader(hidden_file)
        for row in reader:
            hidden_layer.append(map(eval,row))
    input_layer=np.array(input_layer)
    hidden_layer=np.array(hidden_layer)
    
    #obtain the data we want to predict
    target='../data/Test Data/test_trump_nlp.csv'
    feature=pd.read_csv(target,error_bad_lines=False)
    feature_X = feature[['user_friends_count', 'user_followers_count', 'retweet_count', 'exclamation_number', 'length',\
                         'question_number', 'uppercase_ratio', 'nlppred']].values
    feature_X=feature_X.astype(np.float64)
    
    z_score(feature_X)
    
    #prediect people's happiness
    hidden_value = sigmoid(np.dot(feature_X,input_layer))
    output_value=  sigmoid(np.dot(hidden_value, hidden_layer))
    output_value=output_value.tolist()
    for index in output_value:
        if index[0] <0.125 :
            index[0]=-2
        elif 0.125<= index[0] <0.375:
            index[0]=-1
        elif 0.375<= index[0] <=0.625:
            index[0]= 0
        elif 0.625< index[0] <=0.875:
            index[0]= 1
        else:
            index[0]= 2
    #print(output_value)
    
    #store output_value
    with open('predict_value.csv','wb')as p:  
        writer=csv.writer(p)
        with open(target,'r') as f:
            reader=csv.reader(f)
            count=-1
            for row in reader:
                count+=1
                if count==0:
                    row.append('predict_value')
                else: 
                    row.append(output_value[count-1][0])
                writer.writerow(row)