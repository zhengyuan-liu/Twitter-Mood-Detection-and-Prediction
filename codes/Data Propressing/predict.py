import json
import pandas as pd
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

# we're still going to use a Tokenizer here, but we don't need to fit it
tokenizer = Tokenizer(num_words=10000)
# for human-friendly printing
labels = ['-1','1']

# read in our saved dictionary
dictionary = []
with open('dictionarylarge.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file, )
# for line in open('dictionary.txt', 'r'):
#     dictionary.append(json.loads(line))
# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        # else:
        #     print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices

# read in your saved model structure
json_file = open('modellarge.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('modellarge.h5')

# okay here's the interactive part

df1 = pd.read_csv("labeleddata/outtotal.csv")
test_tweets = df1['tweets'].values
# sentiment_test = df1['label'].values
test_labels = df1['happiness_index']
count  = 0
totalcount = 0
nlppred = []
for i in range(0, len(test_tweets)):
# format your input for the neural net
    testArr = convert_text_to_index_array(test_tweets[i])
    input = tokenizer.sequences_to_matrix([testArr], mode='binary')
    # predict which bucket your input belongs in
    pred = model.predict(input)
    nlppred.append(labels[np.argmax(pred)])

#     if(labels[np.argmax(pred)] == str(test_labels[i])):
#         count+=1
#     totalcount+=1
#     # and print it for the humons
# print(float(count)/float(totalcount))

df1 = df1.assign(nlppred = nlppred)
df1.to_csv('data.csv')