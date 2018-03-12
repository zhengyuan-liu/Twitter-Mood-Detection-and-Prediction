import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
# fix random seed for reproducibility
np.random.seed(7)
top_words = 10000
df = pd.read_csv("output_3.csv",error_bad_lines=False)
X_train = df['SentimentText'].values
y_train = df['Sentiment'].values

# X_test = df1['tweets'].values
# num_labels = len(np.unique(y_train))
# y_test = df['happiness_index']

tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(X_train)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# list = []
# for d in dictionary:
#     unicode(d, errors='replace')
#     list.append(d)
# Let's save this out so we can use it later
dict = {}
with open('dictionary.json', 'w') as dictionary_file:
    for d, v in dictionary.iteritems():
        try:
            json.dump(d, dictionary_file)
            dict[d] = v
        except UnicodeDecodeError:
            print d

with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dict, dictionary_file)
def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in X_train:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = keras.utils.to_categorical(y_train, 2)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(512, input_shape=(top_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])

model.fit(train_x, train_y,
  batch_size=32,
  epochs=4,
  verbose=1,
  validation_split=0.1,
  shuffle=True)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

