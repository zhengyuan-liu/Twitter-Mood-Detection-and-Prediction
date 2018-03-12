Instructions for NLP.py and predict.py:
NLP.py takes input train.csv and train the NN model with it. It will save the model into 3 files: dictionary_.json, model_newnlp.json and model_newnlp.h5. These 3 files will provide the needed weights and dictionary for the predict.py.

predict.py makes prediction based on the input test dataset (for instance: test_neg_god.csv) and the saved models from the NLP.py. The predicted result will be attached to the input test file as an independent column. And also some other features will be extracted and attached to the csv file. 

Environment requirements:
keras
tensorflow
numpy
pandas

Running: 
python NLP.py
python predict.py
