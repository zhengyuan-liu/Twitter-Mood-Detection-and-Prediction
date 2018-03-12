# README

## ML.py
### Description
Conduct both linear regression and logistic regression on 'data.csv'.
###Requirements
numpy, pandas, imbalanced-learn, scikit-learn
###Input
The code reads 'data.csv' to train model and reads 'test.csv' to use the trained model for prediction.
###Output
By commenting/uncommenting, the coefficients, Mean Squared Error (MSE) and variance score of the linear regression,
and the accuracy of the logistic regression of each fold during cross validation (CV) would be printed to the console,
as well as the average MSE of the linear regression and the average accuracy of the logistic regression after 10-fold
CV. In addition, the coefficients of both models used for prediction could be printed to the console. And predicted
results could be saved to a new file 'predictions.csv'.
###Usage
There are two parts of the code. From the beginning to the k-fold cross validation loop, it is used for feature
selection. First let data_X contain all 10 candidate features. Then use data_X = np.delete(data_X, np.s_[0], axis=1) to
drop one feature. Change the value of the parameter in np.s_[] to drop different features. The code after k-fold cross
validation loop is used for prediction.

##train.py
###Description
Conduct XGBoost (https://xgboost.readthedocs.io/en/latest/).
###Requirements

numpy, pandas, xgboost

Install XGBoost

On Mac:

    Using Anaconda
    conda install -c anaconda py-xgboost
    Without Anaconda
    brew install gcc5
    pip install xgboost

On Ubuntu (Python package):

    pip install xgboost

###Input
The code reads 'data.csv' to train model.
###Output
The test error of using softmax or softprob with hold-out validation or cross-validation is printed to the console.
###Usage
Set the parameters according to the result of GridSearchCV.py

##GridSearchCV.py
###Description
Parameter grid search with xgboost
###Requirements
numpy, pandas, scikit-learn, xgboost
###Input
The code reads 'data.csv' to train model.
###Output
The best parameters and the corresponding accuracy of the model is printed to the console. The best model is printed to
the console and is saved in file 'XGB.model'.
###Usage
Change the param_grid to search the best parameters.

##predict.py
###Description
Predict happiness index from tweets.
###Requirements
pandas, xgboost
###Input
The code uses the model in 'XGB.model' to do prediction. The code reads all csv file in the folder './test_data/' and
do prediction on each dataset.
###Output
The predicted happiness index along with the created time of the tweet are saved in files '*_result.csv' under the
folder './predictions/'.
###Usage
Make sure 'XGB.model' is in the same directory, all the test data csv files is in the folder './test_data/' named
as 'test_*_nlp.csv', and there is a folder './predictions', then run this python file.

##svm_classifier.py
###Libraries Needed:
pandas, numpy, sklearn, imblearn, LIBSVM
###Description:
This file read our preprocessed corpus, selected features, then do the random oversampling to solve the
unbalanced data problem, and use the 10-Fold cross validation to compute the accuracy of the model, and finally use all
the training data to train the final model.
###Input:
Our preprocessed corpus: data_with_nlp.csv
###Output:
Average accuracy of our svm model by 10-Fold cross validation,
The final svm model: happiness_index_svm.model
###Usage:
Let data_with_nlp.csv in the specified folder, prepare all the libraries needed, and run.

##svm_test.py
###Libraries Needed:
pandas, sklearn, LIBSVM
Description: This file read our testing file (Twitter users' tweets in a timeline), then standardize the input data,
and use svm model to compute happiness index for every record.
###Input:
testing file: test_disney_nlp.csv/test_trump_nlp.csv/test_neg_god_nlp.csv
trained svm model: happiness_index_svm.model
###Output:
Twitter users' happiness index in a timeline

##NN.py
###Libraries Needed: 
pandas, numpy, sklearn, random, islice
###Description:
This file constructs a neural network with one hidden layer to predict the happiness index by using the selected features.
Firstly, the file reads our preprocessed corpus, which are stored in CSV file and contains the features that we selected to use in model construction.
Secondly, the file do the random oversampling to solve the unbalanced data problem, (i.e. the number of data for each happiness index differs greatly, which will affect the accuracy of prediction.)
Thirdly, the file randomly chooses 1 out of 15 data in the training data set to train the weights for the 2 layers.
Finally, the file uses 10-Fold cross validation to compute the accuracy of the model.
###Input: 
data_with_nlp.csv
###Output: 
weights for 2 layers after training, average accuracy of our neural network model by 10-Fold cross validation.
###Usage: 
Let data_with_nlp.csv and nn7.py in the same fold, prepare all the libraries needed, and run.

##predict_NN
###Libraries Needed: 
pandas, numpy, csv, sklearn
###Description: 
This file use the weights we get from NN7.py to predict the happiness index of data we collected. First we
read the information from test_trump_nlp.csv, then we extracted related attributes and use the NN model we built to
obtain happiness index of these twitters. Finally, we output our results into a new file called "predict_value.csv".
###Input: 
input_layer.csv, output_layer.csv, test_trump_nlp.csv 
###Output: 
predict_value.csv
###Usage: 
Let input_layer.csv, ouput_layer.csv, test_trump_nlp.csv and predict_NN.py in the specified fold, prepare all the
libraries needed, and run.

##vis.py
###Description
Visualize twitter users' predicted happiness index using two kinds of graphs: line chart and heat map.
###Requirements
pandas, plotly
###Input
The code reads either a csv file from given filename or all csv files under the given directory ('./predictions/*.csv'
by default).
###Output
Two html files are generated as the visualization results of the prediction.
###Usage
Before running, make sure the data files are named as 'test_*_result.csv'.
Set aggregate_by_date: whether the predicted values should be aggregated by date;
Set log_value: whether perform the natural logarithm on the predicted values;
Select the method of loading data: either load a single csv file or load all csv files under a directory.

##Structure of this Project

* --report.pdf                  The final report in PDF

* --README.md                   The REAME file for the structure of the submission file (This file)

* --codes/                  The directory of codes

    *  --**codes**                  All codes of this project

    *  --README.txt             The README file for codes

    *  --run.sh                 The script to execute codes

* --data/                   The directory of data

    *  --**data**                   All data of this project

    *  --README.txt             The README file for data

* --prediction/                 The directory of prediction results for our test data

    *  --test_*username*_result.csv     Prediction results by XGBoost

    *  --test_*username*_svm_result.csv     Prediction results by SVM

* --Result Visualization/           The visualization file of our prediction results

    *  --SVM/                   The directory of the visualization file of our prediction results by SVM
   
        *  --line_chart_*username*_svm.html
   
    *  --XGBoost                    The directory of the visualization file of our prediction results by XGBoost
   
        * --line_chart_*username*_xgboost.html

