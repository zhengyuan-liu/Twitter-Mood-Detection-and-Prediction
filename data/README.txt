Structure of the Data Folder

data
|--Raw Data/				The directory of raw crawled data
|  |--Training/				The directory of raw data for training
|  |  |--*keyword*.csv			The csv files of crawled tweets by keywords
|  |--Test/				The directory of raw data for testing
|     |--test_*username*.csv	  	The csv files of crawled tweets by usernames
|--Labeled Data/			The directory of labeled data for training
|  |--*keyword*_*count*.csv		The labeled csv files of tweets by keywords
|--Training Data/			The directory of preprocessed training data
|  |--data_with_nlp.csv			The training corpus with NLP scores (combined by all the training files)
|--Test Data/				The directory of preprocessed test data
   |--test_disney_nlp.csv		The chronological tweets of disney with NLP scores
   |--test_neg_god_nlp.csv		The chronological tweets of god with NLP scores
   |--test_trump_nlp.csv		The chronological tweets of trump with NLP scores