Structure of the Submission File

11_DividedBy0
|--report.pdf					The final report in PDF
|--README.txt					The REAME file for the structure of the submission file (This file)
|--codes/					The directory of codes
|  |--**codes**					All codes of this project
|  |--README.txt				The README file for codes
|  |--run.sh					The script to execute codes
|--data/					The directory of data
|  |--**data**					All data of this project
|  |--README.txt				The README file for data
|--prediction/					The directory of prediction results for our test data
|  |--test_*username*_result.csv		Prediction results by XGBoost
|  |--test_*username*_svm_result.csv		Prediction results by SVM
|--Result Visualization/			The visualization file of our prediction results 
   |--SVM/					The directory of the visualization file of our prediction results by SVM
   |  |--line_chart_*username*_svm.html
   |--XGBoost					The directory of the visualization file of our prediction results by XGBoost
      |--line_chart_*username*_xgboost.html