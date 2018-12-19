Dataset Structure:

- The  'semeval2017-task8-dataset' folder in the Dataset folder contains the training data and 'semeval2017-task8-test-data' contains test data. It is a public dataset at http://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools.
Also, the labeled tweets in form tweet id:label are in test_label.json.

- We also use GoogleNews-vectors-negative300 for word2Vec features. GoogleNews-vectors-negative300 folder is required by Data.py script. This is available at https://github.com/mmihaltz/word2vec-GoogleNews-vectors.


- badwords.txt contains the list of swear words which is used to find the features in Data.py.

- We have additionally labeled datasets for Las Vegas Shooting and California Shooting which are available at https://docs.google.com/spreadsheets/d/1fnZwO-f14QSKVbruinEX0KUV86t5kEHuyslxLs60fuI/edit?usp=sharing and 
https://docs.google.com/spreadsheets/d/1Y0cfFmK82J6KGjQpN9BuAGnm4oSFBcMgnkGTZ_MWhHU/edit?usp=sharing
A combined csv for only the labeled samples is in newdata.csv. The raw json tweets are also in las_vegas_shootout.json and tweets_california_shootout.json

------------------------------------------------------------------------------------------------------------------------------

Code Files:

Data.py
nn.py
NBClassifier.py
SVM_LRClassifier.py
ngram_models.ipynb

python version 2.7

1. First run the Data.py file. This python script will will read the dataset and after preprocessing steps finds the features of the tweets. These features are then used by different classifiers.

2. After running the Data.py the features are saved as pickle files. To get the results of SVM and Logestic Regression run the SVM_LRClassifier.py file

3. nn.py is the python script for the neural network model. Run the nn.py file to get the results for neural network model. Keras Library is used in this script.

4. Run the NBClassifier.py file for Naive bayes results


----------------------------------------------------------------------------------------------

The python notebook ngram_models.ipynb (Python version 3.6) includes code and output for

- loading the rumour eval training and test data
- preprocessing the data
- vectorizing the data using CountVectorizer and Tfidf, for unigram, bigram and trigram
- running different classifiers on it, including
-- MultinomialNB
-- SVM
-- Logistic Regression
-- RandomForest
-- XGBoost
- an attempt at LSTM
- loading and collecting the newly collected data
