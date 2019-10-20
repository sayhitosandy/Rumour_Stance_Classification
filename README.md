# Rumour Stance Classification using Improved Feature Extraction  

The task of determining veracity and authenticity of social media content has been of recent interest to the field of NLP. False claims and rumours affect peoples perceptions of events and their behaviour, sometimes in harmful ways. Here, the task is to classify tweets in threads of tweets based on the stance possessed by the tweet, which can be of 4 categories: supporting (S), denying (D), querying (Q), or commenting (C), i.e., SDQC.   

By improving feature extraction and incorporating tweet dependent (and also other textual) features such as hashtags, link content, etc., we were able to achieve accuracies close to that of the state of the art on most of our models. Our highest reported accuracy is 77.6%, which is comparable to that of the state of the art model (78.4%). We also tried to improve the recall on Deny and Query classes by augmenting the training dataset. We also tried to do the same using ensemble methods, and bagging/boosting techniques. 

#### Links
1. [*SemEval-2017 Task 8 Dataset*](http://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools)
2. [*Report*](https://github.com/sayhitosandy/Rumour_Stance_Classification/blob/master/Report.pdf)

##### Dataset Structure:
- The `semeval2017-task8-dataset` folder in the `Dataset` folder contains the training data and `semeval2017-task8-test-data` contains test data. It is a public dataset. Also, the labeled tweets in form `tweet id:label` are in `test_label.json`.
- We also use `GoogleNews-vectors-negative300` for `word2Vec` features. `GoogleNews-vectors-negative300` folder is required by `Data.py` script. This is available at https://github.com/mmihaltz/word2vec-GoogleNews-vectors.
- `badwords.txt` contains the list of swear words which is used to find the features in `Data.py`.
- We have additionally labeled datasets for Las Vegas Shooting and California Shooting which are available at https://docs.google.com/spreadsheets/d/1fnZwO-f14QSKVbruinEX0KUV86t5kEHuyslxLs60fuI/edit?usp=sharing and https://docs.google.com/spreadsheets/d/1Y0cfFmK82J6KGjQpN9BuAGnm4oSFBcMgnkGTZ_MWhHU/edit?usp=sharing. A combined csv for only the labeled samples is in `newdata.csv`. The raw json tweets are also in `las_vegas_shootout.json` and `tweets_california_shootout.json`.

#### Steps to Run the Code:
Python version: 2.7  
1. First run the `Data.py` file. This python script will will read the dataset, preprocess it and find the features of the tweets. These features are used by different classifiers. The features are stored as `pickle` files.
2. To get the results of SVM and Logestic Regression, run the `SVM_LRClassifier.py` file.
3. `nn.py` is a python script for the neural network model. Run the `nn.py` file to get the results for neural network model. Keras Library is used in this script.
4. Run the `NBClassifier.py` file for Naive bayes results.
5. The jupyter notebook `ngram_models.ipynb` (Python version: 3.6) includes code and output for:
- loading the rumour eval training and test data
- preprocessing the data
- vectorizing the data using CountVectorizer and TFIDF, for Unigram, Bigram and Trigram
- running different classifiers on it, including
  - MultinomialNB
  - SVM
  - Logistic Regression
  - RandomForest
  - XGBoost
- an attempt at LSTM
- loading and collecting the newly collected data

#### References
1. Derczynski et. Al, SemEval-2017 Task 8: RumourEval: Determining rumour veracity and support for rumours.
2. Kochkina et. Al, Turing at SemEval-2017 Task 8: Sequential Approach to Rumour Stance Classification with Branch-LSTM.
3. Dataset Link, SemEval-2017 Task 8 Dataset.
4. Bahuleyan et. Al, UWaterloo at SemEval-2017 Task 8: Detecting Stance towards Rumours with Topic Independent Features.