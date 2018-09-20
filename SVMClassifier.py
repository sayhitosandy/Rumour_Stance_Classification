import os, sys
import pickle
import json
# from sklearn.naive_bayes import MultinomialNB as nb
from sklearn.linear_model import SGDClassifier as svm
import numpy as np

cwd = os.getcwd() #Current working directory

#Read training data
f = open(os.path.join(cwd, r'training.pkl'), 'rb')
(X_data, X_label) = pickle.load(f)
f.close()

#Read test data
f = open(os.path.join(cwd, r'testing.pkl'), 'rb')
(Y_data, Y_label) = pickle.load(f)
f.close()

train = []
trainLabel = []

label2no = {u'support':0, u'query':1, u'deny':2, u'comment':3}

#Convert list of lists to nd array (Required for SVM Training)
for key in X_label.keys():
	train.append(X_data[key])
	trainLabel.append(label2no[X_label[key]])

train = np.array(train)
trainLabel = np.array(trainLabel)

#SVM Training
svm_clf = svm(loss='squared_hinge', max_iter=1000, penalty='l2').fit(train, trainLabel.transpose())

test = []
testLabel = []

for key in Y_label.keys():
	test.append(Y_data[key])
	testLabel.append(label2no[Y_label[key]])

test = np.array(test)
testLabel = np.array(testLabel)

#SVM Testing
predicted = svm_clf.predict(test)

#Accuracy
print(np.mean(predicted == testLabel)*100)
