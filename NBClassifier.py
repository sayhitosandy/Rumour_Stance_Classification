import os
import sys
import pickle
import json
from sklearn.naive_bayes import MultinomialNB as nb
from sklearn.linear_model import SGDClassifier as svm

nb_clf = nb.fit(train, labels)
svm_clf = svm.fit(train, labels)