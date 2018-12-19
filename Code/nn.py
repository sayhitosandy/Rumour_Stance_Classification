# MODEL
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pickle
import os
from sklearn.metrics import classification_report
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import accuracy_score


#Read training data
f = open('training.pkl', 'rb')
(X_data, X_label) = pickle.load(f)
f.close()

#Read test data
f = open('testing.pkl', 'rb')
(Y_data, Y_label) = pickle.load(f)
f.close()

X_train = []
Y_train = []

label2no = {u'support':0, u'query':1, u'deny':2, u'comment':3}

#Convert list of lists to nd array (Required for SVM Training)
for key in X_label.keys():
	temp_lb = np.zeros(4)
	X_train.append(X_data[key])
	temp_lb[label2no[X_label[key]]] = 1
	Y_train.append(temp_lb)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = []
Y_test = []

for key in Y_label.keys():
	temp_lb = np.zeros(4)
	X_test.append(Y_data[key])
	temp_lb[label2no[Y_label[key]]] = 1
	Y_test.append(temp_lb)		

X_test = np.array(X_test)
Y_test = np.array(Y_test)

model = Sequential()


# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(32))
model.add(Activation('relu'))
# model.add(Dropout(0.5))


model.add(Dense(4))
model.add(Activation('sigmoid'))
# COMPILE
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(np.array(X_train), np.array(Y_train), batch_size= 128, validation_split = 0.1, epochs = 40) # tried on 30, 40, 50, 100 # batch size 16, 32
pred = model.predict(np.array(X_test))

count = 0

# print pred

Y_pred = np.zeros((len(pred),4))
for i in range(len(pred)):
	pred_class = np.argmax(pred[i])
	Y_pred[i][pred_class] = 1
print classification_report(np.array(Y_test),Y_pred)
num = np.multiply(Y_pred,Y_test)
den = np.logical_or(Y_pred,Y_test)
# print (1.0*num.sum())/den.sum()
print "accuracy - "
print accuracy_score(np.array(Y_test),Y_pred)

y_score = pred
y_test = np.array(Y_test)


