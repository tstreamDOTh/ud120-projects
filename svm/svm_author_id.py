#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]
print(features_test.shape)
clf = svm.SVC(kernel="linear")
print("Training started...")
clf.fit(features_train, labels_train)
print("Evaluation started...")
test_result = clf.predict(features_test)
print("Accuracy is")
print(accuracy_score(labels_test, test_result))




