import dataset_parsers as dp
import os
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
import numpy as np


filename = "/home/iindyk/PycharmProjects/AdversarialVirusDetection/dictionary.txt"
# dp.create_dictionary()
dictionary = open(filename, 'r').read().split()
dp.files2freq_pickle(os.getcwd()+'/dumps', dictionary, 0.2, 0.2)

C = 1.0
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
svm = svm.SVC(kernel='linear', C=C).fit(train_dataset, train_labels)
err_test = 1 - accuracy_score(test_labels, svm.predict(test_dataset))
err_train = 1 - accuracy_score(train_labels, svm.predict(train_dataset))
err_valid = 1 - accuracy_score(valid_labels, svm.predict(valid_dataset))
print(err_test)
print(err_train)
print(err_valid)
print(svm.coef_)
