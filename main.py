import dataset_parsers as dp
import os
import sklearn.svm as svm
from sklearn.metrics import accuracy_score


filename = "/home/iindyk/PycharmProjects/AdversarialVirusDetection/dictionary.txt"
# dp.create_dictionary()
dictionary = open(filename, 'r').read().split()
dp.files2freq_pickle(os.getcwd()+'/dumps', dictionary, 0.2, 0.2)

#C = 1.0
#train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
#svm = svm.SVC(kernel='linear', C=C).fit(train_dataset, train_labels)
#pred_labels = svm.predict(test_dataset)
#err = 1 - accuracy_score(test_labels, pred_labels)
#print(err)
