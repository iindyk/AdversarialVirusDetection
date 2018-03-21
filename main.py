from data import dataset_parsers as dp
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
import players as pl
import numpy as np
import graph_2dim_results as gr

# saving data to pickle file
# filename = "/home/iindyk/PycharmProjects/AdversarialVirusDetection/data/dictionary.txt"
# # dp.create_dictionary()
# dictionary = open(filename, 'r').read().split()
# dp.files2freq_pickle(os.getcwd()+'/data/dumps', dictionary, 0.2, 0.2)
#
# C = 1.0
# train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
# svm = svm.SVC(kernel='linear', C=C).fit(train_dataset, train_labels)
# err_test = 1 - accuracy_score(test_labels, svm.predict(test_dataset))
# err_train = 1 - accuracy_score(train_labels, svm.predict(train_dataset))
# err_valid = 1 - accuracy_score(valid_labels, svm.predict(valid_dataset))
# print(err_test)
# print(err_train)
# print(err_valid)
# print(svm.coef_)


# testing one time adversarial attack
C = 1.0
n = 1000
m = 2
train_dataset, train_labels, test_dataset, test_labels = dp.get_toy_dataset(n, m)
svm_orig = svm.SVC(kernel='linear', C=C).fit(train_dataset, train_labels)
err_orig = 1 - accuracy_score(test_labels, svm_orig.predict(train_dataset))
adversary = pl.Adversary(None, None, train_dataset, train_labels, len(train_dataset[0]))
train_dataset_infected, _ = adversary.get_infected_dataset(train_dataset, train_labels)
svm_infected = svm.SVC(kernel='linear', C=C).fit(train_dataset_infected, train_labels)
err_infected = 1 - accuracy_score(test_labels, svm_infected.predict(test_dataset))
print('err by orig svm is ', err_orig)
print('err by infected svm is ', err_infected)
colors = []
inf_points = []
for i in range(int(0.5*n)):
    if train_labels[i]==1:
        colors.append((1, 0, 0))
    else:
        colors.append((0, 0, 1))
    if np.linalg.norm(train_dataset_infected[i]-train_dataset[i])>0.01:
        inf_points.append(i)

gr.graph_results(0, 1,train_dataset, train_labels, train_dataset_infected, inf_points,
                 colors, svm_orig, svm_infected)