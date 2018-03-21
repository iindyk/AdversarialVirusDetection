from data import dataset_parsers as dp
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
import players as pl

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
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
train_dataset = train_dataset[:1000, :]
train_labels = train_labels[:1000]
svm_orig = svm.SVC(kernel='linear', C=C).fit(train_dataset, train_labels)
err_orig = 1 - accuracy_score(test_labels, svm_orig.predict(test_dataset))
adversary = pl.Adversary(None, None, test_dataset, test_labels, len(train_dataset[0]))
train_dataset_infected, _ = adversary.get_infected_dataset(train_dataset, train_labels)
svm_infected = svm.SVC(kernel='linear', C=C).fit(train_dataset_infected, train_labels)
err_infected = 1 - accuracy_score(test_labels, svm_infected.predict(test_dataset))
print('err by orig svm is ', err_orig)
print('err by infected svm is ', err_infected)
