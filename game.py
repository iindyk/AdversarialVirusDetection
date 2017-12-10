import dataset_parsers as dp
import players as pl
import numpy as np
from sklearn.metrics import accuracy_score
import sklearn.svm as svm


n_initial = 100  # size of initial training dataset
n_steps = 100     # number of steps in game
C = 1.0          # classifier parameter
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
steps = np.arange(n_initial, len(train_labels), int(np.ceil((len(train_labels)-n_initial)/n_steps)))
svm1 = svm.SVC(kernel='linear', C=C).fit(train_dataset, train_labels)
err_best = 1 - accuracy_score(test_labels, svm1.predict(test_dataset))
print('best possible performance on test dataset is ', err_best)
classifiers = []

classifier1 = pl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 10)
err_test1 = 1 - accuracy_score(test_labels, classifier1.predict(test_dataset))
print('classifier1: initiated, error on test dataset is ', err_test1)
classifiers.append(classifier1)

classifier2 = pl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 20)
err_test2 = 1 - accuracy_score(test_labels, classifier2.predict(test_dataset))
print('classifier2: initiated, error on test dataset is ', err_test2)
classifiers.append(classifier2)

classifier3 = pl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 30)
err_test3 = 1 - accuracy_score(test_labels, classifier3.predict(test_dataset))
print('classifier3: initiated, error on test dataset is ', err_test3)
classifiers.append(classifier3)

classifier4 = pl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 10, 'asc')
err_test4 = 1 - accuracy_score(test_labels, classifier4.predict(test_dataset))
print('classifier4: initiated, error on test dataset is ', err_test4)
classifiers.append(classifier4)

classifier5 = pl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 30, 'desc')
err_test5 = 1 - accuracy_score(test_labels, classifier5.predict(test_dataset))
print('classifier5: initiated, error on test dataset is ', err_test5)
classifiers.append(classifier5)


adversary = pl.Adversary(train_dataset[:n_initial, :], train_labels[:n_initial], test_dataset, test_labels)
test_errs = np.zeros((len(classifiers), len(steps)))
test_errs[0,0] = err_test1
test_errs[1,0] = err_test2
test_errs[2,0] = err_test3
test_errs[3,0] = err_test4
test_errs[4,0] = err_test5

for i in range(len(steps)-1):
    print('step #', i)
    new_train_dataset = train_dataset[steps[i]:steps[i+1], :]
    new_train_labels = train_labels[steps[i]:steps[i+1]]
    new_infected_data, attack_norm = adversary.get_infected_dataset(new_train_dataset, new_train_labels)
    print('adversary: data is infected, attack norm: ', attack_norm)
    for j in range(len(classifiers)):
        if classifiers[j].is_valid(new_infected_data, new_train_labels):
            print('classifier'+str(j+1)+': data is valid, adding to training set')
            classifiers[j].partial_fit(new_infected_data, new_train_labels)
            adversary.eps = adversary.eps*1.1
        else:
            print('classifier'+str(j+1)+': data is not valid, disregarding')
            adversary.eps = adversary.eps*0.9
        # todo: strategy for classifier.crit_value & adversary.eps change
        test_errs[j, i+1] = 1 - accuracy_score(test_labels, classifiers[j].predict(test_dataset))
    adversary.add_train_data(new_infected_data, new_train_labels)
    #err_test = 1 - accuracy_score(test_labels, classifier1.predict(test_dataset))
    #print('classifier: error on validation dataset is ', err_valid)
    #print('classifier: error on test dataset is ', err_test)

#svm2 = svm.SVC(kernel='linear', C=C).fit(classifier.train_dataset, classifier.train_labels)
#err_check = 1 - accuracy_score(test_labels, svm2.predict(test_dataset))
#print('svm check error on test dataset ', err_check)
for i in range(len(classifiers)):
    print('classifier'+str(i+1)+' errors are '+str(test_errs[i, :]))
