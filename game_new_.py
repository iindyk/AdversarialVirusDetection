from data import dataset_parsers as dp
import classifier as cl
import adversary_gd as adv
import numpy as np
import graphing as gr
from sklearn.metrics import accuracy_score
import sklearn.svm as svm


n_initial = 100  # size of initial training dataset
n_steps = 100     # number of steps in game
C = 1.0          # classifier parameter
norm_trsh = 0.06
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
m = len(train_dataset[0])
print('data read complete')
steps = np.arange(n_initial, len(train_labels), int(np.ceil((len(train_labels)-n_initial)/n_steps)))
svm1 = svm.SVC(kernel='linear', C=C).fit(train_dataset, train_labels)
err_best = 1 - accuracy_score(test_labels, svm1.predict(test_dataset))
print('best possible performance on test dataset is ', 1-err_best)
classifiers = []


classifier1 = cl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 30)
err_test1 = 1 - accuracy_score(test_labels, classifier1.predict(test_dataset))
print('classifier1: initiated, error on test dataset is ', err_test1)
classifiers.append(classifier1)

classifier2 = cl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 10, 'asc')
err_test2 = 1 - accuracy_score(test_labels, classifier2.predict(test_dataset))
print('classifier2: initiated, error on test dataset is ', err_test2)
classifiers.append(classifier2)

classifier3 = cl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 50, 'desc')
err_test3 = 1 - accuracy_score(test_labels, classifier3.predict(test_dataset))
print('classifier3: initiated, error on test dataset is ', err_test3)
classifiers.append(classifier3)

classifier4 = cl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 10000)
err_test4 = 1 - accuracy_score(test_labels, classifier4.predict(test_dataset))
print('classifier4: initiated, error on test dataset is ', err_test4)
classifiers.append(classifier4)

classifier5 = cl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C, 10000)
err_test5 = 1 - accuracy_score(test_labels, classifier5.predict(test_dataset))
print('classifier5: initiated, error on test dataset is ', err_test5)
classifiers.append(classifier5)


adversary = adv.Adversary(train_dataset[:n_initial, :], train_labels[:n_initial], test_dataset, test_labels,
                         len(train_dataset[0]))
test_errs = np.zeros((len(classifiers), len(steps)))
test_errs[0, 0] = err_test1
test_errs[1, 0] = err_test2
test_errs[2, 0] = err_test3
test_errs[3, 0] = err_test4

data_batches = []
labels_batches = []
inf_batches = []
false_pos = np.zeros((len(classifiers), len(steps)))
false_neg = np.zeros((len(classifiers), len(steps)))
for i in range(len(steps)-1):
    print('step #', i)
    new_train_dataset = train_dataset[steps[i]:steps[i+1], :]
    new_train_labels = train_labels[steps[i]:steps[i+1]]
    new_infected_data, attack_norm = adversary.get_infected_dataset(new_train_dataset, new_train_labels)
    print('adversary: data is infected, attack norm: ', attack_norm)
    data_batches.append(new_infected_data)
    labels_batches.append(new_train_labels)
    inf_batches.append(attack_norm > norm_trsh)
    for j in range(len(classifiers)):
        t_d_tmp, t_l_tmp = train_dataset[:n_initial, :], train_labels[:n_initial]
        for d_batch, l_batch, infected in zip(data_batches, labels_batches, inf_batches):
            if classifiers[j].is_valid(d_batch, l_batch):
                t_d_tmp, t_l_tmp = np.append(t_d_tmp, d_batch, axis=0), np.append(t_l_tmp, l_batch)
                if infected:
                    false_neg[j, i+1] += 1./sum(inf_batches)
                # adversary.eps *= 1.1
            else:
                if not infected:
                    false_pos[j, i+1] += 1./sum([not inf for inf in inf_batches])
                # adversary.eps *= 0.9

        if j == 3:
            classifiers[3].fit(train_dataset[:steps[i + 1], :], train_labels[:steps[i + 1]])
        else:
            classifiers[j].fit(t_d_tmp, t_l_tmp)
        # todo: strategy for classifier.crit_value & adversary.eps change
        test_errs[j, i+1] = 1 - accuracy_score(test_labels, classifiers[j].predict(test_dataset))

    #adversary.add_train_data(new_infected_data, new_train_labels)

print('false_pos', false_pos[:, -1])
print('false_neg', false_neg[:, -1])
gr.graph_multidim_results(test_errs, ['constant', 'INC', 'DEC', 'no perturbation'], n_steps)
