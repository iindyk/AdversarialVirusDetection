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
norm_trsh = 0.001
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
m = len(train_dataset[0])
print('data read complete')
steps = np.arange(n_initial, len(train_labels), int(np.ceil((len(train_labels)-n_initial)/n_steps)))
adversary = adv.Adversary(train_dataset[:n_initial, :], train_labels[:n_initial], test_dataset, test_labels,
                         len(train_dataset[0]))

td_cumulative = train_dataset[:n_initial]
lb_cumulative = train_labels[:n_initial]
svm1 = svm.SVC(kernel='linear', C=C).fit(td_cumulative, lb_cumulative)
last_err = 1 - accuracy_score(valid_labels, svm1.predict(valid_dataset))

for i in range(len(steps)-1):
    print('step #', i)
    new_train_dataset = train_dataset[steps[i]:steps[i+1], :]
    new_train_labels = train_labels[steps[i]:steps[i+1]]
    new_infected_data, attack_norm = adversary.get_infected_dataset(new_train_dataset, new_train_labels)
    print('adversary: data is infected, attack norm: ', attack_norm)
    td_tmp = np.append(td_cumulative, new_infected_data, axis=0)
    lb_tmp = np.append(lb_cumulative, new_train_labels)
    svm1 = svm.SVC(kernel='linear', C=C).fit(td_tmp, lb_tmp)
    err = 1 - accuracy_score(valid_labels[:n_initial], svm1.predict(valid_dataset[:n_initial]))
    if err < last_err:
        td_cumulative = np.copy(td_tmp)
        lb_cumulative = np.copy(lb_tmp)
        last_err = err

svm1 = svm.SVC(kernel='linear', C=C).fit(td_cumulative, lb_cumulative)
err = 1 - accuracy_score(test_labels, svm1.predict(test_dataset))
print('test error = ', err)