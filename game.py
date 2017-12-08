import dataset_parsers as dp
import players as pl
import numpy as np
from sklearn.metrics import accuracy_score


n_initial = 100  # size of initial training dataset
n_steps = 100     # number of steps in game
C = 1.0          # classifier parameter
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
steps = np.arange(n_initial, len(train_labels), int(np.ceil((len(train_labels)-n_initial)/n_steps)))
classifier = pl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C)
err_test = 1 - accuracy_score(test_labels, classifier.predict(test_dataset))
print('classifier: initiated, error on test dataset is ', err_test)
adversary = pl.Adversary(train_dataset[:n_initial, :], train_labels[:n_initial], test_dataset, test_labels)

for i in range(n_steps-1):
    print('step #', i)
    new_train_dataset = train_dataset[steps[i]:steps[i+1], :]
    new_train_labels = train_labels[steps[i]:steps[i+1]]
    new_infected_data, attack_norm = adversary.get_infected_dataset(new_train_dataset, new_train_labels)
    print('adversary: data is infected, attack norm: ', attack_norm)
    if classifier.is_valid(new_infected_data, new_train_labels):
        print('classifier: data is valid, adding to training set')
        classifier.partial_fit(new_infected_data, new_train_labels)
        adversary.add_train_data(new_infected_data, new_train_labels)
    else:
        print('classifier: data is not valid, disregarding')
        # todo: strategy for classifier.crit_value & adversary.eps change
    err_valid = 1 - accuracy_score(valid_labels, classifier.predict(valid_dataset))
    err_test = 1 - accuracy_score(test_labels, classifier.predict(test_dataset))
    print('classifier: error on validation dataset is ', err_valid)
    print('classifier: error on test dataset is ', err_test)
