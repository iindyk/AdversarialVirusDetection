import dataset_parsers as dp
import players as pl
import numpy as np


n_initial = 100 # size of initial training dataset
n_steps = 10    # number of steps in game
C = 1.0         # classifier parameter
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
steps = np.arange(n_initial, len(train_labels), np.ceil((len(train_labels)-n_initial)/n_steps))
classifier = pl.Classifier(train_dataset[:n_initial, :], train_labels[:n_initial]
                           , valid_dataset, valid_labels, C)
adversary = pl.Adversary(train_dataset[:n_initial, :], train_labels[:n_initial], test_dataset, test_labels)

print(steps)
#for i in range(n_steps):
