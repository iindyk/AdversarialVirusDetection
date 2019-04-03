from data import dataset_parsers as dp
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
import numpy as np
from statsmodels import robust


n_initial = 100
kappa = 0.5
p = 50
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
n, m = np.shape(train_dataset)
print('data read complete')
svc = svm.SVC(kernel='linear', C=1.).fit(train_dataset[:n_initial], train_labels[:n_initial])
err_init = 1 - accuracy_score(test_labels, svc.predict(test_dataset))
print('error on initial set = ', err_init)

# construct P
directions = []
for i in range(p):
    # take direction between 2 random points in the training set
    indices = np.random.randint(low=n_initial, high=n, size=2)
    new_dir = train_dataset[indices[0]] - train_dataset[indices[1]]
    new_dir /= np.linalg.norm(new_dir)
    directions.append(new_dir)

directions = np.array(directions)

# separate training set
train_dataset_pos = []
train_dataset_neg = []

for i in range(n_initial, n):
    if train_labels[i] == 1:
        train_dataset_pos.append(train_dataset[i])
    else:
        train_dataset_neg.append(train_dataset[i])

train_dataset_pos = np.array(train_dataset_pos)
n_pos = len(train_dataset_pos)
n_pos_refined = int(np.floor(n_pos*kappa))
train_dataset_neg = np.array(train_dataset_neg)
n_neg = len(train_dataset_neg)
n_neg_refined = int(np.floor(n_neg*kappa))

# calculate SD outlyingness for pos
sd_pos = np.zeros(n_pos)
for i in range(n_pos):
    for a in directions:
        sd = abs(a@train_dataset_pos[i]-np.median(train_dataset_pos@a))/robust.scale.mad(train_dataset_pos@a)
        if sd > sd_pos[i]:
            sd_pos[i] = sd

# calculate SD outlyingness for neg
sd_neg = np.zeros(n_neg)
for i in range(n_neg):
    for a in directions:
        sd = abs(a@train_dataset_neg[i]-np.median(train_dataset_pos@a))/robust.scale.mad(train_dataset_pos@a)
        if sd > sd_neg[i]:
            sd_neg[i] = sd

pos_indices_refined = sd_pos.argsort()[:n_pos_refined]
neg_indices_refined = sd_neg.argsort()[:n_neg_refined]

train_data_refined = np.append(train_dataset_pos[pos_indices_refined],
                               train_dataset_neg[neg_indices_refined], axis=0)
train_data_refined = np.append(train_data_refined, train_dataset[:n_initial], axis=0)
train_labels_refined = np.append(np.ones(n_pos_refined), -1*np.ones(n_neg_refined))
train_labels_refined = np.append(train_labels_refined, train_labels[:n_initial])

svc = svm.SVC(kernel='linear', C=1.).fit(train_data_refined, train_labels_refined)
err_sd = 1 - accuracy_score(test_labels, svc.predict(test_dataset))
print('error with SD SVM = ', err_sd)
