from data import dataset_parsers as dp
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
import numpy as np
import adversary_simple as adv
from statsmodels import robust


n_initial = 100
kappa = 0.5
p = 50
n_steps = 100
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = dp.read_data()
n, m = np.shape(train_dataset)
print('data read complete')
svc = svm.SVC(kernel='linear', C=1.).fit(train_dataset[:n_initial], train_labels[:n_initial])
err_init = 1 - accuracy_score(test_labels, svc.predict(test_dataset))
print('error on initial set = ', err_init)

# get infected training set

steps = np.arange(n_initial, len(train_labels), int(np.ceil((len(train_labels)-n_initial)/n_steps)))
adversary = adv.Adversary(train_dataset[:n_initial, :], train_labels[:n_initial], test_dataset, test_labels,
                         m)

train_data_infected = train_dataset[:n_initial]
for i in range(len(steps)-1):
    print('step #', i)
    new_train_dataset = train_dataset[steps[i]:steps[i+1], :]
    new_train_labels = train_labels[steps[i]:steps[i+1]]
    new_infected_data, attack_norm = adversary.get_infected_dataset(new_train_dataset, new_train_labels)
    print('adversary: data is infected, attack norm: ', attack_norm)
    train_data_infected = np.append(train_data_infected, new_infected_data, axis=0)
    if np.random.rand() < 0.5:
        adversary.eps += 0.00001
    else:
        adversary.eps -= 0.00001
n = len(train_data_infected)

# construct P
directions = []
for i in range(p):
    # take direction between 2 random points in the training set
    indices = np.random.randint(low=n_initial, high=n, size=2)
    new_dir = train_data_infected[indices[0]] - train_data_infected[indices[1]]
    new_dir /= np.linalg.norm(new_dir)
    directions.append(new_dir)

directions = np.array(directions)

# separate training set
train_dataset_pos = []
train_dataset_neg = []


for i in range(n_initial, n):
    if train_labels[i] == 1:
        train_dataset_pos.append(train_data_infected[i])
    else:
        train_dataset_neg.append(train_data_infected[i])

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
