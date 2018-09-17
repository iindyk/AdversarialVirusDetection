from sklearn.svm import SVC
from random import random


n = 1000
dataset = []
labels = []
for i in range(n):
    new_x = random()
    new_y = random()
    dataset.append([new_x, new_y])
    labels.append(1 if new_x+new_y > 1 else -1)

svc = SVC(kernel='linear')
svc.fit(dataset, labels)
print(svc.dual_coef_)
print(svc.n_support_)
print(svc.support_)