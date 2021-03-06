import cvxpy as cvx
import adversary_gd as adv
import data.dataset_parsers as dp
import numpy as np
from sklearn.svm import SVC
import graphing as gf


dataset, labels, _, _ = dp.get_toy_dataset(300, 2)
adv = adv.Adversary([], [], dataset, labels, 2)

infected_dataset, norm = adv.get_infected_dataset(dataset, labels)

# find infected points
infected_points = []
colors = []
for i in range(len(dataset)):
    if np.linalg.norm(np.array(infected_dataset[i])-dataset[i, :]) > 0.1:
        infected_points.append(infected_dataset[i])

    if labels[i] == 1:
        colors.append((1, 0, 0))
    else:
        colors.append((0, 0, 1))

svc_orig = SVC(C=1, kernel='linear')
svc_orig.fit(dataset, labels)

svc_inf = SVC(C=1, kernel='linear')
svc_inf.fit(infected_dataset, labels)
print('attack norm = ', norm)
print('w0= ', svc_inf.coef_[0][0], 'w1= ', svc_inf.coef_[0][1], 'b= ', svc_inf.intercept_[0])

gf.graph_2dim_results(0, 1, dataset, labels, infected_dataset, infected_points, colors, svc_orig, svc_inf)


