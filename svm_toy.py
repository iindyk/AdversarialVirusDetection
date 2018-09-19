import cvxpy as cvx
import adversary_exp as adv
import data.dataset_parsers as dp
from sklearn.svm import SVC

adv = adv.Adversary([], [], [], [], 2)
dataset, labels, _, _ = dp.get_toy_dataset(60, 2)

print(adv.get_infected_dataset(dataset, labels))
