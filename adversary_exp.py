import numpy as np
import datetime
import cvxpy as cvx


class Adversary:
    eps = 0.1
    a = 1.0

    def __init__(self, initial_train_dataset, initial_train_labels, test_dataset, test_labels, dim):
        self.train_dataset = initial_train_dataset
        self.train_labels = initial_train_labels
        self.test_dataset = test_dataset
        self.test_labels = test_labels
        self.current_w = np.zeros(dim)
        self.current_b = 0.0

    def add_train_data(self, new_train_data, new_train_labels):
        self.train_dataset = np.append(self.train_dataset, new_train_data, axis=0)
        self.train_labels = np.append(self.train_labels, new_train_labels)

    def get_support_points_split(self, dataset):
        support_ineq = dataset[:len(dataset)//3]
        support_eq = dataset[len(dataset)//3 : 2*len(dataset)//3]
        non_support = dataset[2*len(dataset)//3:]
        return [[support_ineq, support_eq, non_support]]

    # @profile
    def get_infected_dataset(self, new_train_data, new_train_labels):
        n, m = np.shape(new_train_data)

        for support_ineq, support_eq, non_support in self.get_support_points_split(new_train_data):
            w = cvx.Variable(m)
            b = cvx.Variable()
            l = cvx.Variable(len(support_eq))
            g = cvx.Variable(n)

            obj = cvx.Minimize(sum([max(-1.0, new_train_labels[i]*(w*new_train_data[i]+b)) for i in range(n)])/n)


