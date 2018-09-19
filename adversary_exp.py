import numpy as np
import datetime
import cvxpy as cvx

from sklearn.svm import SVC


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

    def get_support_points_split(self, n):
        # todo: come up with a better algorithm
        if n == 1:
            yield [[], [0], []]
            raise StopIteration
        else:
            for support_ineq, support_eq, non_support in self.get_support_points_split(n-1):
                yield [support_ineq+[n-1], support_eq, non_support]
                yield [support_ineq, support_eq+[n-1], non_support]
                yield [support_ineq, support_eq, non_support+[n-1]]

    # @profile
    def get_infected_dataset(self, new_train_data, new_train_labels):
        n, m = np.shape(new_train_data)
        svc = SVC(C=self.a, kernel='linear')
        svc.fit(new_train_data, new_train_labels)
        w_old = svc.coef_[0]
        b_old = svc.intercept_

        for support_ineq, support_eq, non_support in self.get_support_points_split(len(new_train_data)):
            sup_ineq_data = [new_train_data[i] for i in support_ineq]
            sup_ineq_labels = [new_train_labels[i] for i in support_ineq]
            sup_eq_data = [new_train_data[i] for i in support_eq]
            sup_eq_labels = [new_train_labels[i] for i in support_eq]
            non_sup_data = [new_train_data[i] for i in non_support]
            non_sup_labels = [new_train_labels[i] for i in non_support]

            w = cvx.Variable(m)
            b = cvx.Variable()
            l = cvx.Variable(len(support_eq))
            g = cvx.Variable(n)
            z = cvx.Variable(n)

            obj = cvx.Minimize(cvx.sum(z)/n)

            cons = [z >= -1,
                    l >= 0,
                    l <= self.a,    # todo: double check
                    l*sup_eq_labels + self.a*sum(sup_ineq_labels) == 0,
                    cvx.norm(g) <= self.eps,
                    w*w_old >= 0.5]
            for i in range(n):
                cons.append(z[i] >= new_train_labels[i]*(w*new_train_data[i]+b))
                if i in support_ineq:
                    cons.append(new_train_labels[i]*(w*new_train_data[i]+g[i]+b) <= 1)
                elif i in support_eq:
                    cons.append(new_train_labels[i] * (w * new_train_data[i] + g[i] + b) == 1)
                else:
                    cons.append(new_train_labels[i] * (w * new_train_data[i] + g[i] + b) >= 1)

            for j in range(m):
                cons.append(w[j] == cvx.sum([l[i]*sup_eq_labels[i]*sup_eq_data[i][j] for i in range(len(support_eq))]) +
                                 sum([self.a*sup_ineq_labels[i]*sup_ineq_data[i][j] for i in range(len(support_ineq))]))

            prob = cvx.Problem(obj, cons)
            prob.solve()
            print(prob.value)
