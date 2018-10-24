import numpy as np
import datetime
import cvxpy as cvx

from sklearn.svm import SVC


class Adversary:
    eps = 1
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

    def get_support_points_split(self, n, simple, svc):
        if simple:
            sup = svc.support_
            dual_c = svc.dual_coef_[0]

            support_eq = []
            support_ineq = []
            non_support = []

            for i in range(n):
                if i not in sup:
                    non_support.append(i)
                elif -0.999 <= dual_c[sup.tolist().index(i)] <= 0.999:
                    support_ineq.append(i)
                else:
                    support_eq.append(i)
            #print([[support_ineq, support_eq, non_support]])
            yield [support_ineq, support_eq, non_support]
            raise StopIteration

        # todo: come up with a better algorithm
        if n == 1:
            yield [[], [0], []]
            raise StopIteration
        else:
            for support_ineq, support_eq, non_support in self.get_support_points_split(n-1, False, None):
                yield [support_ineq+[n-1], support_eq, non_support]
                yield [support_ineq, support_eq+[n-1], non_support]
                yield [support_ineq, support_eq, non_support+[n-1]]

    # @profile
    def get_infected_dataset(self, new_train_data, new_train_labels):
        n, m = np.shape(new_train_data)
        svc = SVC(C=self.a, kernel='linear')
        svc.fit(new_train_data, new_train_labels)
        w_old = svc.coef_[0]
        b_old = svc.intercept_[0]

        b_t = 1000

        h_hat ={} #cvx.Variable((n, n, m))
        for k in range(m):
            h_hat[k] = cvx.Variable(n, n)
        g = cvx.Variable(n, n)
        b = cvx.Variable()
        a_slack = cvx.Variable(n)
        c_slack = cvx.Variable(n)
        c_dual = cvx.Variable(n)
        z = cvx.Bool(n)

        labels = cvx.Constant(np.array(new_train_labels))
        data = cvx.Constant(np.array(new_train_data))

        cons = [a_slack >= -1,
                c_dual >= 0,
                c_dual <= self.a*z,    # todo: double check
                labels * c_dual == 0,
                c_slack >= 0,
                c_slack <= b_t*z]

        w = {}
        for j in range(m):
            num1 = c_dual.T * cvx.vstack([labels[i]*data[i, j] for i in range(n)])
            num2 = labels * cvx.diag(h_hat[j])
            w[j] = num1 + num2

        for i in range(n):
            g_add = 0
            for k in range(n):
                h_kij = cvx.vstack([h_hat[j][k, i] for j in range(m)])
                g_add += (data[k, :]*h_kij)*labels[k]+labels[k]*g[k, i]

            cons.append(a_slack[i] >= labels[i]*(data[i, :]*np.array(list(w))+b))
            cons.append(labels[i]*(data[i, :]*np.array(list(w))+g_add+b) >= 1-c_slack[i])
            cons.append(labels[i]*(data[i, :]*np.array(list(w))+g_add+b)-1+c_slack[i]<=b_t*z[i])

            for j in range(n):
                cons.append(g[i, j] >= -c_dual[i]*(self.eps**2))
                cons.append(g[i, j] <= c_dual[i]*(self.eps**2))

                for k in range(m):
                    cons.append(h_hat[k][j, i] >= -self.eps*c_dual[j])
                    cons.append(h_hat[k][j, i] <= self.eps*c_dual[j])

        obj = cvx.Minimize(cvx.sum_entries(a_slack))
        prob = cvx.Problem(obj, cons)
        prob.solve()
        print(prob.status)
        # print(prob.value)

        # recovering infected dataset
        h = []
        infected_dataset = []

        for i in range(n):
            hi = [(h_hat[j][i, i].value/c_dual[i].value if c_dual[i].value != 0 else 0.0) for j in range(m)]
            h.append(hi)

            infected_dataset.append([new_train_data[i, j]+hi[j] for j in range(m)])

        print(w[0].value)
        print(w[1].value)
        print(b.value)

        return np.array(infected_dataset), np.linalg.norm(h)/n

