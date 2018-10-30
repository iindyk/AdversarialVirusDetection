import numpy as np
from memory_profiler import profile
import datetime
import sklearn.svm as svm


class Adversary:
    eps = 0.02
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

    # @profile
    def get_infected_dataset(self, new_train_data, new_train_labels):
        maxit = 100
        delta = 0.0001
        step = 1e-4
        # todo:train_dataset_ext = np.append(self.train_dataset, new_train_data, axis=0)
        train_data_ext = new_train_data
        n, m = np.shape(new_train_data)
        n_t = len(self.test_labels)

        def obj_grad(train_data_current):
            # build svm on current data and get parameters
            svc = svm.SVC(kernel='linear', C=self.a).fit(train_data_current, new_train_labels)
            w = svc.coef_[0]        # normal vector
            b = svc.intercept_[0]   # intercept

            # construct dual variables vector
            l = np.zeros(n)
            tmp_i = 0
            for i in range(n):
                if i in svc.support_:
                    l[i] = svc.dual_coef_[0, tmp_i]*new_train_labels[i]
                    tmp_i += 1
            # get approximate gradient of w
            dw_dh = np.array([[l[i]*new_train_labels[i] for j in range(m)] for i in range(n)])

            # get approximate gradient of b
            # 1: find point on the margin's boundary
            idx = 0
            for i in range(n):
                if 0.001 < l[i] < 0.999:
                    idx = i
                    break
            db_dh = np.array([l[j]*new_train_labels[j]*new_train_data[idx] for j in range(n)])

            obj_grad_val = np.zeros((n, m))
            for k in range(n_t):
                for i in range(n):
                    bin_ = self.test_labels[k] if self.test_labels[k]*(np.dot(w, self.test_dataset[k])+b) > -1 else 0.0
                    obj_grad_val[i, :] += (np.multiply(dw_dh[i, :], self.test_dataset[k, :])+db_dh[i, :])*\
                                          self.test_labels[k]*bin_

            return obj_grad_val

        # perform gradient descent
        nit = 0
        h = np.zeros((n, m))
        change = np.ones((n, m))
        _train_data_current = np.copy(new_train_data)
        while nit < maxit and np.linalg.norm(change) > delta:
            _train_data_current += h
            change = -1*step*obj_grad(_train_data_current)
            h -= change
            nit += 1
            if np.linalg.norm(h) >= self.eps*n:
                break
        print(nit)
        return new_train_data+h, np.linalg.norm(h)/n