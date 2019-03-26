import numpy as np
from memory_profiler import profile
import datetime
import sklearn.svm as svm


class Adversary:
    eps = 0.1  # 0.04
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
        delta = 1e-2
        step = 1e-1
        n, m = np.shape(new_train_data)
        n_t = len(self.test_labels)

        #
        if np.random.random() < 0.5: return new_train_data, 0.
        #

        def obj_grad(train_data_current):   # returns approximate gradient of adversary's objective

            # get extended dataset
            if len(self.train_dataset) != 0:
                train_data_ext = np.append(train_data_current, self.train_dataset, axis=0)
                train_labels_ext = np.append(new_train_labels, self.train_labels)
            else:
                train_data_ext = train_data_current
                train_labels_ext = new_train_labels

            # build svm on extended data and get parameters
            svc = svm.SVC(kernel='linear', C=self.a).fit(train_data_ext, train_labels_ext)
            w = svc.coef_[0]        # normal vector
            b = svc.intercept_[0]   # intercept

            # construct extended dual variables vector
            l_ext = np.zeros(len(train_labels_ext))
            tmp_i = 0
            n_ext = len(train_labels_ext)
            for i in range(n_ext):
                if i in svc.support_:
                    l_ext[i] = svc.dual_coef_[0, tmp_i]*train_labels_ext[i]
                    tmp_i += 1
            l = l_ext[:n]

            # get approximate gradient of w
            dw_dh = np.array([[l[i]*new_train_labels[i] for j in range(m)] for i in range(n)])

            # get approximate gradient of b
            # 1: find point on the margin's boundary
            idx = 0
            for i in range(n):
                if 0.001 < l[i] < 0.999:
                    idx = i
                    break
            db_dh = np.array([np.multiply(dw_dh[j, :], new_train_data[idx]) for j in range(n)])

            obj_grad_val = np.zeros((n, m))
            for k in range(n_t):
                bin_ = self.test_labels[k] if self.test_labels[k] * (np.dot(w, self.test_dataset[k]) + b) > -1 else 0.0
                if bin_ != 0:
                    for i in range(n):
                        obj_grad_val[i, :] += (np.multiply(dw_dh[i, :], self.test_dataset[k, :])+db_dh[i, :])*bin_

            return obj_grad_val/n_t

        # perform gradient descent
        nit = 0
        h = np.zeros((n, m))
        change = np.ones((n, m))
        _train_data_current = np.copy(new_train_data)
        while nit < maxit and np.linalg.norm(change) > delta:
            _train_data_current += change
            change = -1*step*obj_grad(_train_data_current)
            h += change
            nit += 1
            if np.linalg.norm(h) >= self.eps*n:
                #break
                h = self.eps * n * h / np.linalg.norm(h)

        print(nit)

        return new_train_data+h, np.linalg.norm(h)/n
