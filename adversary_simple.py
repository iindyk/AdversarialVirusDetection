import numpy as np
from memory_profiler import profile
import datetime
import sklearn.svm as svm


class Adversary:
    eps = 0.02  # 0.04
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
        n, m = np.shape(new_train_data)

        #
        if np.random.random() < 0.5:  # < 0.5:
            return new_train_data, 0.
        #
        if len(self.train_dataset) != 0:
            train_data_ext = np.append(new_train_data, self.train_dataset, axis=0)
            train_labels_ext = np.append(new_train_labels, self.train_labels)
        else:
            train_data_ext = new_train_data
            train_labels_ext = new_train_labels

        # build svm on extended data and get parameters
        svc = svm.SVC(kernel='linear', C=self.a).fit(train_data_ext, train_labels_ext)
        w = svc.coef_[0]  # normal vector
        b = svc.intercept_[0]  # intercept

        # construct extended dual variables vector
        l_ext = np.zeros(len(train_labels_ext))
        tmp_i = 0
        n_ext = len(train_labels_ext)
        for i in range(n_ext):
            if i in svc.support_:
                l_ext[i] = svc.dual_coef_[0, tmp_i] * train_labels_ext[i]
                tmp_i += 1
        h = np.array([l*w for l in l_ext[:n]])
        h = self.eps*n*h/np.linalg.norm(h)

        return new_train_data+h, np.linalg.norm(h)/n
