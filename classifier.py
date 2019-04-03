import sklearn.svm as svm
import numpy as np
from sklearn.metrics import accuracy_score


class Classifier:

    def __init__(self, train_dataset, train_labels, valid_dataset, valid_labels, C, crit_val, crit_val_alg='const'):
        self.svc = svm.SVC(kernel='linear', C=C).fit(train_dataset, train_labels)
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels
        self.C = C
        self.val_errors = []
        self.val_errors.append(1 - accuracy_score(self.valid_labels, self.svc.predict(self.valid_dataset)))
        self.crit_val = crit_val
        self.crit_val_alg = crit_val_alg
        self.test_results = [False, False, False]
        # part of test statistics
        self.part_test_stat_h = 0.0
        h_indeces = np.where(valid_labels == 1)
        v_indeces = np.where(valid_labels == -1)
        self.part_test_stat_v = 0.0
        for i in h_indeces:
            for j in h_indeces:
                self.part_test_stat_h += np.linalg.norm(valid_dataset[i, :] - valid_dataset[j, :])
        self.part_test_stat_h /= 2 * (len(h_indeces) ** 2)
        for i in v_indeces:
            for j in v_indeces:
                self.part_test_stat_v += np.linalg.norm(valid_dataset[i, :] - valid_dataset[j, :])
        self.part_test_stat_v /= 2 * (len(v_indeces) ** 2)

    def predict(self, test_dataset):
        return self.svc.predict(test_dataset)

    def get_error(self, test_dataset, test_labels):
        pred_labels = self.predict(test_dataset)
        return 1 - accuracy_score(test_labels, pred_labels)

    def is_valid(self, train_dataset, train_labels, test_stat=0):
        if test_stat == 0:
            test_stat = self.get_test_stat(train_dataset, train_labels)

        #print('classifier: test performed, statistics value is ', test_stat_h + test_stat_v)
        if test_stat > self.crit_val and self.crit_val_alg == 'asc' and len(self.test_results) > 1:
            if not self.test_results[-1] and not self.test_results[-2] and not self.test_results[-3]:
                self.crit_val += 0.01
        self.test_results.append(test_stat < self.crit_val)
        return test_stat < self.crit_val

    def get_test_stat(self, train_dataset, train_labels):
        test_stat_h = self.part_test_stat_h
        test_stat_v = self.part_test_stat_v
        train_h_indeces = np.where(train_labels == 1)[0]
        train_v_indeces = np.where(train_labels == -1)[0]
        valid_h_indeces = np.where(self.valid_labels == 1)[0]
        valid_v_indeces = np.where(self.valid_labels == -1)[0]

        # harmless points
        for i in train_h_indeces:
            for j in valid_h_indeces:
                test_stat_h += np.linalg.norm(train_dataset[i, :] - self.valid_dataset[j, :]) / (
                        len(train_h_indeces) * len(valid_h_indeces))
        for i in train_h_indeces:
            for j in train_h_indeces:
                test_stat_h += np.linalg.norm(train_dataset[i, :] - train_dataset[j, :]) / (
                        2 * len(valid_h_indeces) ** 2)
        test_stat_h *= len(train_h_indeces) * len(valid_h_indeces) / (len(train_h_indeces) + len(valid_h_indeces))

        # virus points
        for i in train_v_indeces:
            for j in valid_v_indeces:
                test_stat_v += np.linalg.norm(train_dataset[i, :] - self.valid_dataset[j, :]) / (
                        len(train_v_indeces) * len(valid_v_indeces))
        for i in train_v_indeces:
            for j in train_v_indeces:
                test_stat_v += np.linalg.norm(train_dataset[i, :] - train_dataset[j, :]) / (
                        2 * len(valid_v_indeces) ** 2)
        test_stat_v *= len(train_v_indeces) * len(valid_v_indeces) / (len(train_v_indeces) + len(valid_v_indeces))

        return test_stat_h + test_stat_v

    def partial_fit(self, new_train_dataset, new_train_labels):
        self.train_dataset = np.append(self.train_dataset, new_train_dataset, axis=0)
        self.train_labels = np.append(self.train_labels, new_train_labels)
        self.svc = svm.SVC(kernel='linear', C=self.C).fit(self.train_dataset, self.train_labels)
        self.val_errors.append(1 - accuracy_score(self.valid_labels, self.svc.predict(self.valid_dataset)))
        if self.crit_val_alg == 'desc' and self.val_errors[len(self.val_errors) - 1] < self.val_errors[
            len(self.val_errors) - 2]:
            self.crit_val = self.crit_val * 0.95
            print('     decreasing crit_val to ', self.crit_val)
        elif self.crit_val_alg == 'asc' and self.val_errors[len(self.val_errors) - 1] > self.val_errors[
            len(self.val_errors) - 2]:
            self.crit_val = self.crit_val * 1.05
            print('     increasing crit_val to ', self.crit_val)

    def fit(self, train_dataset, train_labels):
        self.svc = svm.SVC(kernel='linear', C=self.C).fit(train_dataset, train_labels)
        self.val_errors.append(1 - accuracy_score(self.valid_labels, self.svc.predict(self.valid_dataset)))
        if self.crit_val_alg == 'desc' and self.val_errors[-1] < self.val_errors[-2]:
            self.crit_val -= .1 #*= * 0.9
            print('     decreasing crit_val to ', self.crit_val)
        elif self.crit_val_alg == 'asc' and self.val_errors[-1] > self.val_errors[-2]:
            self.crit_val += .01 #*= 1.1
            print('     increasing crit_val to ', self.crit_val)

