import sklearn.svm as svm
import numpy as np
from sklearn.metrics import accuracy_score


class Classifier:
    crit_val = 10

    def __init__(self, train_dataset, train_labels, valid_dataset, valid_labels, C):
        self.svc = svm.SVC(kernel='linear', C=C).fit(train_dataset, train_labels)
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels
        self.C = C
        # part of test statistics
        self.part_test_stat_h = 0.0
        h_indeces = np.where(valid_labels == 1)
        v_indeces = np.where(valid_labels == -1)
        self.part_test_stat_v = 0.0
        for i in h_indeces:
            for j in h_indeces:
                self.part_test_stat_h += np.linalg.norm(valid_dataset[i,:]-valid_dataset[j,:])
        self.part_test_stat_h /= 2*(len(h_indeces)**2)
        for i in v_indeces:
            for j in v_indeces:
                self.part_test_stat_v += np.linalg.norm(valid_dataset[i,:]-valid_dataset[j,:])
        self.part_test_stat_v /= 2*(len(v_indeces)**2)

    def predict(self, test_dataset):
        return self.svc.predict(test_dataset)

    def get_error(self, test_dataset, test_labels):
        pred_labels = self.predict(test_dataset)
        return 1 - accuracy_score(test_labels, pred_labels)

    def is_valid(self, train_dataset, train_labels):
        test_stat_h = self.part_test_stat_h
        test_stat_v = self.part_test_stat_v
        train_h_indeces = np.where(train_labels ==1)
        train_v_indeces = np.where(train_labels ==-1)
        valid_h_indeces = np.where(self.valid_labels ==1)
        valid_v_indeces = np.where(self.valid_labels ==-1)

        # harmless points
        for i in train_h_indeces:
            for j in valid_h_indeces:
                test_stat_h += np.linalg.norm(train_dataset[i,:]-self.valid_dataset[j,:])/(len(train_h_indeces)*len(valid_h_indeces))
        for i in train_h_indeces:
            for j in train_h_indeces:
                test_stat_h += np.linalg.norm(train_dataset[i,:]-train_dataset[j,:])/(2*len(valid_h_indeces)**2)
        test_stat_h *= len(train_h_indeces)*len(valid_h_indeces)/(len(train_h_indeces)+len(valid_h_indeces))

        # virus points
        for i in train_v_indeces:
            for j in valid_v_indeces:
                test_stat_v += np.linalg.norm(train_dataset[i,:]-self.valid_dataset[j,:])/(len(train_v_indeces)*len(valid_v_indeces))
        for i in train_v_indeces:
            for j in train_v_indeces:
                test_stat_v += np.linalg.norm(train_dataset[i,:]-train_dataset[j,:])/(2*len(valid_v_indeces)**2)
        test_stat_v *= len(train_v_indeces)*len(valid_v_indeces)/(len(train_v_indeces)+len(valid_v_indeces))

        return test_stat_h+test_stat_v > self.crit_val

    def partial_fit(self, new_train_dataset, new_train_labels):
        self.train_dataset = np.append(self.train_dataset, new_train_dataset, axis=0)
        self.train_labels = np.append(self.train_labels, new_train_labels)
        self.svc = svm.SVC(kernel='linear', C=self.C).fit(self.train_dataset, self.train_labels)


class Adversary:
    eps = 10




