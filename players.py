import sklearn.svm as svm
import numpy as np
from scipy.optimize import minimize
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
    a = 1.0

    def __init__(self, initial_train_dataset, initial_train_labels, test_dataset, test_labels):
        self.train_dataset = initial_train_dataset
        self.train_labels = initial_train_labels
        self.test_dataset = test_dataset
        self.test_labels = test_labels
        self.current_w = np.zeros(len(initial_train_dataset[0]))
        self.current_b = 0.0

    def add_train_data(self, new_train_data, new_train_labels):
        self.train_dataset = np.append(self.train_dataset, new_train_data, axis=0)
        self.train_labels = np.append(self.train_labels, new_train_labels)

    def adv_obj(self, x, dataset, labels):
        n,m = np.shape(dataset)
        w = x[:m]
        b = x[m]
        h = np.reshape(x[m+1:], (n,m))
        ret = 0.0
        # classifier approximation of error on existing training dataset
        n_e = len(self.train_labels)
        for i in range(n_e):
            ret+=max(1-self.train_labels[i]*(np.dot(w, self.train_dataset[i,:])+b),0)/n_e

        # classifier approximation of error on new training points
        for i in range(n):
            ret+=max(1-labels[i]*(np.dot(w, dataset[i,:]+h[i,:])+b),0)/n

        # adversary approximation of error on test set
        n_t = len(self.test_labels)
        for i in range(n_t):
            ret += self.a*max(self.test_labels[i]*(np.dot(w, self.test_dataset[i, :])+b), -1)

    def adv_constr_ineq(self, x, n, m):
        h = np.reshape(x[m+1:], (n,m))
        return n*self.eps**2 - sum([np.dot(h[i, :], h[i, :]) for i in range(n)])

    def adv_constr_eq(self, x, n, m):
        h = np.reshape(x[m + 1:], (n, m))
        ret = []
        for i in range(n):
            ret.append(sum(h[i, :]))
        return np.array(ret)

    def get_attack(self, new_train_data, new_train_labels):
        n, m = np.shape(new_train_data)
        x0 = np.zeros(m+1+n*m)
        x0[:m] = self.current_w
        x0[m] = self.current_b
        con1 = {'type': 'ineq', 'fun': lambda x: self.adv_constr_ineq(x, n, m)}
        con2 = {'type': 'ineq', 'fun': lambda x: self.adv_constr_eq(x, n, m)}
        cons = [con1, con2]
        options = {'maxiter': 1000}
        sol = minimize(lambda x: self.adv_obj(x, new_train_data, new_train_labels),
                       x0, constraints=cons, options=options)
        return np.reshape(sol.x[m+1:], (n, m))



