import numpy as np
from memory_profiler import profile
import pyipopt
import datetime


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

    # @profile
    def get_infected_dataset(self, new_train_data, new_train_labels):
        n, m = np.shape(new_train_data)
        x0 = np.zeros(m + 1 + n * m)
        x0[:m] = self.current_w
        x0[m] = self.current_b
        if self.train_dataset is not None:
            n_e = len(self.train_labels)
        else:
            n_e = 0
        n_t = len(self.test_labels)

        def adv_obj(x):
            w = x[:m]
            b = x[m]
            h = np.reshape(x[m + 1:], (n, m))
            ret = 0.0
            # classifier approximation of error on existing training dataset
            for i in range(n_e):
                ret += max(1 - self.train_labels[i] * (np.dot(w, self.train_dataset[i, :]) + b), 0) / n_e

            # classifier approximation of error on new training points
            #for i in range(n):
            #    ret += max(1 - new_train_labels[i] * (np.dot(w, new_train_data[i, :] + h[i, :]) + b), 0) / n

            # adversary approximation of error on test set
            for i in range(n_t):
                ret += self.a * max(self.test_labels[i] * (np.dot(w, self.test_dataset[i, :]) + b), -1) / n_t
            return ret

        def adv_obj_grad(x):
            # with respect to w:
            ret = []
            h = np.reshape(x[m + 1:], (n, m))
            for j in range(m):
                # classifier approximation of error on existing training dataset
                if self.train_dataset is not None:
                    tmp = sum([-1 * self.train_labels[i] * self.train_dataset[i, j] *
                               (1.0 if self.train_labels[i] * (
                                       np.dot(x[:m], self.train_dataset[i, :]) + x[m]) < 1.0 else 0.0)
                               for i in range(n_e)]) / n_e
                else:
                    tmp = 0
                # classifier approximation of error on new training points
                #tmp += sum([-1 * new_train_labels[i] * (new_train_data[i, j] + h[i, j]) *
                #            (1.0 if new_train_labels[i] * (
                #                    np.dot(x[:m], new_train_data[i, :] + h[i, :]) + x[m]) < 1.0 else 0.0)
                #            for i in range(n)]) / n
                # adversary approximation of error on test set
                tmp += self.a * sum([self.test_labels[i] * self.test_dataset[i][j] *
                                     (1.0 if self.test_labels[i] * (
                                             np.dot(x[:m], self.test_dataset[i]) + x[m]) > -1.0 else 0.0)
                                     for i in range(0, n_t)]) / n_t
                ret.append(tmp)
            # with respect to b:
            if self.train_dataset is not None:
                tmp = sum([-1 * self.train_labels[i] *
                           (1.0 if self.train_labels[i] * (
                                   np.dot(x[:m], self.train_dataset[i, :]) + x[m]) < 1.0 else 0.0)
                           for i in range(n_e)]) / n_e
            else:
                tmp = 0
            ret.append(tmp + sum([-1 * new_train_labels[i] *
                                  (1.0 if new_train_labels[i] * (np.dot(x[:m],
                                                                        new_train_data[i,
                                                                        :] + h[i, :]) + x[
                                                                     m]) < 1.0 else 0.0)
                                  for i in range(n)]) / n + self.a * sum(
                [self.test_labels[i] *
                 (1.0 if self.test_labels[i] * (np.dot(x[:m], self.test_dataset[i, :]) + x[m]) > -1.0 else 0.0)
                 for i in range(0, n_t)]) / n_t)
            # with respect to h:
            der_h = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    der_h[i, j] = -1 * new_train_labels[i] * x[j] * (1.0 if new_train_labels[i] * (
                            np.dot(x[:m], new_train_data[i, :] + h[i, :]) + x[m]) < 1.0 else 0.0) / n
            return np.append(np.array(ret), np.reshape(der_h, (-1)))

        nvar = m + 1 + n * m  # number of variables
        ncon = n + 1  # number of constraints
        nnzj = n * m * 2  # number of nonzero elements in Jacobian of constraints function
        nnzh = 0  # number of nonzero elements in Hessian of Lagrangian

        def adv_constr(x):
            h = np.reshape(x[m + 1:], (n, m))
            ret = np.zeros(n + 1)
            ret[0] = n * self.eps ** 2 - np.dot(x[m + 1:], x[m + 1:])
            for i in range(n):
                ret[i + 1] = sum(h[i, :])
            return np.array(ret)

        def adv_constr_jac(x):
            ret = np.zeros((n + 1, m + 1 + n * m))
            # gradient of norm constraint
            for i in range(m + 1, n * m + m + 1):
                ret[0, i] = -2 * x[i]
            # jacobian of affine constraints:
            for i in range(n):
                der_h = np.zeros((n, m))
                der_h[i, :] = 1.0
                ret[i + 1, m + 1:] = np.reshape(der_h, (-1))
            return ret

        def eval_jac_g(x, flag):
            if flag:
                i_s = []
                j_s = []
                for i in range(m + 1, n * m + m + 1):
                    i_s.append(0)
                    j_s.append(i)
                for i in range(n):
                    for j in range(m):
                        i_s.append(i + 1)
                        j_s.append(m + 1 + i * m + j)
                # for i in range(ncon):
                #    for j in range(nvar):
                #        i_s.append(i)
                #        j_s.append(j)
                return np.array(i_s), np.array(j_s)
            else:
                jac = adv_constr_jac(x)
                assert np.shape(jac) == (ncon, nvar)
                ret = []
                for i in range(m + 1, n * m + m + 1):
                    ret.append(jac[0, i])
                for i in range(n):
                    for j in range(m):
                        ret.append(jac[i + 1, m + 1 + i * m + j])
                # for i in range(ncon):
                #    for j in range(nvar):
                #        ret.append(jac[i, j])
                return np.array(ret)

        x_L = np.zeros(nvar)
        x_U = np.zeros(nvar)
        x_L[:m + 1] = -100.0
        x_L[m + 1:] = -1 * self.eps * n
        x_U[:m + 1] = 100.0
        x_U[m + 1:] = self.eps * n
        g_L = np.zeros(ncon)
        g_U = np.zeros(ncon)
        g_U[0] = n * self.eps  # figure something out

        nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, adv_obj,
                             adv_obj_grad, adv_constr, eval_jac_g)
        nlp.str_option("derivative_test", "none")

        nlp.num_option('derivative_test_tol', 1e-2)
        nlp.num_option('tol', 1e-3)
        nlp.num_option('acceptable_constr_viol_tol', 0.1)

        nlp.int_option('print_level', 5)
        nlp.int_option('max_iter', 200)
        nlp.int_option('print_frequency_iter', 10)

        print(datetime.datetime.now(), ": Going to call solve")
        x_opt, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
        nlp.close()
        print('status: ', status)

        return np.reshape(x_opt[m + 1:], (n, m)) + new_train_data, np.sqrt(np.dot(x_opt[m + 1:], x_opt[m + 1:]) / n)
