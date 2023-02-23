#######################
#
# Name: Brian Schweigler
# Matriculation Number: 16-102-071
#
#######################

import numpy as np
import operator
from numpy.linalg import norm
from math import *
from decimal import Decimal


# L_2 Norm
def euclidean_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# L_1 Norm
def manhattan_dist(x, y):
    return np.abs(np.sum((x - y)))


def cosine_dist(x, y):
    return np.dot(x, y) / (norm(x) * norm(y))


# Preliminary function for minkowski dist
def p_root(value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) **
                 Decimal(root_value), 3)


# L_p Norm
def minkowski_dist(x, y, p):
    return (p_root(sum(pow(abs(a - b), p)
                       for a, b in zip(x, y)), p))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, metric):
        predictions = []
        for i in range(len(x_test)):
            if metric == "euclidean":
                dist = np.array([euclidean_dist(x_test[i], x_t) for x_t in
                                 self.x_train])
            elif metric == "manhattan":
                dist = np.array([manhattan_dist(x_test[i], x_t) for x_t in
                                 self.x_train])
            elif metric == "cosine":
                dist = np.array([cosine_dist(x_test[i], x_t) for x_t in
                                 self.x_train])
            elif metric == "minkowski":  # Use hard-coded 3 as p-value for minkowski
                dist = np.array([minkowski_dist(x_test[i], x_t, 3) for x_t in
                                 self.x_train])
            dist_sorted = dist.argsort()[:self.k]
            neigh_count = {}
            for idx in dist_sorted:
                if self.y_train[idx] in neigh_count:
                    neigh_count[self.y_train[idx]] += 1
                else:
                    neigh_count[self.y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(),
                                        key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0])
        return predictions


def accuracy(x, y):
    return np.sum(x == y) / len(x)


def main():
    k_vals = [1, 3, 5, 10, 15]  # Values from 1 to 15
    accuracies = []
    train = np.loadtxt('../data/mnist_small/mnist_small_knn/train.csv', delimiter=',')
    x_train = train[:, 1:train.shape[1]]
    y_train = train[:, 0]  # only first column
    test = np.loadtxt('../data/mnist_small/mnist_small_knn/test.csv', delimiter=',')
    x_test = test[:, 1:test.shape[1]]
    y_test = test[:, 0]  # only first column
    metrics = ["euclidean", "manhattan", "cosine", "minkowski"]
    for metric in metrics:
        for k in k_vals:
            model = KNN(k=k)
            model.fit(x_train, y_train)
            pred = model.predict(x_test, metric)
            acc = accuracy(y_test, pred)
            accuracies.append(acc)
            print("K = " + str(k) + "; Accuracy: " + str(acc) + "; Metric: " + metric)


if __name__ == '__main__':
    main()
