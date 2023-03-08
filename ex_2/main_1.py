import numpy as np
import operator
from operator import itemgetter
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
    def __init__(self, K=3):
        self.K = K

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, metric):
        predictions = []
        for i in range(len(x_test)):
            match metric:
                case euclidean:
                    dist = np.array([euclidean(x_test[i], x_t) for x_t in
                                     self.x_train])
                case manhattan:
                    dist = np.array([manhattan(x_test[i], x_t) for x_t in
                                     self.x_train])
                case cosine:
                    dist = np.array([cosine(x_test[i], x_t) for x_t in
                                     self.x_train])
                case minkowski: # Use hard-coded 3 as p-value for minkowski
                    dist = np.array([minkowski(x_test[i], x_t, 3) for x_t in
                                     self.x_train])
            dist_sorted = dist.argsort()[:self.K]
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