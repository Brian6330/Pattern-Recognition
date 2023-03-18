#######################
#
# Name: Brian Schweigler
# Matriculation Number: 16-102-071
#
#######################
import random
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist


def kmeans(x: np.ndarray, k: int, metric="euclidean"):
    """

    :param x: The array representing the images.
    :param k: The number of clusters to create.
    :param metric: The metric to be used for the distance calculations.
    :return:
    """
    random.seed(6)  # for reproducibility
    cluster_centers = x[random.sample(range(len(x)), k)]
    prior_cluster_centers = np.zeros(cluster_centers.shape, dtype=float)

    while not np.array_equal(prior_cluster_centers, cluster_centers):
        prior_cluster_centers = cluster_centers
        dist = cdist(x, cluster_centers, metric)
        closest_medoid = np.argmin(dist, 1)
        # I tried with np.matrix and np.zeros, but always had trouble with the size. After googling for dict alternatives
        # I found the defaultdict, which seems to work, although it's not as clear as working with np.matrix or np.zeros
        cluster_set = defaultdict(list)
        for c, s in zip(closest_medoid, x):
            cluster_set[c].append(s)
        new_clusters = [np.array(cluster_set[c]) for c in range(k)]
        cluster_centers = np.array([np.mean(cluster, 0) for cluster in new_clusters])  # the new centers

    return new_clusters


def dunn_index(clusters: list, k: int) -> float:
    """
    The ratio of the smallest distance between observations not in the same cluster to the largest intra-cluster dist.

    :param clusters: The list of clusters
    :param k: The number of clusters

    :return: The dunn index
    """
    inter_cluster_dist = []
    for i in range(k - 1):  # for each cluster
        for j in range(i + 1, len(clusters)):  # iterate over all other clusters
            # calculate distance between cluster i and j using the provided metric
            inter_cluster_dist.append(np.min(cdist(clusters[i], clusters[j])))
    # Take the min of all inter-cluster-distances and divide it by the max of all intra-cluster-distances
    d_index = min(inter_cluster_dist) / max([np.max(pdist(cl)) for cl in clusters])  # the actual index
    return d_index


def davies_bouldin_index(clusters: list, k: int, metric="euclidean") -> float:
    """
    An internal evaluation scheme that measures the similarity between clusters.

    :param clusters: The list of clusters
    :param k: The number of clusters
    :param metric: The metric to be used for the distance calculations

    :return: The davis-bouldin index
    """
    m, d = [], []
    for i in range(k):  # for each cluster, calculate m and d as defined in the lecture
        m.append(np.sum(clusters[i], axis=0) / len(clusters[i]))  # the mean of the cluster
        d.append(np.sum(cdist(clusters[i], np.array([m[i]]), metric)) / len(clusters[i]))  # the mean dist to medoid(?)
    r = np.zeros((k, k))
    for i in range(k - 1):  # for each cluster
        for j in range(i + 1, k):  # iterate over all other clusters
            r[i, j] = (d[i] + d[j]) / (cdist(np.array([m[i]]), np.array([m[j]]), metric))  # the ratio
            r[j, i] = r[i, j]  # the matrix is symmetric
    d_index = np.sum(np.max(r, axis=1)) / k
    return d_index


def main():
    k_vals = [5, 7, 9, 10, 12, 15]  # Values from 5 to 15
    train = np.loadtxt('../data/mnist_small/mnist_small_knn/train.csv', delimiter=',')
    x_train = train[:, 1:train.shape[1]]

    # test = np.loadtxt('../data/mnist_small/mnist_small_knn/test.csv', delimiter=',')
    # x_test = test[:, 1:test.shape[1]]
    metrics = ["euclidean", "cityblock", "cosine"]
    for metric in metrics:
        for k in k_vals:
            clusters = kmeans(x_train, k, metric)
            dunn_idx = dunn_index(clusters, k)
            davies_bouldin_idx = davies_bouldin_index(clusters, k, metric)

            print("K = " + str(k) + "; Metric: " + metric + "; Dunn-Index: " + str(round(dunn_idx, 3)) +
                  "; Davis Bouldin Index: " + str(round(davies_bouldin_idx, 3)))


if __name__ == '__main__':
    main()
