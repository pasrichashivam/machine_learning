import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k):
        self.k = k

    def distance(self, p1, p2):
        return np.sum(np.square(p1 - p2))

    def assigne_centroid_to_points(self, X, mycentroids):
        length = X.shape[0]
        idxs = np.zeros((length, 1))
        for x in range(length):
            mypoint = X[x]
            mindist, idx = 9999999, 0
            for i in range(self.k):
                mycentroid = mycentroids[i]
                distsquared = self.distance(mycentroid, mypoint)
                if distsquared < mindist:
                    mindist = distsquared
                    idx = i
            idxs[x] = idx
        return idxs

    def move_centroids(self, X, myidxs):
        group_with_index = []
        for x in range(len(np.unique(myidxs))):
            group_with_index.append(np.array([X[i] for i in range(X.shape[0]) if myidxs[i] == x]))
        return np.array([np.mean(group, axis=0) for group in group_with_index])

    def plot_data(self, X, mycentroids, myidxs=None):
        colors = ['r', 'g', 'b']
        if myidxs is not None:
            subX = []
            for x in range(mycentroids[0].shape[0]):
                subX.append(np.array([X[i] for i in range(X.shape[0]) if myidxs[i] == x]))
        else:
            subX = [X]
        plt.figure(figsize=(7, 5))
        for x in range(len(subX)):
            newX = subX[x]
            plt.plot(newX[:, 0], newX[:, 1], 'o', color=colors[x], alpha=0.75)
        plt.show()

    def fit(self, myX, initial_centroids, K, _iter):
        centroid_history = []
        current_centroids = initial_centroids
        for i in range(_iter):
            centroid_history.append(current_centroids)
            idxs = self.assigne_centroid_to_points(myX, current_centroids)
            current_centroids = self.move_centroids(myX, idxs)

        return idxs, centroid_history


if __name__ == "__main__":
    k_mean = KMeans(k=3)
    data = 'data/clustering_data.csv'
    X = np.genfromtxt(data, delimiter=',')
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idxs, centroid_history = k_mean.fit(X, initial_centroids, K=3, _iter=100)
    k_mean.plot_data(X, centroid_history, idxs)
