import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class AnomayDetection:
    def __init__(self, epsilon=0.0008, use_multivariate=True):
        self.use_multivariate = use_multivariate
        self.epsilon = epsilon

    def gaussian(self, dataset, mu, si):
        p = multivariate_normal(mean=mu, cov=si)
        return p.pdf(dataset)

    def plot_data(self, X, outliers=None):
        plt.plot(X[:, 0], X[:, 1], 'b+')
        if len(outliers):
            plt.plot(X[outliers, 0], X[outliers, 1], 'ro')
        plt.show()

    def custum_gaussian(self, myX, mymu, mysig2):
        m = myX.shape[0]
        n = myX.shape[1]
        if np.ndim(mysig2) == 1:
            mysig2 = np.diag(mysig2)

        norm = 1. / (np.power((2 * np.pi), n / 2) * np.sqrt(np.linalg.det(mysig2)))
        myinv = np.linalg.inv(mysig2)
        myexp = np.zeros((m, 1))
        for irow in range(m):
            xrow = myX[irow]
            myexp[irow] = np.exp(-0.5 * ((xrow - mymu).T).dot(myinv).dot(xrow - mymu))
        return norm * myexp

    def get_gaussian_params(self, dataset):
        m = dataset.shape[0]
        mean = np.mean(dataset, axis=0)
        if self.use_multivariate:
            variance = ((dataset - mean).T.dot(dataset - mean)) / float(m)
            return mean, variance
        else:
            variance = np.sum(np.square(dataset - mean), axis=0) / float(m)
            return mean, variance

    def find_anomalies(self, X):
        mean, variance = self.get_gaussian_params(X)
        prob = self.gaussian(X, mean, variance)
        outliers = np.asarray(np.where(prob < self.epsilon))
        return outliers


if __name__ == "__main__":
    data = 'data/cluster_data.csv'
    X = np.genfromtxt(data, delimiter=',')
    anomalies = AnomayDetection(epsilon=0.0008, use_multivariate=True)
    outliers = anomalies.find_anomalies(X)
    anomalies.plot_data(X, outliers=outliers)
