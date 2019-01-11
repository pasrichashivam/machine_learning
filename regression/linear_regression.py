import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr=0.001, _iter=20000, add_intercept=True, verbose=True):
        self.lr = lr
        self._iter = _iter
        self.verbose = verbose
        self.add_intercept = add_intercept

    def add_ones(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def loss(self, h, y):
        return np.sum((h - y) ** 2) / (2 * len(y))

    def fit(self, X, y):
        if self.add_intercept:
            X = self.add_ones(X)

        self.theta = np.zeros(X.shape[1])
        for i in range(self._iter):
            h = np.dot(X, self.theta)
            loss = h - y
            gradient = np.dot(X.T, loss) / len(y)
            self.theta -= self.lr * gradient
            self.cost = self.loss(h, y)
            if (self.verbose and i % 10000 == 0):
                print('loss:', self.cost)

    def get_final_weights(self):
        return self.theta

    def get_cost(self):
        return self.cost

    def predict(self, X):
        if self.add_intercept:
            X = self.add_ones(X)
        return X.dot(self.theta)

    # RMSE
    def rmse(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(sum((y - y_pred) ** 2) / len(y))

    # R2 Score
    def r2_score(self, X, y):
        pred = self.predict(X)
        mean_y = np.mean(y)
        ss_tot = sum((y - mean_y) ** 2)
        ss_res = sum((y - pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


if __name__ == "__main__":
    model = LinearRegression(lr=0.0001, _iter=20000)
    path = "data/student.csv"
    data = pd.read_csv(path)
    X = data.as_matrix(['Math'])
    y = data["Writing"].values
    model.fit(X, y)
    params = model.get_final_weights()
    # Y = M*X + C
    line = params[1] * X + params[0]
    plt.plot(X, y, 'o', X, line)
    plt.show()
