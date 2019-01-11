import numpy as np
import pandas as pd
import preprocessing.preprocess as pp


class LogisticRegression:
    def __init__(self, lr=0.01, _iter=20000, add_intercept=True, verbose=True):
        self.lr = lr
        self._iter = _iter
        self.verbose = verbose
        self.add_intercept = add_intercept

    def add_ones(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.add_intercept:
            X = self.add_ones(X)

        self.theta = np.zeros(X.shape[1])
        for i in range(self._iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            loss = h - y
            gradient = np.dot(X.T, loss) / y.size
            self.theta -= self.lr * gradient
            self.cost = self.loss(h, y)
            if (self.verbose and i % 10000 == 0):
                print('loss:', self.cost)

    def get_prob(self, X):
        if self.add_intercept:
            X = self.add_ones(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.get_prob(X) >= threshold

    def get_final_weights(self):
        return self.theta

    def get_cost(self):
        return self.cost


if __name__ == "__main__":
    model = LogisticRegression(lr=0.01, _iter=20000)
    path = "data/advertisement.csv"
    data = pd.read_csv(path)
    preprocess = pp.PreprocessData()
    # Normalize data
    data[['Age', 'EstimatedSalary']] = preprocess.normalize_data(data[['Age', 'EstimatedSalary']], True)
    # Convert male female to 1, 0
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    X = data.as_matrix(['Gender', 'Age', 'EstimatedSalary'])
    y = data["Purchased"].values

    # Split data in Training and testing dataset
    train_X, train_Y, test_X, test_Y = preprocess.train_test_split(X, y, 0.8)
    # Train the model
    model.fit(train_X, train_Y)
    # Make predictions
    preds = model.predict(test_X, threshold=0.5)
    accuracy = (preds == test_Y).mean()
    cost = model.get_cost()
    weights = model.get_final_weights()
    print("Accuracy : {0}, Cost : {1}, Final weights : {2}".format(accuracy, cost, weights))
