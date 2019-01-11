import numpy as np


class PreprocessData:
    def normalize_data(self, X, is_train):
        if is_train:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    def train_test_split(self, X, y, split=0.8):
        m = X.shape[0]
        train_size = int(split * m)
        return X[:train_size], y[:train_size], X[train_size:], y[train_size:]
