import numpy as np
import matplotlib.pyplot as plt
from utils import plot_decision_boundary


# One hidden layer and output layer.
class NeuralNetwork:
    def __init__(self, learning_rate=0.2, num_iterations=1000, print_cost=True):
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def layer_sizes(self, X, Y):
        n_x = X.shape[0]  # size of input layer
        n_h = 4  # size of hidden layer
        n_y = Y.shape[0]  # size of output layer
        return (n_x, n_h, n_y)

    def initialize_parameters(self, n_x, n_h, n_y):
        np.random.seed(2)
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros(shape=(n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros(shape=(n_y, 1))
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def forward_propagation(self, X, parameters):
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def compute_cost(self, A2, Y):
        cost = -(np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))).mean()
        cost = np.squeeze(cost)
        return cost

    def backward_propagation(self, parameters, cache, X, Y):
        m = X.shape[1]
        # First, retrieve W1 and W2 from the dictionary "parameters".
        W2 = parameters['W2']
        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache['A1']
        A2 = cache['A2']
        # Backward propagation: calculate dW1, db1, dW2, db2.
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def update_parameters(self, parameters, grads):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        W1 = W1 - self.learning_rate * dW1
        b1 = b1 - self.learning_rate * db1
        W2 = W2 - self.learning_rate * dW2
        b2 = b2 - self.learning_rate * db2
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def nn_model(self, X, Y, n_h):
        np.random.seed(3)
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        for i in range(0, self.num_iterations):
            A2, cache = self.forward_propagation(X, parameters)
            cost = self.compute_cost(A2, Y)
            grads = self.backward_propagation(parameters, cache, X, Y)
            parameters = self.update_parameters(parameters, grads)
            if self.print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        return parameters

    def predict(self, parameters, X):
        A2, _ = self.forward_propagation(X, parameters)
        predictions = np.round(A2)
        return predictions


if __name__ == "__main__":
    data = 'data/data.csv'
    data = np.genfromtxt(data, delimiter=',')

    X, Y = data[:, :-1], data[:, -1]
    plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), s=40)
    plt.show()
    X, Y = X.T, Y.reshape((1, 400))
    model = NeuralNetwork(learning_rate=1.2, num_iterations=10000, print_cost=True)

    parameters = model.nn_model(X, Y, n_h=4)
    print("Accuracy : ", (model.predict(parameters, X) == Y).mean())
    plot_decision_boundary(lambda x: model.predict(parameters, x.T), X, Y)
