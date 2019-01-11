import numpy as np
from utils import sigmoid, sigmoid_backward, relu, relu_backward, load_dataset
import matplotlib.pyplot as plt


# Build a deeper neural network (with more than 1 hidden layer)
class DeepNeuralNetwork:
    def __init__(self, layers, learning_rate=0.2, num_iterations=1000, print_cost=True):
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.learning_rate = learning_rate
        self.layers = layers

    # Initialize the parameters for a two-layer network and for an $L$-layer neural network.
    def initialize_parameters(self, layer_dims):
        parameters = {}
        L = len(layer_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        return parameters

    def forward_propagation(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            Z = np.dot(W, A_prev) + b
            A = sigmoid(Z)
            cache = ((A_prev, W, b), Z)
            return A, cache
        elif activation == "relu":
            Z = np.dot(W, A_prev) + b
            A = relu(Z)
            cache = ((A_prev, W, b), Z)
            return A, cache

    # L-Layer Model
    # Implement the linear activation foreward with relu from 1 -> L-1 times for layer
    # Then Implement with Sigmoid activation for layer L.
    def l_layer_forward(self, A, parameters):
        caches = []
        L = len(parameters) // 2
        for l in range(1, L):
            A_prev = A
            A, cache = self.forward_propagation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                                activation='relu')
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.forward_propagation(A, parameters['W' + str(L)], parameters['b' + str(L)],
                                             activation='sigmoid')
        caches.append(cache)
        return AL, caches

    # Cost function
    def compute_cost(self, AL, Y):
        cost = - (np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))).mean()
        cost = np.squeeze(cost)
        return cost

    # To perform linear backward for layer l
    # We need to compute (dA[l-1], dW[l], db[l])
    # To compute (dA[l-1]) We need dZ[l], & W[l]
    # To compute (dW[l]) We need A[l-1] (A_prev), & dZ[l]
    # To compute (db[l]) We need dZ[l]
    # A[l-1] & W comes from linear cache we stored in linear foreward step.
    def linear_backward(self, dZ, linear_cache):
        A_prev, W, _ = linear_cache
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    # To perform linear activation backward for layer l i.e to calculate dZ[l]
    # i.e implement the backpropagation for the LINEAR->ACTIVATION layer.
    # We can compute dZ[l] from dA[l] using activation gradient.
    # To compute dZ[l] We need dA[l], and Z[l] ()
    # dA[l] = dA, & Z[l] = Z_cache  = Z which computed in linear forward step.
    # We need to know which backward activation to be appled (relu or sigmoid)
    #  dZ[l] = dA[l] ∗ g′(Z[l])
    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, Z = cache
        if activation == 'relu':
            dZ = relu_backward(dA, Z)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, Z)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def l_layer_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        # Derivative of classification loss function with respect to AL (cross entropy)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L - 1]  # Cache for Sigmoid Layer only
        # linear_cache, Z = current_cache
        dA_prev, grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache,
                                                                                              'sigmoid')
        for i in reversed(range(L - 1)):  # L = 4  this will output [2,1,0]
            # linear_cache, Z = current_cache
            current_cache = caches[i]  # Cache for Relu Layers in backward direction
            dA, dW, db = self.linear_activation_backward(dA_prev, current_cache, 'relu')
            dA_prev = dA
            grads["dW" + str(i + 1)] = dW
            grads["db" + str(i + 1)] = db
        return grads

    def update_parameters(self, parameters, grads):
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l + 1)] -= self.learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] -= self.learning_rate * grads["db" + str(l + 1)]
        return parameters

    def l_layer_network(self, X, Y):
        costs = []
        # Parameters initialization.
        parameters = self.initialize_parameters(self.layers)
        for i in range(self.num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            # Here length of caches == L
            AL, caches = self.l_layer_forward(X, parameters)
            # Compute cost.
            cost = self.compute_cost(AL, Y)
            # Backward propagation.
            grads = self.l_layer_backward(AL, Y, caches)
            # Update parameters.
            parameters = self.update_parameters(parameters, grads)
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                costs.append(cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("alpha =" + str(self.learning_rate))
        plt.show()
        return parameters


if __name__ == "__main__":
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Normalize colord pixels
    train_x = train_x_flatten / 255.  # (12288, 209)
    test_x = test_x_flatten / 255.  # (12288, 50)
    layers = [12288, 20, 7, 5, 1]
    model = DeepNeuralNetwork(layers, learning_rate=0.005, num_iterations=2000, print_cost=True)
    parameters = model.l_layer_network(train_x, train_y)
