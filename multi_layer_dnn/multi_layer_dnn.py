import time
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    A = np.maximum(0, Z)
    return A


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- array containing the dimensions of each layer

    Returns:
    parameters -- dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_activation_forward(A_prev, W, b, activation="relu"):
    """
    Implement the forward propagation for one layer

    Arguments:
    A_prev -- previous activation
    W -- weights matrix
    b -- bias vector
    activation -- the activation func to be used in this layer

    Returns:
    A -- post activation
    cache -- tuple of values (linear_cache, activation_cache)
    """

    Z, linear_cache = (np.dot(W, A_prev) + b, (A_prev, W, b))
    activation_cache = Z
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation

    Arguments:
    X -- input of network
    parameters -- W and b

    Returns:
    AL -- output of network
    caches -- list of caches, each is from linear_activation_forward()
    """

    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    """
    Arguments:
    AL -- output of network
    Y -- ground truth

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = -(np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T)) / m
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for one layer

    Arguments:
    dZ -- Gradient of the linear output
    cache -- linear_cache (A_prev, W, b)

    Returns:
    dA_prev -- Gradient of the previous activation 
    dW -- Gradient of W
    db -- Gradient of b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation="relu"):
    """
    Implement the backward propagation for one layer.

    Arguments:
    dA -- Gradient of post activation
    cache -- tuple of values (linear_cache, activation_cache)

    Returns:
    dA_prev -- Gradient of previous activation
    dW -- Gradient of W
    db -- Gradient of b
    """
    linear_cache, activation_cache = cache
    Z = activation_cache
    if activation == "relu":
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
        dZ = dA * A * (1-A)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation

    Arguments:
    AL -- output of network
    Y -- ground truth
    caches -- list of caches, each is from linear_activation_forward()

    Returns:
    grads -- A dictionary of gradients
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dA = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    for l in reversed(range(L)):
        current_cache = caches[l]
        activation = "relu"
        if l == L-1:
            activation = "sigmoid"
        dA, dW_temp, db_temp = linear_activation_backward(dA, current_cache, activation)
        dA, dW_temp, db_temp = np.clip(dA, -5, 5), np.clip(dW_temp, -5, 5), np.clip(db_temp, -5, 5)
        grads["dA" + str(l)] = dA
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- dictionary containing parameters
    grads -- A dictionary of gradients

    Returns:
    parameters -- dictionary containing updated parameters
    """
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters


def predict(X, y, parameters):
    """
    predict results of network.

    Arguments:
    X -- input of network
    parameters -- parameters of network

    Returns:
    p -- predictions of network
    """
    m = X.shape[1]
    p = np.zeros((1, m))
    probas, _ = L_model_forward(X, parameters)
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("Accuracy: " + str(np.sum((p == y) / m)))
    return p


def print_mislabeled(classes, X, y, p):
    """
    Plots mislabeled samples.
    X -- input
    y -- ground truth
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_samples = len(mislabeled_indices[0])
    for i in range(num_samples):
        index = mislabeled_indices[1][i]
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))


def L_layer_model(X, Y, layer_dims, learning_rate=0.001, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network

    Arguments:
    X -- input of network
    Y -- ground truth
    layer_dims -- array containing the dimensions of each layer
    print_cost -- if True, it prints the cost

    Returns:
    parameters -- parameters learnt by the model
    """
    costs = []
    parameters = initialize_parameters_deep(layer_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 500 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 500 == 0 or i == num_iterations:
            costs.append(cost)
    return parameters, costs

N = 200  # number of points per class
D = 2  # dimension
K = 2  # number of classes
num_examples = N * K

def generate_data():
    X = np.zeros((num_examples, D))
    y = np.zeros(num_examples, dtype='uint8')
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    # X, y = make_circles(n_samples=N*K, noise=0.1)
    # X, y = make_blobs(n_samples=N*K, centers=K)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title("samples")
    plt.show()
    return X, y


if __name__ == '__main__':
    layers_dims = [2, 3, 3, 2, 1]
    X, y = generate_data()
    X = X.T
    y = y.reshape((1, -1))
    parameters, costs = L_layer_model(X, y, layers_dims, learning_rate=0.005, num_iterations=20000, print_cost=True)
    predict(X, y, parameters)