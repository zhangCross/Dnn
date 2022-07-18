import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

N = 100  # number of points per class
D = 2  # dimension
K = 3  # number of classes
num_examples = N * K
reg = 0.001
step_size = 0.1
steps = 50000
h = 100


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


def forward_softmax(X, y, W, b):
    scores = np.dot(X, W) + b
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss
    return loss, probs


def forward_dnn(X, y, W1, W2, b1, b2):
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    return loss, probs, hidden_layer


def backward_softmax(probs, W, X):
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg * W
    return dW, db


def backward_dnn(probs, W1, W2, hidden_layer):
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden_layer <= 0] = 0
    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)
    dW2 += reg * W2
    dW1 += reg * W1
    return dW1, db1, dW2, db2


def predict_softmax(X, W, b):
    scores = np.dot(X, W) + b
    y_p = np.argmax(scores, axis=1)
    return y_p


def eval_softmax(X, y, W, b):
    y_p = predict_softmax(X, W, b)
    print('training accuracy: %.2f' % (np.mean(y_p == y)))


def predict_dnn(X, W1, b1, W2, b2):
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2
    y_p = np.argmax(scores, axis=1)
    return y_p


def eval_dnn(X, y, W1, b1, W2, b2):
    y_p = predict_dnn(X, W1, b1, W2, b2)
    print('training accuracy: %.2f' % (np.mean(y_p == y)))


def demo_softmax(X, y):
    W = 0.01 * np.random.randn(D, K)
    b = np.zeros((1, K))
    for i in range(steps):
        loss, probs = forward_softmax(X, y, W, b)
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))
        dW, db = backward_softmax(probs, W, X)
        W += -step_size * dW
        b += -step_size * db
    eval_softmax(X, y, W, b)
    pred_func = partial(predict_softmax, W=W, b=b)
    plot_decision_boundary(pred_func, X, y, "softmax")


def demo_dnn(X, y):
    W1 = 0.001 * np.random.randn(D, h)
    b1 = np.zeros((1, h))
    W2 = 0.001 * np.random.randn(h, K)
    b2 = np.zeros((1, K))
    for i in range(steps):
        loss, probs, hidden_layer = forward_dnn(X, y, W1, W2, b1, b2)
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))
        dW1, db1, dW2, db2 = backward_dnn(probs, W1, W2, hidden_layer)
        W1 += -step_size * dW1
        b1 += -step_size * db1
        W2 += -step_size * dW2
        b2 += -step_size * db2
    eval_dnn(X, y, W1, b1, W2, b2)
    pred_func = partial(predict_dnn, W1=W1, b1=b1, W2=W2, b2=b2)
    plot_decision_boundary(pred_func, X, y, "dnn")


def plot_decision_boundary(pred_func, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    X, y = generate_data()
    demo_softmax(X, y)
    demo_dnn(X, y)

