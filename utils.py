import numpy as np

def MNIST_load():
    from keras.datasets import mnist
    (X, y), (_, _ )= mnist.load_data()
    X = X.reshape(-1, 784)
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int32)
    return X, y

def one_hot_encode(y):
    num_classes = np.unique(y).shape[0]
    y_one_hot = np.zeros((y.shape[0], num_classes), dtype=int)
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot

def train_test_split(X, y, split_ratio = 0.7):
    N = X.shape[0]
    rnd_idx = np.random.permutation(N)
    train_idx = rnd_idx[:int(split_ratio * X.shape[0])]
    test_idx = rnd_idx[int(split_ratio * X.shape[0]):]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    return X_train, y_train, X_test, y_test

def ReLU(x):
    return np.maximum(0, x)