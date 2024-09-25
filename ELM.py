import numpy as np
from utils import *

class ELM:
    def __init__(self, X, y, hidden_dim):
        self.X = X #X_train
        self.y = y #y_train (one_hot_encoded)
        self.input_dim = self.X.shape[1]
        self.hidden_dim = hidden_dim
        self.W = np.random.randn(self.input_dim, self.hidden_dim)
        
    def input_hidden(self, X): #compute output of input -> hidden layer
        return ReLU(np.dot(X, self.W))
    
    def train(self):
        X_ = self.input_hidden(self.X)
        self.β = np.dot(np.linalg.pinv(X_), self.y)
    
    def predict(self, X):
        X_ = self.input_hidden(X)
        X_ = np.dot(X_, self.β)
        pred = np.argmax(X_, axis = 1)
        return pred

    def get_accuracy(self, X_train, X_test, y_train, y_test):
        # y (non one_hot_encoded)
        train_pred = self.predict(X_train)
        test_pred = self.predict(X_test)
        train_acc = np.sum(train_pred == y_train)/X_train.shape[0]
        test_acc = np.sum(test_pred == y_test)/X_test.shape[0]
        return train_acc, test_acc