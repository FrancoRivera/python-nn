#!/usr/bin/env python3

import numpy as np  # for matrix operations
import pandas as pd # for reading the CSV

from matplotlib import pyplot as plt

# read into data
data = pd.read_csv("train.csv")


# make it a np array
data = np.array(data)

m, n = data.shape # get the shape

np.random.shuffle(data) # randomize the data before getting train and dev set

data_dev = data[0:1000].T # transposing data

Y_dev = data_dev[0] # labels for dev
X_dev = data_dev[1:n] # data for dev
X_dev = X_dev / 255 # regularizing data between 0 and 1 (as its a 255 grayscale)

data_train = data[100:m].T
Y_train = data_train[0] # labels for train
X_train = data_train[1:n] # data for train
X_train = X_train / 255 # regularizing data between 0 and 1 (as its a 255 grayscale)

_,m_train = X_train.shape

print(Y_train)

# Structure of NN is
# 784 => Layer 0 (input)
# 10 => Layer 1 (HL1)
# 10 => Layer 2 (HL2)
# 10 => Output layer

# layer 1 and layer 2 weights and biases (these are mtarixes)
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    W3 = np.random.rand(10, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2, W3, b3

def get_saved_params():
    W1 = np.array(pd.read_csv("params/W1.csv")).T[1:].T
    b1 = np.array(pd.read_csv("params/b1.csv")).T[1:].T

    W2 = np.array(pd.read_csv("params/W2.csv")).T[1:].T
    b2 = np.array(pd.read_csv("params/b2.csv")).T[1:].T

    W3 = np.array(pd.read_csv("params/W3.csv")).T[1:].T
    b3 = np.array(pd.read_csv("params/b3.csv")).T[1:].T

    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    # input layer
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    # Second layer
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)

    # out put layer
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)

    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3,  alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = get_saved_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3

W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.10, 100)

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


test_prediction(0, W1, b1, W2, b2, W3, b3)
test_prediction(1, W1, b1, W2, b2, W3, b3)
test_prediction(2, W1, b1, W2, b2, W3, b3)
test_prediction(3, W1, b1, W2, b2, W3, b3)


dev_predictions = make_predictions(X_dev, W1, b1, W2, b2, W3, b3)
print(get_accuracy(dev_predictions, Y_dev))
pd.DataFrame(W1).to_csv("params/W1.csv")
pd.DataFrame(b1).to_csv("params/b1.csv")
pd.DataFrame(W2).to_csv("params/W2.csv")
pd.DataFrame(b2).to_csv("params/b2.csv")
pd.DataFrame(W3).to_csv("params/W3.csv")
pd.DataFrame(b3).to_csv("params/b3.csv")
