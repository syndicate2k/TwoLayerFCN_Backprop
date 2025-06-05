import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.weights) + self.bias
        return self.z

    def backward(self, dz, learning_rate):
        m = self.x.shape[0]
        self.dw = np.dot(self.x.T, dz) / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m

        self.weights -= learning_rate * self.dw
        self.bias -= learning_rate * self.db

        return np.dot(dz, self.weights.T)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.dense1 = Dense(input_size, hidden_size)
        self.dense2 = Dense(hidden_size, output_size)

    def forward(self, X):
        z1 = self.dense1.forward(X)
        a1 = relu(z1)

        z2 = self.dense2.forward(a1)
        a2 = softmax(z2)

        return a2

    def backward(self, X, Y, learning_rate):
        m = X.shape[0]

        a2 = self.forward(X)
        dz2 = (a2 - Y) / m

        da1 = self.dense2.backward(dz2, learning_rate)

        dz1 = da1 * relu_derivative(self.dense1.z)

        self.dense1.backward(dz1, learning_rate)

    def compute_loss(self, Y, output):
        return categorical_crossentropy(Y, output)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def save_weights(self, path):
        np.savez(path, W1=self.dense1.weights, b1=self.dense1.bias,
                 W2=self.dense2.weights, b2=self.dense2.bias)

    @staticmethod
    def load_weights(path):
        data = np.load(path)
        return data['W1'], data['b1'], data['W2'], data['b2']
