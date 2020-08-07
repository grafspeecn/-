import numpy as np
import matplotlib.pyplot as plt


class Myperceptron:
    def __init__(self):
        self.w = None
        self.b = 0
        self.l_rate = 0.2

    def fit(self, X_train, y_train):
        self.w = np.zeros(X_train.shape[1])
        i = 0
        while i < X_train.shape[0]:
            x = X_train[i]
            y = y_train[i]
            if y * (np.dot(self.w, x) + self.b) <= 0:
                self.w = self.w + self.l_rate * np.dot(x, y)
                self.b = self.b + self.l_rate * y
                i = 0
            else:
                i += 1


def draw(x, y, w, b):
    X_new = np.array([[0], [6]])
    y_predict = (-b - (w[0] * X_new)) / w[1]
    length = y.shape[0]
    for i in range(length):
        if y[i] == 1:
            plt.plot(x[i, 0], x[i, 1], 'bx')
        if y[i] == -1:
            plt.plot(x[i, 0], x[i, 1], 'rx')
    plt.plot(X_new, y_predict, 'g-')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def main():
    X_train = np.array([[3, 3], [4, 3], [3, 2], [6, 2], [3, 5], [2.9, 2.5]])
    y_train = np.array([1, 1, -1, -1, 1, -1])
    perceptron = Myperceptron()
    perceptron.fit(X_train, y_train)
    print('结束', perceptron.w)
    print(perceptron.b)
    draw(X_train, y_train, perceptron.w, perceptron.b)


if __name__ == '__main__':
    main()
