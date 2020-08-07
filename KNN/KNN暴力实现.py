import numpy as np
import time


class KNN:
    def __init__(self, k, p, x_train, y_train):
        self.k = k
        self.p = p
        self.x_train = x_train
        self.y_train = y_train

    def distense_sort(self, x_new):
        dist_list = [(np.linalg.norm(x_new - self.x_train[i], ord=self.p), self.y_train[i])
                     for i in range(len(self.x_train))]
        dist_list.sort(key=lambda i: i[0])
        return dist_list

    def define(self, x_new):
        x = 0
        d_list = self.distense_sort(x_new)
        for i in range(self.k):
            x += d_list[i][1]
        if x >= 0:
            y = 1
        else:
            y = -1
        return y


def main():
    t0 = time.time()
    # 训练数据
    x_train = np.array([[5, 4],
                        [9, 6],
                        [4, 7],
                        [2, 3],
                        [8, 1],
                        [7, 2]])
    y_train = np.array([1, 1, 1, -1, -1, -1])
    # 测试数据
    x_new = np.array([[5, 3], [9, 2]])
    for k in [5]:
        clf = KNN(x_train=x_train, y_train=y_train, k=k, p=2)
        for x in x_new:
            y_predict = clf.define(x)
            print("点{}当k={}时,被分类为：{}".format(x, k, y_predict))
    print("用时:{}s".format(round(time.time() - t0), 2))


if __name__ == "__main__":
    main()
