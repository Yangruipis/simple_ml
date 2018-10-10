# -*- coding:utf-8 -*-

import numpy as np
# from sklearn.cluster import KMeans

class Kmeans:

    def __init__(self, k, c=0.0):
        self.k = k
        self.c = c
        self.X = None
        self.labels = None
        self.centers = None

    def fit(self, X):
        self.X = X
        self.labels, self.centers = self._fit()

    def transform(self):
        for i, x in enumerate(self.X):
            label = int(self.labels[i])
            center = self.centers[label]
            self.X[i] = self.fill(x, center)
        return self.X

    def fill(self, x, y):
        # 确保y没有缺失值
        assert not np.any(np.isnan(y))
        for i in range(len(x)):
            if np.isnan(x[i]):
                x[i] = y[i]
        return x

    def get_distance(self, a, b):
        a, b = a[(np.logical_not(np.isnan(a))) & (np.logical_not(np.isnan(b)))], \
               b[(np.logical_not(np.isnan(a))) & (np.logical_not(np.isnan(b)))]
        dis = np.square(np.sum(np.square(a-b))) + self.c * len(a)
        if np.isnan(dis):
            return 0
        else:
            return dis

    def get_new_center(self, label):
        centers = np.zeros((self.k, self.X.shape[1]))
        for i in range(self.k):
            centers[i] = np.nanmean(self.X[label == i])
        return centers

    def get_init_center(self):
        idx = np.random.randint(self.X.shape[0])
        centers = [self.X[idx]]
        while len(centers) < self.k:
            center = np.nanmean(np.array(centers), axis=0)
            _max = (0, -1)
            for i, x in enumerate(self.X):
                dis = self.get_distance(x, center)
                if dis > _max[0]:
                    _max = (dis, i)
            centers.append(self.X[_max[1]])
        return np.array(centers)

    def _fit(self):
        init_center = self.get_init_center()
        labels = np.zeros(self.X.shape[0])

        while 1:
            for idx, x in enumerate(self.X):
                _min = (1e5, -1)
                for i, center in enumerate(init_center):
                    dis = self.get_distance(x, center)
                    if dis < _min[0]:
                        _min = (dis, i)
                labels[idx] = _min[1]
            new_center = self.get_new_center(labels)
            if np.array_equal(new_center, init_center):
                break
            init_center = new_center
        return labels, init_center


# if __name__ == '__main__':
#     from simple_ml.classify_data import get_iris
#     X, y = get_iris()
#     np.random.seed(918)
#     n = 50
#     a = np.random.choice(np.arange(X.shape[0]), n, False)
#     b = np.random.choice(np.arange(X.shape[1]), n, True)
#     for i in range(n):
#         X[a[i], b[i]] = np.nan
#     km = Kmeans(3)
#     km.fit(X)
#     print(km.labels)
#     new_X = km.transform()
#     km2 = Kmeans(3)
#     km2.fit(new_X)
#     print(km2.labels)


