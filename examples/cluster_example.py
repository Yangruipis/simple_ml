# -*- coding:utf-8 -*-

from simple_ml.cluster import *
from simple_ml.classify_data import *
import matplotlib.pyplot as plt


def iris_example():
    x, y = get_iris()

    k_means = KMeans(k=4, dis_type=DisType.CosSim)
    k_means.fit(x[:, :2])
    print(k_means.labels)

    plt.scatter(x=x[:, 0], y=x[:, 1], c=k_means.labels)
    plt.show()

    h_cluster = Hierarchical(dis_type=DisType.Manhattan)
    h_cluster.fit(x[:, :2])
    # 选取距离为最大距离的四分之一
    h_cluster.cluster(h_cluster.max_dis/4)
    print(h_cluster.labels)
    plt.scatter(x=x[:, 0], y=x[:, 1], c=h_cluster.labels)
    plt.show()


if __name__ == '__main__':
    iris_example()
