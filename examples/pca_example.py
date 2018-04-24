# -*- coding:utf-8 -*-

from simple_ml.pca import *
from simple_ml.classify_data import *
import matplotlib.pyplot as plt
import numpy as np


def iris_example():
    x, y = get_iris()
    pca = PCA(2)
    new_x = pca.fit_transform(x)
    print(new_x.shape)
    print(pca.explain_ratio)
    ax1 = plt.subplot(1, 2, 2)
    ax1.scatter(new_x[:, 0], new_x[:, 1], c=y)
    ax1.set_title("After PCA")
    ax2 = plt.subplot(1, 2, 1)
    ax2.scatter(x[:, 0], x[:, 1], c=y)
    ax2.set_title("Before PCA")
    plt.show()


def super_PCA_example():
    x = np.random.rand(101, 100)
    pca = SuperPCA(2)
    new_x = pca.fit_transform(x)
    print(new_x.shape)
    print(pca.explain_ratio)


if __name__ == '__main__':
    # iris_example()
    super_PCA_example()