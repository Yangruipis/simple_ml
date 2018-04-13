# -*- coding:utf-8 -*-

"""
注意：
- 该画图方法是在内部训练进行画图，如果特征大于2，则降至2维再进行训练，而不是先训练后作图，因为要对图上每一个二维点都进行预测
- 因此，模型必须支持2维训练集（比如随机森林 m>2 时就不支持2维训练集）
- 如果想先训练再作图，且特征大于2维，则无法做出区域
"""

import matplotlib.pyplot as plt
from simple_ml.base.base import BaseClassifier
from simple_ml.base.base_error import LabelTypeError
import numpy as np
from simple_ml.base.base_enum import LabelType


def transform_y(y):
    if list(np.unique(y)) == [-1, 1]:
        return np.array([0 if i == -1 else i for i in y])
    else:
        return y


def classify_plot(model: BaseClassifier, x_train, y_train, x_test, y_test, title="", px=200):
    feature_num = x_train.shape[1]
    # 检查特征数，如果大于2，则降维至2
    if feature_num == 1:
        raise Exception("特征数过少")
    elif feature_num > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    model.fit(x_train, y_train)
    # 先进行预测得分，在预测的同时检查是否有数据不匹配的问题
    score = model.score(x_test, y_test)

    if model.label_type == LabelType.continuous:
        raise LabelTypeError

    # 获取总样本的横轴纵轴范围
    x = np.row_stack((x_train, x_test))
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = ((x_max - x_min) / px + (y_max - y_min) / px) / 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = plt.cm.RdBu   # ListedColormap(['#FF0000', '#29A086', '#0000FF'])
    ax = plt.subplot(1, 2, 1)
    # Plot the training points
    ax.scatter(x_train[:, 0], x_train[:, 1], c=transform_y(y_train), cmap=cm_bright)
    # and testing points
    ax.scatter(x_test[:, 0], x_test[:, 1], c=transform_y(y_test), cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax = plt.subplot(1, 2, 2)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    z = transform_y(z)
    z = z.reshape(xx.shape)
    print(z)
    ax.contourf(xx, yy, z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(x_train[:, 0], x_train[:, 1], c=transform_y(model.y), cmap=cm_bright)
    # and testing points
    ax.scatter(x_test[:, 0], x_test[:, 1], c=transform_y(y_test), cmap=cm_bright, alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')

    plt.show()
