# -*- coding:utf-8 -*-

"""
模型评价

ref:
    - http://scikit-learn.org/stable/modules/model_evaluation.html
    - http://blog.csdn.net/u012856866/article/details/

针对二分类问题：
    - accuracy
    - precision
    - recall
    - f1
针对多分类问题：
    - f1_micro
    - f1_macro
    - f1_weight
针对回归问题：
    - explained_variance
    - absolute_error
    - squared_error
    - RMSE(root mean squared error)
    - RMSLE(root mean squared log error, in case of the abnormal value)
    - r2
    - median_absolute_error

"""
from __future__ import division, absolute_import

from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from simple_ml.base.base_model import BaseClassifier
from simple_ml.base.base_enum import *
from simple_ml.base.base_error import *
from simple_ml.data_handle import transform_y, train_test_split, get_k_folder_idx
from simple_ml.pca import PCA

__all__ = [
    'classify_f1',
    'classify_accuracy',
    'classify_f1_macro',
    'classify_plot',
    'classify_auc',
    'classify_f1_micro',
    'classify_f1_weighted',
    'classify_precision',
    'classify_recall',
    'classify_roc',
    'classify_roc_plot',

    'regression_r2',
    'regression_absolute_error',
    'regression_explained_variance',
    'regression_median_absolute_error',
    'regression_rmse',
    'regression_rmsel',
    'regression_squared_error',
    'regression_plot',

    'cross_validation',
    'CrossValidationType'
]


def _check_input(y_predict, y_true):
    if len(y_predict.shape) != 1 or len(y_true.shape)!= 1:
        raise InputTypeError("函数输入必须是一维数组")
    if len(y_predict) != len(y_true):
        raise LabelLengthMismatchError


def _get_binary_confusion_matrix(y_predict, y_true):
    """
    返回混淆矩阵
    :param y_predict:
    :param y_true:
    :return:
    预测\实际 |   1   |    0   |
         1   |       |        |
         0   |       |        |
    """
    _check_input(y_predict, y_true)
    y_unique = np.unique(np.append(y_predict, y_true))
    if np.array_equal(y_unique, np.array([0, 1])) or \
        np.array_equal(y_unique, np.array([0])) or \
            np.array_equal(y_unique, np.array([1])):
        pass
    else:
        raise ParamInputError("混淆矩阵必须输入二分类标签")

    joint = list(zip(y_predict, y_true))
    joint_counted = dict(Counter(joint))
    # if len(joint_counted) < 4:
    #     raise ValueError
    matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    for i in [0, 1]:
        for j in [0, 1]:
            if (i, j) in joint_counted:
                matrix[1 - i][1 - j] = joint_counted[(i, j)]
    return matrix
    # return [[joint_counted[(1,1)],\
    #          joint_counted[(1,0)]],[\
    #          joint_counted[(0,1)],\
    #          joint_counted[(0,0)]]]


def classify_accuracy(y_predict, y_true):
    _check_input(y_predict, y_true)
    con_matrix = _get_binary_confusion_matrix(y_predict, y_true)
    right_case = con_matrix[0][0] + con_matrix[1][1]
    total_case = con_matrix.sum()
    return right_case / total_case


def classify_precision(y_predict, y_true):
    _check_input(y_predict, y_true)
    con_matrix = _get_binary_confusion_matrix(y_predict, y_true)
    tp = con_matrix[0][0]
    tp_plus_fp = tp + con_matrix[0][1]  # 存伪
    if tp_plus_fp == 0:
        return 0
    return tp / tp_plus_fp


def classify_recall(y_predict, y_true):
    _check_input(y_predict, y_true)
    con_matrix = _get_binary_confusion_matrix(y_predict, y_true)
    tp = con_matrix[0][0]
    tp_plus_fn = tp + con_matrix[1][0]  # 弃真
    if tp_plus_fn == 0:
        return 0
    return tp / tp_plus_fn


def classify_f1(y_predict, y_true):
    """
    避免precision和recall一个为1，另一个为0的极端情况
    """
    _check_input(y_predict, y_true)
    con_matrix = _get_binary_confusion_matrix(y_predict, y_true)
    tp = con_matrix[0][0]
    tp_plus_fp = tp + con_matrix[0][1]  # 存伪
    tp_plus_fn = tp + con_matrix[1][0]  # 弃真
    precision = 0
    recall = 0
    if tp_plus_fn != 0:
        precision = tp / tp_plus_fn
    if tp_plus_fp != 0:
        recall = tp / tp_plus_fp
    if precision + recall == 0:
        return 0
    return 2 * recall * precision / (recall + precision)


def _gen_binary_pairs(y_predict, y_true):
    """
    对于多分类的情况，每个类生成一组0-1的(y_predict, y_true)pair
    """
    _check_input(y_predict, y_true)
    label_set = set(y_predict) | set(y_true)
    for label in label_set:
        yield np.array([1 if i == label else 0 for i in y_predict]), np.array([1 if i == label else 0 for i in y_true])


def classify_f1_micro(y_predict, y_true):
    """
    - 针对多分类的情况
    - 在求recall和precision时，在计算前进行加总（recall = sum(tp_i) / (sum(tp_i+fn_i))
    """
    _check_input(y_predict, y_true)
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for y_pred_binary, y_true_binary in _gen_binary_pairs(y_predict, y_true):
        con_matrix = _get_binary_confusion_matrix(y_pred_binary, y_true_binary)
        tp_sum += con_matrix[0][0]
        fp_sum += con_matrix[0][1]
        fn_sum += con_matrix[1][0]
    recall = tp_sum / (tp_sum + fn_sum)
    precision = tp_sum / (tp_sum + fp_sum)
    return 2 * recall * precision / (recall + precision)


def classify_f1_macro(y_predict, y_true):
    """
    - 针对多分类情况
    - 先求每一个类的recall和precision以及f1，然后求平均
    """
    _check_input(y_predict, y_true)
    f1_list = []
    for y_pred_binary, y_true_binary in _gen_binary_pairs(y_predict, y_true):
        con_matrix = _get_binary_confusion_matrix(y_pred_binary, y_true_binary)
        tp = con_matrix[0][0]
        recall = tp / (tp + con_matrix[1][0])
        precision = tp / (tp + con_matrix[0][1])
        f1_list.append(2 * recall * precision / (recall + precision))
    return np.mean(f1_list)


def classify_f1_weighted(y_predict, y_true):
    """
    - 针对多分类情况
    - 用样本中正例数目加权（避免了样本不平衡的问题）
    """
    _check_input(y_predict, y_true)
    f1_list = []
    weight_list = []
    for y_pred_binary, y_true_binary in _gen_binary_pairs(y_predict, y_true):
        con_matrix = _get_binary_confusion_matrix(y_pred_binary, y_true_binary)
        tp = con_matrix[0][0]
        recall = tp / (tp + con_matrix[1][0])
        precision = tp / (tp + con_matrix[0][1])
        f1_list.append(2 * recall * precision / (recall + precision))
        weight_list.append(sum(y_true_binary))
    temp = 0
    for i, f1 in enumerate(f1_list):
        temp += f1*weight_list[i]
    return temp / sum(weight_list)


def classify_roc(y_predict, y_true):
    """
    当输出y_pred为概率时可用，如logistic
    此时，y_pred为连续值，y_true为二分类值
    """
    _check_input(y_predict, y_true)

    if not isinstance(y_predict[0], float):
        raise LabelTypeError("ROC curve only support probability output")

    pair = zip(y_predict, y_true)
    pair = sorted(pair, key=lambda x: x[0], reverse=True)

    def get_tpr_and_fpr(threshold):
        y_pred_bin = [1 if i >= threshold else 0 for i in y_predict]
        con_matrix = _get_binary_confusion_matrix(y_pred_bin, y_true)
        _tpr = con_matrix[0][0]/(con_matrix[0][0] + con_matrix[1][0])
        _fpr = con_matrix[0][1]/(con_matrix[0][1] + con_matrix[1][1])
        return _tpr, _fpr

    tpr_list = []
    fpr_list = []
    for i in pair:
        tpr, fpr = get_tpr_and_fpr(i[0])
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return np.array(tpr_list), np.array(fpr_list), sorted(y_predict, reverse=True)


def classify_auc(y_predict, y_true):
    """
    直接计算roc下方面积太麻烦，可以利用其物理意义进行计算，复杂度为O(N)N为总样本数
    ref: https://www.cnblogs.com/gatherstars/p/6084696.html
    假设有正样本M个，负样本N个
    - 首先对y_pred排序
    - 对于排序最大的正样本，假设其排序为rank_1，那么比他score小的正样本有M-1个, 则比他小的负样本有(rank_1-1) - (M-1)个
    - 对于排序为rank_i 的样本， 比他score小的正样本有M-i个，那么比他小的负样本有(rank_i - 1) - (M-i)
    - 当 i = M时， 比其小的负样本有rank_i-1-(M-M) = rank_i - 1个
    - 总共有M*N个负样本对，所以得到auc求和公式: auc = \frac{\sum_{正样本i}^M rank_i - M*(M+1)/2}{M*N}
    """
    _check_input(y_predict, y_true)
    length = len(y_true)
    pair = zip(y_predict, y_true)
    pair = sorted(pair, key=lambda x: x[0])
    rank_pair = np.column_stack((np.arange(1, length+1), pair))
    positive_pair = rank_pair[rank_pair[:, 2] == 1]
    positive_count = positive_pair.shape[0]
    return (np.sum(positive_pair[:, 0]) - positive_count * (positive_count + 1) / 2) / \
           (positive_count * (length - positive_count))


def classify_roc_plot(y_predict, y_true):
    _check_input(y_predict, y_true)
    tpr, fpr, _ = classify_roc(y_predict, y_true)
    auc = classify_auc(y_predict, y_true)
    plt.plot([0, 1], [0, 1], '--')
    plt.plot(fpr, tpr, 'k--', label='Mean ROC (area = %0.2f)' % auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


def regression_explained_variance(y_predict, y_true):
    """
    SST = SSR + SSE
    SST = sum(y - \bar{y})^2
    SSE = sum(y - \hat{y})^2
    SSR = sum(\hat{y} - \bar{y})^2
    """
    _check_input(y_predict, y_true)
    y_hat = np.mean(y_true)
    sum_list = list(map(lambda x: (y_hat - x)**2, y_predict))
    return np.sum(sum_list)


def regression_absolute_error(y_predict, y_true):
    _check_input(y_predict, y_true)
    error_list = list(map(lambda x, y: np.abs(x - y), y_predict, y_true))
    return np.sum(error_list)


def regression_squared_error(y_predict, y_true):
    _check_input(y_predict, y_true)
    error_list = list(map(lambda x, y: (x-y)**2, y_predict, y_true))
    return np.sum(error_list)


def regression_rmse(y_predict, y_true):
    _check_input(y_predict, y_true)
    error_list = list(map(lambda x, y: (x-y)**2, y_predict, y_true))
    return np.sqrt(np.mean(error_list))


def regression_rmsel(y_predict, y_true):
    _check_input(y_predict, y_true)
    error_list = list(map(lambda x, y: (np.log(x+1)-np.log(y+1))**2, y_predict, y_true))
    return np.sqrt(np.mean(error_list))


def regression_r2(y_predict, y_true):
    _check_input(y_predict, y_true)
    y_bar = np.mean(y_true)
    sse = np.sum(np.square(y_predict - y_true))
    ssr = np.sum(list(map(lambda x: (y_bar - x)**2, y_predict)))
    return ssr/(ssr+sse)


def regression_median_absolute_error(y_predict, y_true):
    _check_input(y_predict, y_true)
    error_list = list(map(lambda x, y: np.abs(x-y), y_predict, y_true))
    return np.median(error_list)


def classify_plot(model, x_train, y_train, x_test, y_test, title="",compare=False, px=100):
    """
    注意：
    - 该画图方法是在内部训练进行画图，如果特征大于2，则降至2维再进行训练，而不是先训练后作图，因为要对图上每一个二维点都进行预测
    - 因此，模型必须支持2维训练集（比如随机森林 m>2 时就不支持2维训练集）
    - 如果想先训练再作图，且特征大于2维，则无法做出区域
    """
    feature_num = x_train.shape[1]
    # 检查特征数，如果大于2，则降维至2
    if feature_num == 1:
        raise Exception("特征数过少")
    elif feature_num > 2:
        pca = PCA(2)
        x_train = pca.fit_transform(x_train)
        if x_test.shape[1] + 1 == x_train.shape[1]:
            x_test = np.column_stack((np.ones(x_test.shape[0]), x_test))

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
    # cm_bright = plt.cm.RdBu   # ListedColormap(['#FF0000', '#29A086', '#0000FF'])
    colors = ['#67001F', '#053061', '#29A086', '#0000FF']
    if compare:
        ax = plt.subplot(1, 2, 1)

        for idx, i in enumerate(np.unique(y_train)):
            ax.scatter(x_train[y_train == i, 0], x_train[y_train == i, 1], c=colors[idx], label="train, y=%s" % int(i))

        for idx, i in enumerate(np.unique(y_test)):
            ax.scatter(x_test[y_test == i, 0], x_test[y_test == i, 1], c=colors[idx], label="test , y=%s" % int(i),
                       alpha=0.6)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)

        # p1 = ax.scatter(x_train[:, 0],x_train[:, 1],c=transform_y(y_train),cmap=cm_bright,label=transform_y(y_train))
        # p2 = ax.scatter(x_test[:, 0], x_test[:, 1], c=transform_y(y_test), cmap=cm_bright, alpha=0.6)
        # ax.legend([p1, p2], ['train', 'test'])

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
        ax.contourf(xx, yy, z, cmap=cm, alpha=.8)

        for idx, i in enumerate(np.unique(y_train)):
            ax.scatter(x_train[y_train == i, 0], x_train[y_train == i, 1], c=colors[idx], label="train, y=%s" % int(i))

        for idx, i in enumerate(np.unique(y_test)):
            ax.scatter(x_test[y_test == i, 0], x_test[y_test == i, 1], c=colors[idx], label="test , y=%s" % int(i),
                       alpha=0.6)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)    # , mode="expand", borderaxespad=0.)

        # p3 = ax.scatter(x_train[:, 0], x_train[:, 1], c=transform_y(model.y), cmap=cm_bright)
        # p4 = ax.scatter(x_test[:, 0], x_test[:, 1], c=transform_y(y_test), cmap=cm_bright, alpha=0.6)
        # ax.legend([p3, p4], ['train', 'test'])

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.4f' % score).lstrip('0'),
                size=15, horizontalalignment='right')

        plt.show()
    else:
        figure = plt.figure(figsize=(3, 6))
        ax = figure.add_subplot(111)
        z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        z = transform_y(z)
        z = z.reshape(xx.shape)
        ax.contourf(xx, yy, z, cmap=cm, alpha=.8)

        for idx, i in enumerate(np.unique(y_train)):
            ax.scatter(x_train[y_train == i, 0], x_train[y_train == i, 1], c=colors[idx], label="train, y=%s" % int(i))

        for idx, i in enumerate(np.unique(y_test)):
            ax.scatter(x_test[y_test == i, 0], x_test[y_test == i, 1], c=colors[idx], label="test , y=%s" % int(i),
                       alpha=0.6)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)  # , mode="expand", borderaxespad=0.)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.4f' % score).lstrip('0'),
                size=15, horizontalalignment='right')

        plt.show()


def cross_validation(model, x, y, method=CrossValidationType.holdout, test_size=0.3, cv=5, seed=918):
    """
    交叉验证函数
    :param model:         模型，继承predict和score方法
    :param x:             特征
    :param y:             标签
    :param method:        交叉验证方法
    :param test_size:     训练集占比，仅对holdout方法有用
    :param cv:            交叉验证次数，如果是k_folder法，则k=cv
    :param seed:          随机种子
    :return:
    """
    if not isinstance(x, np.ndarray):
        raise FeatureTypeError

    if x.shape[0] != len(y):
        raise SampleNumberMismatchError

    result = np.zeros(cv)
    if method == CrossValidationType.holdout:
        for i in range(cv):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size)
            model.fit(x_train, y_train)
            result[i] = model.score(x_test, y_test)
        return result
    elif method == CrossValidationType.k_folder:
        for i, (test_ids, train_ids)in enumerate(get_k_folder_idx(x.shape[0], cv, seed)):
            x_test, y_test = x[test_ids], y[test_ids]
            x_train, y_train = x[train_ids], y[train_ids]
            model.fit(x_train, y_train)
            result[i] = model.score(x_test, y_test)
        return result
    else:
        raise CrossValidationTypeError


def regression_plot(x_train, y_train, x_test, y_test_true, y_test_predict, x_column_id=None, title=""):
    if x_train.shape[1] != x_test.shape[1]:
        raise FeatureNumberMismatchError
    if x_train.shape[0] != y_train.shape[0] or x_test.shape[0] != y_test_true.shape[0]:
        raise SampleNumberMismatchError
    if y_test_true.shape != y_test_predict.shape:
        raise SampleNumberMismatchError

    figure = plt.figure(figsize=(6, 6))
    ax = figure.add_subplot(111)
    if x_column_id is None:
        x_column_train = np.arange(x_train.shape[0])
        x_column_test = np.arange(x_train.shape[0], x_train.shape[0]+x_test.shape[0])
    else:
        if x_column_id < 0:
            raise ParamInputError('x_column_id must be greater than 0')
        if x_column_id >= x_train.shape[1]:
            raise ParamInputError('x_column_id out of bound')
        x_column_train = x_train[:, x_column_id]
        x_column_test = x_test[:, x_column_id]
    colors = ['#67001F', '#053061', '#29A086', '#0000FF']
    ax.scatter(x=x_column_train, y=y_train, c=colors[0], label='train', alpha=0.8)
    ax.scatter(x=x_column_test, y=y_test_true, c=colors[1], label='test true', alpha=0.8)
    ax.scatter(x=x_column_test, y=y_test_predict, c=colors[1], label='test predict', alpha=0.5)
    ax.legend()
    ax.set_title(title)
    plt.show()
