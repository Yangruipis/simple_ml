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

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from simple_ml.base.base_error import LabelLengthMismatchError, LabelTypeError


def _check_input(y_predict, y_true):
    if len(y_predict) != len(y_true):
        raise LabelLengthMismatchError


def _get_binary_confusion_matrix(y_predict, y_true):
    _check_input(y_predict, y_true)
    joint = list(zip(y_predict, y_true))
    joint_counted = dict(Counter(joint))
    if len(joint_counted) <= 4:
        raise ValueError
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
    return tp / tp_plus_fp


def classify_recall(y_predict, y_true):
    _check_input(y_predict, y_true)
    con_matrix = _get_binary_confusion_matrix(y_predict, y_true)
    tp = con_matrix[0][0]
    tp_plus_fn = tp + con_matrix[1][0]  # 弃真
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
    precision = tp / tp_plus_fn
    recall = tp / tp_plus_fp
    return 2 * recall * precision / (recall + precision)


def _gen_binary_pairs(y_predict, y_true):
    """
    对于多分类的情况，每个类生成一组0-1的(y_predict, y_true)pair
    """
    _check_input(y_predict, y_true)
    label_set = set(y_predict) | set(y_true)
    for label in label_set:
        yield [1 if i == label else 0 for i in y_predict], [1 if i == label else 0 for i in y_true]


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

    if isinstance(y_predict[0], float):
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
    return tpr_list, fpr_list


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
    tpr, fpr = classify_roc(y_predict, y_true)
    auc = classify_auc(y_predict, y_true)
    plt.plot([0, 1], [0, 1], '--')
    plt.plot(fpr, tpr, 'k--', label='Mean ROC (area = %0.2f)' % auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
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
    y_hat = np.mean(y_true)
    sst = np.sum(list(map(lambda x: (x-y_hat)**2, y_true)))
    ssr = np.sum(list(map(lambda x: (y_hat-x)**2, y_predict)))
    return ssr/sst


def regression_median_absolute_error(y_predict, y_true):
    _check_input(y_predict, y_true)
    error_list = list(map(lambda x, y: np.abs(x-y), y_predict, y_true))
    return np.median(error_list)
