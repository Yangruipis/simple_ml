# -*- coding:utf-8 -*-

from __future__ import division, absolute_import

from simple_ml.base.base_model import *
from simple_ml.base.base_enum import *
from simple_ml.base.base_error import *
import numpy as np
from collections import Counter
from simple_ml.evaluation import *


__all__ = [
    'ID3',
    'CART'
]


class ID3(BaseClassifier):

    __doc__ = "ID3 Decision Tree"

    def __init__(self, max_depth=None, min_samples_leaf=3):
        """
        决策树ID3算法
        :param max_depth:        树最大深度
        :param min_samples_leaf: 叶子节点最大样本数（最好是奇数，用以投票）
        """
        super(ID3, self).__init__()
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.min_samples_leaf = min_samples_leaf
        self._root = None

    @property
    def root(self):
        return self._root

    def fit(self, x, y):
        super(ID3, self).fit(x, y)
        if self.label_type != LabelType.binary and self.label_type != LabelType.multi_class:
            raise LabelTypeError("ID3算法只支持离散标签")
        if LabelType.continuous in self.feature_type:
            raise FeatureTypeError("ID3算法只支持离散特征")

        self._root = self._gen_tree(MultiTreeNode(data_id=np.arange(self.x.shape[0])), 0)

    def _gen_tree(self, node, depth):
        if depth >= self.max_depth or len(node.data_id) <= self.min_samples_leaf:
            node.leaf_label = np.argmax(np.bincount(self.y[node.data_id]))
            return node

        split_feature = self._get_best_split(node.data_id)
        feature = self.x[node.data_id, split_feature]
        nodes = []
        for value in np.unique(feature):
            new_node = MultiTreeNode(None, node.data_id[feature == value], split_feature, value)
            new_node = self._gen_tree(new_node, depth + 1)
            nodes.append(new_node)
        node.child = nodes
        return node

    def _get_best_split(self, data_id):
        data = self.x[data_id]
        y = self.y[data_id]
        best_split_feature = None
        y_entropy = self._get_entropy(y)
        _max_gain = -np.inf
        for i in range(data.shape[1]):
            unique = np.unique(data[:, i])
            if len(unique) <= 1:
                continue
            entropy = 0
            for feature_value in unique:
                y_temp = y[data[:, i] == feature_value]
                entropy += len(y_temp) / len(data_id) * self._get_entropy(y_temp)
            gain = y_entropy - entropy
            if gain > _max_gain:
                _max_gain = gain
                best_split_feature = i
        return best_split_feature

    @staticmethod
    def _get_entropy(arr):
        count = Counter(arr)
        s = 0
        for i in count:
            p = count[i] / len(arr)
            s += -p * np.log(p)
        return s

    def predict(self, x):
        if self._root is None:
            raise ModelNotFittedError
        super(ID3, self).predict(x)
        return np.array([self._predict_single(i, self._root) for i in x])

    def _predict_single(self, x, node):
        if node.leaf_label is not None:
            return node.leaf_label

        for child_node in node.child:
            feature_id = child_node.feature_id
            value = child_node.value
            if x[feature_id] == value:
                return self._predict_single(x, child_node)
        return np.random.choice(self.y, 1)

    def score(self, x, y):
        super(ID3, self).score(x, y)
        y_predict = self.predict(x)
        return classify_f1(y_predict, y)

    def classify_plot(self, x, y, title=""):
        classify_plot(self.new(), self.x, self.y, x, y, title=self.__doc__ + title)

    def new(self):
        return ID3(self.max_depth, self.min_samples_leaf)


class CART(BaseClassifier):

    __doc__ = "Classify and Regression Tree"

    def __init__(self, max_depth=10, min_samples_leaf=5):
        """
        分类回归树
        :param max_depth:        树最大深度
        :param min_samples_leaf: 叶子节点最大样本数（最好是奇数，用以投票）
        """
        super(CART, self).__init__()
        self._function = Function.cls_and_reg
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.min_samples_leaf = min_samples_leaf
        self._root = None

    @property
    def root(self):
        return self._root

    def fit(self, x, y):
        super(CART, self).fit(x, y)
        self._root = self._gen_tree(BinaryTreeNode(None, None, np.arange(self.x.shape[0])), 1)

    def _gen_tree(self, node, depth):
        if depth >= self.max_depth or len(node.data_id) <= self.min_samples_leaf:
            # 获取相应的标签
            if len(node.data_id) != 0:
                if self.label_type == LabelType.continuous:
                    node.leaf_label = np.mean(self.y[node.data_id])
                else:
                    node.leaf_label = np.argmax(np.bincount(self.y[node.data_id]))
            else:
                # 防止出现样本量为0时返回nan的情况，此时temp是一个数组
                temp = self.y[self.x[:, node.feature_id] == node.value]
                if len(temp) == 0:
                    node.leaf_label = np.random.choice(self.y, 1)[0]
                else:
                    node.leaf_label = np.argmax(np.bincount(temp))
            return node

        best_split = self._get_best_split(node.data_id)
        if best_split[0] is None:
            if self.label_type == LabelType.continuous:
                node.leaf_label = np.mean(self.y[node.data_id])
            else:
                node.leaf_label = np.argmax(np.bincount(self.y[node.data_id]))
            return node

        feature_arr = self.x[node.data_id, best_split[0]]
        if best_split[2]:
            left = BinaryTreeNode(None, None, node.data_id[feature_arr <= best_split[1]], best_split[0], best_split[1])
            right = BinaryTreeNode(None, None, node.data_id[feature_arr > best_split[1]], best_split[0], best_split[1])
        else:
            left = BinaryTreeNode(None, None, node.data_id[feature_arr == best_split[1]], best_split[0], best_split[1])
            right = BinaryTreeNode(None, None, node.data_id[feature_arr != best_split[1]], best_split[0], best_split[1])
        node.left = self._gen_tree(left, depth + 1)
        node.right = self._gen_tree(right, depth + 1)
        return node

    def _get_best_split(self, data_id):
        x = self.x[data_id]
        y = self.y[data_id]
        best_split = (None, None, None, None)     # (特征，取值，连续还是离散, error减小)
        error = np.inf
        if self.label_type == LabelType.continuous:
            y_error = self._get_sse(y)
            for i in range(x.shape[1]):
                feature_arr = x[:, i]
                if self.feature_type[i] == LabelType.continuous:
                    # 如果特征为连续型
                    low, high = min(feature_arr), max(feature_arr)
                    step = (high - low) / (self.sample_num // 4)
                    low += step
                    while low < high:
                        temp_error = self._get_sum_sse(y, feature_arr, low)
                        low += step
                        if temp_error < error:
                            error = temp_error
                            best_split = (i, low, True, y_error - error)
                else:
                    # 如果特征为离散值
                    for f in np.unique(feature_arr):
                        temp_error = self._get_sum_sse(y, feature_arr, f, False)
                        if temp_error < error:
                            error = temp_error
                            best_split = (i, f, False, y_error - error)
        else:
            y_error = self._get_gini(y)
            for i in range(x.shape[1]):
                feature_arr = x[:, i]
                if self.feature_type[i] == LabelType.continuous:
                    # 如果特征为连续型
                    low, high = min(feature_arr), max(feature_arr)
                    step = (high - low) / (self.sample_num // 2)
                    while low < high:
                        temp_error = self._get_conditional_gini(y, feature_arr, low)
                        low += step
                        if temp_error < error:
                            error = temp_error
                            best_split = (i, low, True, y_error - error)
                else:
                    # 如果特征为离散值
                    for f in np.unique(feature_arr):
                        temp_error = self._get_conditional_gini(y, feature_arr, f, False)
                        if temp_error < error:
                            error = temp_error
                            best_split = (i, f, False, y_error - error)
        return best_split

    def _get_conditional_gini(self, y, arr, value, continuous=True):
        if continuous:
            y_left = y[arr <= value]
            y_right = y[arr > value]
        else:
            y_left = y[arr == value]
            y_right = y[arr != value]
        entropy = self._get_gini(y_left) * len(y_left) + self._get_gini(y_right) * len(y_right)
        return entropy / len(arr)

    @staticmethod
    def _get_gini(arr):
        count = Counter(arr)
        s = 0
        for i in count:
            s += (count[i] / len(arr))**2
        return 1 - s

    def _get_sum_sse(self, y, arr, value, continuous=True):
        if continuous:
            y_left = y[arr <= value]
            y_right = y[arr > value]
        else:
            y_left = y[arr == value]
            y_right = y[arr != value]

        # 用样本数加权，防止样本越多损失越大
        return self._get_sse(y_left) + self._get_sse(y_right)

    @staticmethod
    def _get_sse(y):
        return np.sum(np.square(y - np.mean(y)))

    def predict(self, x):
        if self._root is None:
            raise ModelNotFittedError
        super(CART, self).predict(x)
        return np.array([self._predict_single(i, self._root) for i in x])

    def _predict_single(self, x, node):
        if node.leaf_label is not None:
            return node.leaf_label

        feature = node.left.feature_id
        if self.feature_type[feature] == LabelType.continuous:
            if x[feature] <= node.left.value:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)
        else:
            if x[feature] == node.left.value:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)

    def score(self, x, y):
        super(CART, self).score(x, y)
        y_predict = self.predict(x)
        if self.label_type == LabelType.continuous:
            return regression_r2(y_predict, y)
        elif self.label_type == LabelType.multi_class:
            return classify_f1_macro(y_predict, y)
        else:
            return classify_f1(y_predict, y)

    def classify_plot(self, x, y, title=""):
        classify_plot(self.new(), self.x, self.y, x, y, title=self.__doc__ + title)

    def new(self):
        return CART(self.max_depth, self.min_samples_leaf)
