# -*- coding:utf-8 -*-

from collections import Counter
from simple_ml.base.base_error import *
from simple_ml.base.base import BaseClassifier
from simple_ml.base.base_enum import LabelType
from simple_ml.helper import classify_plot
from simple_ml.score import *


class BinaryTreeNode:

    def __init__(self, left=None, right=None, dataId=None, featureId=None, value=None, leaf_label=None):
        if dataId is None:
            dataId = []
        self.left = left
        self.right = right
        self.dataId = dataId
        self.featureID = featureId
        self.value = value
        self.leaf_label = leaf_label


class MultiTreeNode:

    def __init__(self, child=None, dataId=None, featureId=None, value=None, leaf_label=None):
        if dataId is None:
            dataId = []
        self.child = child
        self.dataId = dataId
        self.featureID = featureId
        self.value = value
        self.leaf_label = leaf_label


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
        self.root = None

    def fit(self, x, y):
        self._init(x, y)
        if self.label_type != LabelType.binary and self.label_type != LabelType.multi_class:
            raise LabelTypeError("ID3算法只支持离散标签")
        if LabelType.continuous in self.feature_type:
            raise FeatureTypeError("ID3算法只支持离散特征")

        self.root = self._gen_tree(MultiTreeNode(dataId=np.arange(self.x.shape[0])), 0)

    def _gen_tree(self, node: MultiTreeNode, depth) -> MultiTreeNode:
        if depth >= self.max_depth or len(node.dataId) <= self.min_samples_leaf:
            node.leaf_label = np.argmax(np.bincount(self.y[node.dataId]))
            return node

        split_feature = self._get_best_split(node.dataId)
        feature = self.x[node.dataId, split_feature]
        nodes = []
        for value in np.unique(feature):
            new_node = MultiTreeNode(None, node.dataId[feature == value], split_feature, value)
            new_node = self._gen_tree(new_node, depth + 1)
            nodes.append(new_node)
        node.child = nodes
        return node

    def _get_best_split(self, dataID):
        data = self.x[dataID]
        y = self.y[dataID]
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
                entropy += len(y_temp) / len(dataID) * self._get_entropy(y_temp)
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
        if self.root is None:
            raise ModelNotFittedError
        return np.array([self._predict_single(i, self.root) for i in x])

    def _predict_single(self, x, node: MultiTreeNode):
        if node.leaf_label is not None:
            return node.leaf_label

        for child_node in node.child:
            feature_id = child_node.featureID
            value = child_node.value
            if x[feature_id] == value:
                return self._predict_single(x, child_node)
        return np.random.choice(self.y, 1)

    def score(self, x, y):
        y_predict = self.predict(x)
        return classify_f1(y_predict, y)


class CART(BaseClassifier):

    __doc__ = "Classify and Regression Tree"

    def __init__(self, max_depth=10, min_samples_leaf=3):
        """
        分类回归树
        :param max_depth:        树最大深度
        :param min_samples_leaf: 叶子节点最大样本数（最好是奇数，用以投票）
        """
        super(CART, self).__init__()
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, x, y):
        self._init(x, y)
        self.root = self._gen_tree(BinaryTreeNode(None, None, np.arange(self.x.shape[0])), 1)

    def _gen_tree(self, node: BinaryTreeNode, depth) -> BinaryTreeNode:
        if depth >= self.max_depth or len(node.dataId) <= self.min_samples_leaf:
            # 获取相应的标签
            if len(node.dataId) != 0 :
                if self.label_type == LabelType.continuous:
                    node.leaf_label = np.mean(self.y[node.dataId])
                else:
                    node.leaf_label = np.argmax(np.bincount(self.y[node.dataId]))
            else:
                # 防止出现样本量为0时返回nan的情况，此时temp是一个数组
                temp = self.y[self.x[:, node.featureID] == node.value]
                if len(temp) == 0:
                    node.leaf_label = np.random.choice(self.y, 1)[0]
                else:
                    node.leaf_label = np.argmax(np.bincount(temp))
            return node

        best_split = self._get_best_split(node.dataId)
        if best_split[0] is None:
            if self.label_type == LabelType.continuous:
                node.leaf_label = np.mean(self.y[node.dataId])
            else:
                node.leaf_label = np.argmax(np.bincount(self.y[node.dataId]))
            return node

        feature_arr = self.x[node.dataId, best_split[0]]
        if best_split[2]:
            left = BinaryTreeNode(None, None, node.dataId[feature_arr <= best_split[1]], best_split[0], best_split[1])
            right = BinaryTreeNode(None, None, node.dataId[feature_arr > best_split[1]], best_split[0], best_split[1])
        else:
            left = BinaryTreeNode(None, None, node.dataId[feature_arr == best_split[1]], best_split[0], best_split[1])
            right = BinaryTreeNode(None, None, node.dataId[feature_arr != best_split[1]], best_split[0], best_split[1])
        node.left = self._gen_tree(left, depth + 1)
        node.right = self._gen_tree(right, depth + 1)
        return node

    def _get_best_split(self, dataID):
        x = self.x[dataID]
        y = self.y[dataID]
        best_split = (None, None, None)     # (特征，取值，连续还是离散)
        error = np.inf
        if self.label_type == LabelType.continuous:
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
                            best_split = (i, low, True)
                else:
                    # 如果特征为离散值
                    for f in np.unique(feature_arr):
                        temp_error = self._get_sum_sse(y, feature_arr, f, False)
                        if temp_error < error:
                            error = temp_error
                            best_split = (i, f, False)
        else:
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
                            best_split = (i, low, True)
                else:
                    # 如果特征为离散值
                    for f in np.unique(feature_arr):
                        temp_error = self._get_conditional_gini(y, feature_arr, f, False)
                        if temp_error < error:
                            error = temp_error
                            best_split = (i, f, False)
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
        return  1 - s

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
        if self.root is None:
            raise ModelNotFittedError
        try:
            return np.array([self._predict_single(i, self.root) for i in x])
        except:
            pass

    def _predict_single(self, x, node: BinaryTreeNode):
        if node.leaf_label is not None:
            if isinstance(node.leaf_label, np.ndarray) and len(node.leaf_label) == 0:
                return np.random.choice(self.y, 1)[0]
            return node.leaf_label

        feature = node.left.featureID
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
        y_predict = self.predict(x)
        if self.label_type == LabelType.continuous:
            return regression_r2(y_predict, y)
        elif self.label_type == LabelType.multi_class:
            return classify_f1_macro(y_predict, y)
        else:
            return classify_f1(y_predict, y)

    def classify_plot(self, x, y):
        classify_plot(self, self.x, self.y, x, y, title=self.__doc__)


class BaseRandomForest(BaseClassifier):

    __doc__ = "Random Forest"

    def __init__(self, m, tree_num=200):
        super(BaseRandomForest, self).__init__()
        self.m = m
        self.tree_num = tree_num
        self.forest = None

    def fit(self, x, y):
        self._init(x, y)
        if self.m > self.variable_num:
            raise ValueBoundaryError
        self._fit()

    def _fit(self):
        self.forest = [CART() for i in range(self.tree_num)]
        self.feature_list = []
        for i, tree in enumerate(self.forest):
            random_x, random_y, select_features = self._sample_from_x(i)   # 默认以当前树的编号作为随机种子，使每次运行时抽样结果完全一致
            tree.fit(random_x, random_y)
            self.feature_list.append(select_features)

    def _sample_from_x(self, seed):
        np.random.seed(seed)
        selected_sample_ids = np.random.randint(0, self.sample_num, self.sample_num)
        np.random.seed(seed)
        selected_feature_ids = np.random.choice(range(self.variable_num), self.m, False)
        random_x = self.x[selected_sample_ids, :]
        random_x = random_x[:, selected_feature_ids]
        random_y = self.y[selected_sample_ids]
        return random_x, random_y, selected_feature_ids

    def predict(self, x):
        if self.forest is None:
            raise ModelNotFittedError

        # tree_num*sample_num 行为每一棵树的预测，列为对每一个样本的预测
        predict_results_mat = np.array([tree.predict(x[:, self.feature_list[i]])
                                        for i, tree in enumerate(self.forest)])
        return self._vote(predict_results_mat)

    def _vote(self, result):
        if result.shape[0] != self.tree_num:
            raise TreeNumberMismatchError
        voted_result = list(map(self._one_vote, result.T))
        return np.array(voted_result)

    def _one_vote(self, result):
        if len(result) != self.tree_num:
            raise TreeNumberMismatchError
        count = Counter(result)
        return max(count, key=count.get)

    def score(self, x, y):
        y_predict = self.predict(x)
        if self.label_type == LabelType.binary:
            return classify_f1(y_predict, y)
        else:
            return classify_f1_micro(y_predict, y)

    def classify_plot(self, test_x, test_y):
        classify_plot(self.new(1, 1), self.x, self.y, test_x, test_y, title=self.__doc__)

    @classmethod
    def new(cls, *args):
        return cls(*args)