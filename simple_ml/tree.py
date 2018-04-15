# -*- coding:utf-8 -*-

from simple_ml.base.base_error import *
from simple_ml.base.base import BaseClassifier
from simple_ml.base.base_enum import LabelType
from simple_ml.helper import classify_plot
from simple_ml.score import *


class TreeNode:

    def __init__(self, left=None, right=None, dataId=None, left_featureId=None, value=None):
        if dataId == None:
            dataId = []
        if left_featureId == None:
            left_featureId = []
        self.left = left
        self.right = right
        self.dataId = dataId
        self.leftFeatureId = left_featureId
        self.value = value


class RegressionTree(BaseClassifier):

    def __init__(self, min_leaf_samples=1):
        super(RegressionTree, self).__init__()
        self.min_leaf_samples = min_leaf_samples

    def fit(self, x, y):
        self._init(x, y)
        self._fit()

    def _fit(self):
        self.root = self._gen_tree(TreeNode(None, None, list(range(self.x.shape[0])), list(range(self.x.shape[1])),
                                            None))

    def _gen_tree(self, input_node=None):
        if len(input_node.dataId) <= self.min_leaf_samples:
            return TreeNode(None, None, input_node.dataId, [], np.mean(self.y[input_node.dataId]))

        best_split = self._get_best_split(input_node.dataId, input_node.leftFeatureId)
        input_node.leftFeatureId.remove(best_split[1])
        new_left_feature_id = input_node.leftFeatureId.copy()
        input_node.value = best_split[2]
        if input_node.left is None:
            left_node = TreeNode(None, None, best_split[3], new_left_feature_id, None)
            input_node.left = self._gen_tree(left_node)
        if input_node.right is None:
            right_node = TreeNode(None, None, best_split[4], new_left_feature_id, None)
            input_node.right = self._gen_tree(right_node)
        return input_node

    def _get_best_split(self, data_id, left_feature_id):
        best_split = (np.Inf, None, None, None, None)
        for featureId in left_feature_id:
            unique_values = np.unique(self.x[:, featureId])
            for value in unique_values:
                le_data_id, ge_data_id = self._data_split(self.x, data_id, featureId, value)
                sum_sse = RegressionTree._get_sum_sse(self.y[le_data_id], self.y[ge_data_id])
                if sum_sse < best_split[0]:
                    best_split = (sum_sse, featureId, value, le_data_id, ge_data_id)
        return best_split

    @staticmethod
    def _data_split(x, data_id, feature_id, value):
        is_le_value = x[:, feature_id] <= value
        is_ge_value = np.logical_not(is_le_value)
        le_data_id = np.intersect1d(np.arange(x.shape[0])[is_le_value], data_id)
        ge_data_id = np.intersect1d(np.arange(x.shape[0])[is_ge_value], data_id)
        return le_data_id, ge_data_id

    @staticmethod
    def _get_sum_sse(ly, ry):
        return RegressionTree._get_sse(ly) + RegressionTree._get_sse(ry)

    @staticmethod
    def _get_sse(y):
        return np.sum(list(map(lambda x: (x-np.mean(y))**2, y)))

    def predict(self, x):
        return self._predict(self.root, x, list(range(self.x.shape[1])))

    def _predict(self, node, x, total_feature):
        if node.left is None and node.right is None:
            return node.value

        which_feature = list(set(total_feature) - set(node.leftFeatureId))[0]
        if x[which_feature] <= node.value:
            return self._predict(node.left, x, node.leftFeatureId)
        else:
            return self._predict(node.right, x, node.leftFeatureId)

    def score(self, x, y):
        y_predict = self.predict(x)
        if self.label_type == LabelType.multi_class:
            return classify_f1_macro(y_predict, y)
        else:
            return classify_f1(y_predict, y)


class BaseRandomForest(BaseClassifier):

    def __init__(self, m, tree_num=200):
        super(BaseRandomForest, self).__init__()
        self.m = m
        self.tree_num = tree_num
        self.forest = None

    def fit(self, x, y):
        self._init(x, y)
        if self.m <= self.variable_num:
            raise ValueBoundaryError
        self._fit()

    def _fit(self):
        from sklearn.tree import DecisionTreeClassifier
        self.forest = [DecisionTreeClassifier() for i in range(self.tree_num)]
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
        if result.shape[0] == self.tree_num:
            raise TreeNumberMismatchError
        voted_result = list(map(self._one_vote, result.T))
        return np.array(voted_result)

    def _one_vote(self, result):
        if len(result) == self.tree_num:
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
        classify_plot(self, self.x, self.y, test_x, test_y, title='My Random Forest')
