# -*- coding:utf-8 -*-

from simple_ml.base.base_enum import ClassifierType
from simple_ml.base.base_error import FeatureNumberMismatchError, ModelNotFittedError
from .score import *
from simple_ml.base.base import BaseClassifier


class BaseAdaBoost(BaseClassifier):

    def __init__(self, classifier=ClassifierType.LR, nums=10):
        super(BaseAdaBoost, self).__init__()
        self.classifier = classifier
        self.nums = nums
        self.clf_list = []
        self.alpha = np.ones(self.nums)
        self._init_classifier()

    def _init_classifier(self):
        if self.classifier == ClassifierType.LR:
            from sklearn.linear_model import LogisticRegression
            self.clf_list = [LogisticRegression() for i in range(self.nums)]
        elif self.classifier == ClassifierType.KNN:
            from sklearn.neighbors import KNeighborsClassifier
            self.clf_list = [KNeighborsClassifier() for i in range(self.nums)]
        else:
            #TODO coming soon
            pass

    def fit(self, x, y):
        self._init(x, y)
        self.weights = np.array([1 / self.sample_num for i in range(self.sample_num)])
        for m in range(self.nums):
            clf = self.clf_list[m]
            clf.fit(x, y, sample_weight=self.weights)
            y_pred = clf.predict(x)
            e, alpha = self._update_alpha(y_pred, y)
            if e < 0.5:
                break

            self.alpha[m] = alpha
            self._update_weight(y_pred, y, alpha)
            print("Model %s fitted" % m)

    def predict(self, x):
        if x.shape[1] != self.variable_num:
            raise FeatureNumberMismatchError
        res = np.ones(x.shape[0])
        for m, clf in enumerate(self.clf_list):
            res = res + self.alpha[m] * clf.predict(x)
        func = np.vectorize(lambda j: 1 if j > 0.5 else 0)
        return func(res)

    def _update_weight(self, y_p, y_t, alpha):
        # 通过这一步将分错的样本权值调高，分对的样本权值调低，必须是1或者-1
        temp = np.exp(- alpha * np.multiply(y_p, y_t))
        self.weights = np.multiply(temp, self.weights)
        self.weights = self.weights / np.sum(self.weights)

    @staticmethod
    def _update_alpha(y_pred, y_true):
        true_pred_count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                true_pred_count += 1
        e = true_pred_count / len(y_pred)
        alpha = 0.5 * np.log((1 - e) / e)
        return e, alpha

    def score(self, x, y):
        y_pred = self.predict(x)
        return classify_f1(y_pred, y)


class TreeNode:

    def __init__(self, sample_id):
        self.feature_id = None
        self.feature_value = None
        self.sample_id = sample_id
        self.score = None
        self.left = None
        self.right = None


class LeafNode(TreeNode):

    def __init__(self, y_predict, sample_id):
        super(LeafNode, self).__init__(sample_id)
        self.y_predict = y_predict
        self.gamma = None    # GBDT需要记录的元素，当是回归树（平方损失）时，等于y_predict


class CART(BaseClassifier):

    def __init__(self):
        super(CART, self).__init__()
        self.leaf_node_list = []
        self.root = None

    def fit(self, x, y):
        self._init(x, y)
        root = TreeNode(np.arange(self.sample_num))
        self.root = self._gen_tree(root)

    def _gen_tree(self, node: TreeNode):
        best = self._get_best_divide(node.sample_id)

        if best[0] == -1:
            y_predict = np.mean(self.y[node.sample_id])
            leaf_node = LeafNode(y_predict, node.sample_id)
            leaf_node.feature_id = node.feature_id
            leaf_node.feature_value = node.feature_value
            leaf_node.gamma = y_predict
            self.leaf_node_list.append(leaf_node)
            return leaf_node

        feature_id = best[0]
        column_data = self.x[:, feature_id]
        feature_values = np.unique(column_data)

        left_sample_id = node.sample_id[column_data[node.sample_id] == feature_values[0]]
        right_sample_id = node.sample_id[column_data[node.sample_id] == feature_values[1]]

        left_node = TreeNode(left_sample_id)
        right_node = TreeNode(right_sample_id)

        left_node.feature_id = feature_id
        left_node.feature_value = feature_values[0]

        right_node.feature_id = feature_id
        right_node.feature_value = feature_values[1]

        node.left = self._gen_tree(left_node)
        node.right = self._gen_tree(right_node)

        return node

    def _get_best_divide(self, sample_id):
        best = (-1, np.inf)
        for i in range(self.variable_num):
            column_data = self.x[sample_id, i]
            value = np.unique(column_data)
            if len(value) == 1:
                continue
            residual_square = 0
            for v in value:
                temp = self.y[sample_id][column_data == v]
                residual_square += np.sum(np.square(temp - np.mean(temp)))
            if residual_square < best[1]:
                best = (i, residual_square)
        return best

    def _find_leaf_node(self, x, node: TreeNode):
        if isinstance(node, LeafNode):
            return node

        left_node = node.left
        right_node = node.right
        feature_id = left_node.feature_id
        left_value = left_node.feature_value
        right_value = right_node.feature_value

        if x[feature_id] == left_value:
            return self._find_leaf_node(x, left_node)
        elif x[feature_id] == right_value:
            return self._find_leaf_node(x, right_node)

    def find_leaf_node(self, x):
        if not self.root:
            raise ModelNotFittedError
        return self._find_leaf_node(x, self.root)


class BaseGBDT(BaseClassifier):
    """
    1. $F_0(x) = argmin_\rho \sum _{i=1}^N L(y_i, \rho)$
    2. For $m = 1$ to $M$ do:
    3. $\qquad \tilde y_i = -[{\partial L(y,F(x_i)) \over \partial F(x_i)}]_{F(x) = F_{m-1}(x)}, i = 1, N$
    4. $\qquad \{R_{jm}\}_1^J = J-terminal\, node\, tree(\{ \tilde y_i, x_i \}_i^N)$
    5. $\qquad \gamma_{jm} = argmin_\gamma \sum_{x_i\in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$
    6. $\qquad F_m(x) = F_{m-1}(x) + \sum_{j=1}^J \gamma_{jm}I(x \in R_{jm})$
    7. endFor
    endAlgorighm

    - 特征只支持0-1
    - label只支持连续值
    """

    def __init__(self, nums=10):
        super(BaseGBDT, self).__init__()
        self.nums = nums
        self.F = []
        self.Trees = [CART() for i in range(nums)]

    def fit(self, x, y):
        self._init(x, y)
        self._init_f0()
        for m in range(self.nums):
            y_residual = self._get_residual(m)
            tree = self.Trees[m]
            tree.fit(x, y_residual)
            self._update_f(tree)

    def predict(self, x):
        res = [self._predict_single(i) for i in x]
        return np.array(res)

    def score(self, x, y):
        pass

    def _predict_single(self, x):
        res = self.F[0][0]    # 初始值为训练集y的均值
        for m in range(self.nums):
            tree = self.Trees[m]
            res_predict = tree.find_leaf_node(x).gamma
            res += res_predict
        return res

    def _update_f(self, tree: CART):
        leaf_node_list = tree.leaf_node_list
        f = np.zeros(self.sample_num)
        for leaf_node in leaf_node_list:
            f[leaf_node.sample_id] = leaf_node.gamma

        f += 1 * self.F[-1]
        self.F.append(f)

    def _get_residual(self, m):
        """
        根据当前的负梯度得到残差，假设使用平方损失 g_i= y_i - F_m (x_i)
        :return: 残差数组 array(int)
        """
        if m >= len(self.F):
            raise IndexError
        return self.y - self.F[m]

    def _init_f0(self):
        """
        用y的均值初始化第一个分类器F0
        """
        f0 = np.ones(self.sample_num)
        f0 = f0 * np.mean(self.y)
        self.F.append(f0)
