# -*- coding:utf-8 -*-

from __future__ import division, absolute_import

from collections import Counter
import numpy as np
from simple_ml.base.base_enum import *
from simple_ml.base.base_error import *
from simple_ml.base.base_model import *
from simple_ml.evaluation import classify_plot, classify_f1, classify_f1_micro, regression_r2
from simple_ml.tree import CART
from simple_ml.data_handle import get_k_folder_idx
from simple_ml.logistic import LogisticRegression


__all__ = [
    'AdaBoost',
    'GBDT',
    'RandomForest',
    'ClassifierType',
    'Stacking',
]


class AdaBoost(BaseClassifier):

    __doc__ = "AdaBoost Classifier"

    def __init__(self, classifier=ClassifierType.LR, nums=10):
        super(AdaBoost, self).__init__()
        self.classifier = classifier
        self.nums = nums
        self.clf_list = []
        self.alpha = np.ones(self.nums)
        self._init_classifier()
        self.current_clf_num = 0

    def _init_classifier(self):
        if self.classifier == ClassifierType.LR:
            from simple_ml.logistic import Ridge
            self.clf_list = [Ridge() for i in range(self.nums)]
        elif self.classifier == ClassifierType.KNN:
            from simple_ml.knn import KNN
            self.clf_list = [KNN() for i in range(self.nums)]
        elif self.classifier == ClassifierType.CART:
            from simple_ml.tree import CART
            self.clf_list = [CART() for i in range(self.nums)]
        elif self.classifier == ClassifierType.SVM:
            from simple_ml.svm import SVM
            self.clf_list = [SVM() for i in range(self.nums)]
        elif self.classifier == ClassifierType.NB:
            from simple_ml.bayes import NaiveBayes
            self.clf_list = [NaiveBayes() for i in range(self.nums)]
        else:
            raise ClassifierTypeError("暂不支持的分类器，你想你也可以自己添加（先找到我哦）")



    def _re_sample(self, x, y, weight):
        """
        这里采用bootstrap抽样方法，用以解决带权重的分类
        :param x:
        :param y:
        :param weight:
        :return:
        """
        choose_id = np.random.choice(np.arange(self.sample_num), self.sample_num, p=weight, replace=True)
        return x[choose_id], y[choose_id]

    def fit(self, x, y):
        self._init(x, y)
        self.weights = np.array([1 / self.sample_num for i in range(self.sample_num)])
        self.current_clf_num = 0
        for m in range(self.nums):
            clf = self.clf_list[m]
            x, y = self._re_sample(x, y, self.weights)
            clf.fit(x, y)
            y_predict = clf.predict(x)
            e, alpha = self._update_alpha(y_predict, y)

            self.alpha[m] = alpha
            self._update_weight(y_predict, y, alpha)
            # print("Model %s fitted" % m)
            self.current_clf_num += 1
            if e < 0.1:
                break

    def predict(self, x):
        if x.shape[1] != self.variable_num:
            raise FeatureNumberMismatchError
        res = np.zeros(x.shape[0])
        for m in range(self.current_clf_num):
            clf = self.clf_list[m]
            predict = clf.predict(x)
            predict = self._adj_y(predict)
            res += self.alpha[m] * predict
        func = np.vectorize(lambda j: 1 if j > 0 else 0)
        return func(res)

    @staticmethod
    def _adj_y(y):
        return np.array([i if i == 1 else -1 for i in y])

    def _update_weight(self, y_p, y_t, alpha):
        # 通过这一步将分错的样本权值调高，分对的样本权值调低，必须是1或者-1
        y_p = self._adj_y(y_p)
        y_t = self._adj_y(y_t)
        temp = np.exp(- alpha * np.multiply(y_p, y_t))
        self.weights = np.multiply(temp, self.weights)
        self.weights = self.weights / np.sum(self.weights)

    def _update_alpha(self, y_predict, y_true):
        e = np.sum(self.weights[y_predict != y_true]) / np.sum(self.weights)
        alpha = 0.5 * np.log((1 - e) / max(float(e), 1e-10))
        return e, alpha

    def score(self, x, y):
        y_predict = self.predict(x)
        return classify_f1(y_predict, y)

    def classify_plot(self, x, y, title=""):
        classify_plot(self.new(self.classifier, self.nums),
                      self.x, self.y, x, y, title=self.__doc__ + title)

    @classmethod
    def new(cls, a, b):
        return cls(a, b)


class _CARTForGBDT(CART):

    __doc__ = "CART For GBDT"

    def __init__(self, max_depth=4, min_samples_leaf=10):
        super(_CARTForGBDT, self).__init__(max_depth, min_samples_leaf)
        self.leaf_node_list = []
        self.importance = None

    def fit(self, x, y):
        self._init(x, y)
        self.importance = np.zeros(self.variable_num)
        self._root = self._gen_tree(GBDTTreeNode(None, None, np.arange(self.x.shape[0])), 1)

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
                    node.leaf_label = np.random.choice(temp, 1)[0]
            node.gamma = node.leaf_label    # 当平方损失时，gamma就等于y_predict
            self.leaf_node_list.append(node)
            return node

        best_split = self._get_best_split(node.data_id)
        if best_split[0] is None:
            if self.label_type == LabelType.continuous:
                node.leaf_label = np.mean(self.y[node.data_id])
            else:
                node.leaf_label = np.argmax(np.bincount(self.y[node.data_id]))
            node.gamma = node.leaf_label
            self.leaf_node_list.append(node)
            return node

        # 非叶子节点带来的误差的减小做为该特征的重要性
        self.importance[best_split[0]] += best_split[3]

        feature_arr = self.x[node.data_id, best_split[0]]
        if best_split[2]:
            left = GBDTTreeNode(None, None, node.data_id[feature_arr <= best_split[1]], best_split[0], best_split[1])
            right = GBDTTreeNode(None, None, node.data_id[feature_arr > best_split[1]], best_split[0], best_split[1])
        else:
            left = GBDTTreeNode(None, None, node.data_id[feature_arr == best_split[1]], best_split[0], best_split[1])
            right = GBDTTreeNode(None, None, node.data_id[feature_arr != best_split[1]], best_split[0], best_split[1])
        node.left = self._gen_tree(left, depth + 1)
        node.right = self._gen_tree(right, depth + 1)
        return node

    def _predict_single(self, x, node):
        if node.leaf_label is not None:
            return node.gamma

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


class GBDT(BaseClassifier, BaseFeatureSelect):

    __doc__ = "GBDT Regression"

    def __init__(self, nums=10, learning_rate=1):
        """
        1. $F_0(x) = argmin_\rho \sum _{i=1}^N L(y_i, \rho)$
        2. For $m = 1$ to $M$ do:
        3. $\qquad \tilde y_i = -[{\partial L(y,F(x_i)) \over \partial F(x_i)}]_{F(x) = F_{m-1}(x)}, i = 1, N$
        4. $\qquad \{R_{jm}\}_1^J = J-terminal\, node\, tree(\{ \tilde y_i, x_i \}_i^N)$
        5. $\qquad \gamma_{jm} = argmin_\gamma \sum_{x_i\in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$
        6. $\qquad F_m(x) = F_{m-1}(x) + \sum_{j=1}^J \gamma_{jm}I(x \in R_{jm})$
        7. endFor
        end Algorighm
        - 特征只支持0-1
        - label支持连续值和离散值
        """
        super(GBDT, self).__init__()
        self.nums = nums
        self.learning_rate = learning_rate
        self.F = []
        self.trees = [_CARTForGBDT() for i in range(nums)]
        self._importance = None

    @property
    def importance(self):
        return self._importance

    def fit(self, x, y):
        self._init(x, y)
        if self.label_type != LabelType.continuous:
            raise LabelTypeError("GBDT暂时只支持连续标签")

        self._init_f0()
        temp = []
        for m in range(self.nums):
            y_residual = self._get_residual(m)
            tree = self.trees[m]
            tree.fit(x, y_residual)
            self._update_f(tree)
            temp.append(tree.importance)
        self._importance = np.mean(np.array(temp), axis=0)

    def feature_select(self, top_n):
        """
        特征选择
        :param top_n: 前几个特征
        :return:      选的特征的下标
        """
        if self._importance is None:
            raise ModelNotFittedError
        if top_n > self.variable_num:
            raise TopNTooLargeError
        return self._importance.argsort()[-top_n:][::-1]

    def predict(self, x):
        res = np.zeros(x.shape[0]) + self.F[0][0]
        for m in range(self.nums):
            tree = self.trees[m]
            res += self.learning_rate * tree.predict(x)
        return np.array(res)

    def score(self, x, y):
        y_predict = self.predict(x)
        if self.label_type == LabelType.continuous:
            return regression_r2(y_predict, y)
        else:
            raise LabelTypeError

    def _update_f(self, tree):
        leaf_node_list = tree.leaf_node_list
        f = np.zeros(self.sample_num)
        for leaf_node in leaf_node_list:
            f[leaf_node.data_id] = leaf_node.gamma

        f += self.learning_rate * self.F[-1]
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


class RandomForest(BaseClassifier):

    __doc__ = "Random Forest"

    def __init__(self, m, tree_num=200):
        super(RandomForest, self).__init__()
        self.m = m
        self.tree_num = tree_num
        self.forest = None

    @property
    def the_forest(self):
        return self.forest

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

    def classify_plot(self, test_x, test_y, title=""):
        classify_plot(self.new(self.m, self.tree_num), self.x, self.y, test_x, test_y, title=self.__doc__ + title)

    @classmethod
    def new(cls, *args):
        return cls(*args)


class Stacking(BaseClassifier):

    __doc__ = "模型融合 Stacking 方法"

    def __init__(self, models, meta_model=LogisticRegression(), k_folder=5):
        """
        模型融合Stacking方法，支持分类问题
        :param models:      子模型列表，必须包含fit和predict方法，或者继承BaseClassifier
        :param meta_model:  元分类器，用于stacking第二层分类
        :param k_folder:    折叠次数
        """
        super(Stacking, self).__init__()
        if not isinstance(models, list):
            raise ValueError("models 必须是一个继承fit和predict方法的object列表")
        if len(models) == 0:
            raise EmptyInputError("models 不能为空")
        if not isinstance(models[0], BaseClassifier) or not isinstance(meta_model, BaseClassifier):
            raise ClassifierTypeError("必须选择继承BaseClassifier的分类模型")
        self.models = models
        self.meta_model = meta_model
        self.k_folder = k_folder
        self._x_train_stack = None
        self._x_test_stack = None
        self._score_mat = None

    @property
    def model_num(self):
        return len(self.models)

    @property
    def score_mat(self):
        """
        k折训练时的得分矩阵
        :return: (模型数目 x k折数目)
        """
        return self._score_mat

    def fit(self, x, y):
        self._init(x, y)
        self._fit()

    def _fit(self, quiet=True):
        self._x_train_stack = np.zeros((self.sample_num, self.model_num))
        self._score_mat = np.zeros((self.model_num, self.k_folder))
        for i, (test, train) in enumerate(get_k_folder_idx(self.sample_num, self.k_folder)):
            _X_train = self.x[train]
            _y_train = self.y[train]
            _X_test = self.x[test]
            _y_test = self.y[test]
            _y_test_predict = np.zeros((_X_test.shape[0], self.model_num))

            for j, model in enumerate(self.models):
                model.fit(_X_train, _y_train)
                self._score_mat[j, i] = model.score(_X_test, _y_test)
                _y_test_predict[:, j] = model.predict(_X_test)
                if not quiet:
                    print("The %s th folder, %s th model finished." % (i+1, j+1))

            # 训练集分 k_folder 次填充，得到特征[n_train_samples, clfs_num]
            self._x_train_stack[test, :] = _y_test_predict

    def predict(self, x):
        self._get_new_test(x)
        return self._predict_with_meta_classifier()

    def _get_new_test(self, x):
        self._x_test_stack = np.zeros((x.shape[0], self.model_num))
        for i in range(self.k_folder):
            y_test_predict = np.zeros((x.shape[0], self.model_num))
            for j, model in enumerate(self.models):
                predict = model.predict(x)
                y_test_predict[:, j] = predict
            self._x_test_stack += y_test_predict
        self._x_test_stack /= self.k_folder
        sign = lambda x: 1 if x >= 0.5 else 0
        vec = np.vectorize(sign)
        self._x_test_stack = vec(self._x_test_stack)

    def _predict_with_meta_classifier(self):
        self.meta_model.fit(self._x_train_stack, self.y)
        return self.predict(self._x_test_stack)
