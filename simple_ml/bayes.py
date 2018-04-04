# -*- coding:utf-8 -*-


from .base import *
from .base_enum import LabelType
from .score import *
from .classify_plot import classify_plot
from .base_error import *


class BaseNaiveBayes(BaseClassifier):

    def __init__(self):
        super(BaseNaiveBayes, self).__init__()
        self.is_feature_binary = []
        self.prob_array = None

    def fit(self, x, y):
        self._init(x, y)
        if self.label_type == LabelType.continuous:
            raise LabelTypeError

        self.is_feature_binary = list(map(self._is_binary, self.x.T))
        self._fit()

    @staticmethod
    def _is_binary(column):
        keys = np.unique(column)
        if np.array_equal(keys, np.array([0, 1])):
            return True
        elif keys.shape[0] == 2:
            raise LabelTypeError("Binary label must be 0 or 1")
        return False

    @staticmethod
    def _get_normal_prob(value, mu, sigma2):
        cons = np.sqrt(2 * np.pi)
        return 1 / cons * np.exp(- (value - mu)**2 / (2 * sigma2))

    def _fit(self):
        y_values = np.unique(self.y)
        self.prob_array = np.zeros((len(y_values), self.variable_num+1))  # 第一列为标签的先验概率
        self.continuous_record_dict = {}                                  # 用来存储连续变量在prob_array中对应的位置，以及其期望和方差
        for i, y in enumerate(y_values):
            y_amount = self.y[self.y == y].shape[0]
            self.prob_array[i, 0] = (y_amount + 1) / (self.sample_num + len(y_values))  # 拉普拉斯平滑

            for j, is_binary in enumerate(self.is_feature_binary):
                feature_series = self.x[self.y == y, j]    # 此时只有0或1元素，只记录1的概率，0的概率用1减去
                if is_binary:
                    self.prob_array[i, j+1] = (np.sum(feature_series) + 1) / (y_amount + len(np.unique(self.x[:, j])))
                else:
                    mu = np.mean(feature_series)
                    sigma2 = np.var(feature_series)
                    self.prob_array[i, j+1] = -1
                    self.continuous_record_dict[(i, j+1)] = (mu, sigma2)

    def predict(self, x):
        if self.prob_array is None:
            raise ModelNotFittedError

        return np.array(list(map(self._predict_single_sample, x)))

    def _predict_single_sample(self, x):
        # 1. 更新prob_array中连续变量的取值
        p = np.ones(self.prob_array.shape[0])
        for i in range(self.prob_array.shape[0]):
            p *= self.prob_array[i, 0]    # 先验先乘进去
            for j in range(1, self.prob_array.shape[1]):
                if self.prob_array[i, j] == -1:
                    mu, sigma2 = self.continuous_record_dict[(i, j)]
                    p[i] *= self._get_normal_prob(x[j-1], mu, sigma2)
                else:
                    if x[j-1] == 1:
                        p[i] *= self.prob_array[i, j]
                    else:
                        p[i] *= (1 - self.prob_array[i, j])
        return np.argmax(p)

    def score(self, x, y):
        y_predict = self.predict(x)
        if self.label_type == LabelType.binary:
            return classify_f1(y_predict, y)
        else:
            return classify_f1_macro(y_predict, y)

    def classify_plot(self, x, y):
        classify_plot(self, self.x, self.y, x, y, title='My Naive Bayes')


class BaseBayesMinimumError(BaseClassifier):

    def __init__(self):
        super(BaseBayesMinimumError, self).__init__()
        self.mu = None
        self.sigma = None
        self.prior = None
        self.labels = None

    def fit(self, x, y):
        self._init(x, y)
        if self.label_type == LabelType.continuous:
            raise LabelTypeError

        self._get_normal_distribution()

    def _get_normal_distribution(self):
        _y = np.unique(self.y)
        self.labels = _y
        self.prior = [len(self.y[self.y == i]) / self.sample_num for i in _y]
        self.mu = []
        self.sigma = []
        for i in _y:
            _x = self.x[self.y == i]
            self.mu.append(self._get_mu(_x))
            self.sigma.append(self._get_sigma(_x))

    def _get_probability(self, x):
        res = []
        for i in range(len(self.mu)):
            temp = 1/(np.sqrt(2*np.pi)**self.variable_num * np.sqrt(np.linalg.det(self.sigma[i])))
            temp *= np.exp(-1/2 * np.dot(np.dot((x - self.mu[i]), np.linalg.inv(self.sigma[i])), (x - self.mu[i])))
            temp *= self.prior[i]
            res.append(temp)
        return res

    @staticmethod
    def _get_mu(x):
        return np.mean(x, axis=0)

    @staticmethod
    def _get_sigma(x):
        return np.cov(x.T)

    def predict(self, x):
        if not self.mu:
            raise ModelNotFittedError
        return np.array([self._predict_single(i) for i in x])

    def _predict_single(self, x):
        res = self._get_probability(x)
        return self.labels[np.argmax(res)]

    def score(self, x, y):
        y_predict = self.predict(x)
        if self.label_type == LabelType.binary:
            return classify_f1(y_predict, y)
        else:
            return classify_f1_macro(y_predict, y)

    def classify_plot(self, x, y):
        classify_plot(self, self.x, self.y, x, y, title='My Bayes Minimum Error')


class MyBayesMinimumRisk(BaseBayesMinimumError):

    def __init__(self, cost_mat):
        """
        初始化，保存分类损失矩阵
        :param cost_mat: 分类损失矩阵
               要求：
                   1. m x m 维，m为所有类别数目
                   2. 第i行第j列表示将属于类别i的样本分到类别j所造成的损失
                   3. 每一行，每一列的类别必须按照数值从小到大的顺序排列，
                      比如第i行表示在np.unique(y)中第i个label
        """
        super(MyBayesMinimumRisk, self).__init__()
        self.cost_mat = cost_mat

    def fit(self, x, y):
        label_num = len(np.unique(y))
        if self.cost_mat.shape[0] != label_num or self.cost_mat.shape[1] != label_num:
            raise CostMatMismatchError
        super(MyBayesMinimumRisk, self).fit(x, y)

    def _predict_single(self, x):
        prob = self._get_probability(x)
        alpha = np.dot(prob, self.cost_mat)
        return self.labels[np.argmin(alpha)]
