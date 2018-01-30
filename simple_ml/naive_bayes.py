# -*- coding:utf-8 -*-


from .my_classifier import *
from .my_enumrate import LabelType
from .score import *
from .classify_plot import classify_plot
from .my_error import *


class MyNaiveBayes(MyClassifier):

    def __init__(self):
        super(MyNaiveBayes, self).__init__()
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
