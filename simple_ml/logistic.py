# -*- coding:utf-8 -*-

"""
- 目前只支持二元响应变量
- 暂时不支持加入惩罚项的功能
- 结果经过检验，和stata的logit回归结果一致
"""

from .my_classifier import *
from .score import *
from .classify_plot import classify_plot
from .my_error import FeatureNumberMismatchError


class MyLogisticRegression(MyClassifier):

    def __init__(self, tol=0.01, step=0.01, threshold=0.5, has_intercept=True):
        super(MyLogisticRegression, self).__init__()
        self.tol = tol
        self.step = step
        self.has_intercept = has_intercept
        self.threshold = threshold
        self.w = None

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _update_w(self, w_old, step):
        sigmoid_array = self._sigmoid(np.dot(self.x, w_old.reshape(-1, 1)).reshape(1, -1)[0])
        epsilon_array = self.y - sigmoid_array
        w_new = w_old + step * np.dot(self.x.T, epsilon_array)
        return w_new

    def _loss_function_value(self, w):
        sigmoid_array = self._sigmoid(np.dot(self.x, w.reshape(-1, 1)).reshape(1, -1)[0])    # 存在重复计算的问题
        return -1 / self.variable_num * np.sum(self.y * np.log(sigmoid_array) + (1 - self.y) *
                                               np.log(1 - sigmoid_array))

    @staticmethod
    def _add_ones(x):
        return np.column_stack((np.ones(x.shape[0]), x))

    def _init(self, x, y):
        super(MyLogisticRegression, self)._init(x, y)
        if self.has_intercept:
            self.x = self._add_ones(self.x)
            self.variable_num += 1

    def fit(self, x, y):
        self._init(x, y)
        self.w, loss = self._fit()
        if len(self.w) == self.variable_num:
            raise FeatureNumberMismatchError

    def _fit(self):
        """
        梯度下降进行求解
        ref: http://m.blog.csdn.net/zjuPeco/article/details/77165974
        """
        w_old = np.zeros(self.variable_num)
        loss_init = self._loss_function_value(w_old)
        t = np.Inf
        loss_new = loss_init
        while t > self.tol:
            w_old = self._update_w(w_old, self.step)
            loss_new = self._loss_function_value(w_old)
            t = abs(loss_new - loss_init)
            loss_init = loss_new
        return w_old, loss_new

    def _predict(self, x):
        if self.has_intercept:
            x = self._add_ones(x)

        if x.shape[1] == self.variable_num:
            raise FeatureNumberMismatchError
        self.y = np.dot(self.x, self.w)
        return self._sigmoid(self.y)

    def predict(self, x):
        if self.w is None:
            raise ModelNotFittedError

        return np.array([1 if i >= self.threshold else 0 for i in self._predict(x)])

    def predict_prob(self, x):
        return self._predict(x)

    def score(self, x, y):
        # predict_prob = self._predict(x)
        predict_y = self.predict(x)
        return classify_precision(predict_y, y)

    def auc_plot(self, x, y):
        predict_y = self.predict_prob(x)
        classify_roc_plot(predict_y, y)

    def classify_plot(self, x, y):
        classify_plot(self, self.x, self.y, x, y, title='My Logistic Regression')
