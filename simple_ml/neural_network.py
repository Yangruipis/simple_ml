# -*- coding:utf-8 -*-

from __future__ import division, absolute_import

from simple_ml.base.base_error import *
from simple_ml.base.base_model import *
from simple_ml.base.base_enum import *
import numpy as np
from simple_ml.evaluation import classify_f1, classify_plot, classify_roc_plot


__all__ = ['NeuralNetwork', 'ActiveFunction', 'CostFunction']


class NeuralNetwork(BaseClassifier):

    __doc__ = "BP Neural Network"

    def __init__(self, alpha=0.5, threshold=0.5, iter_times=100, output_neuron_num=1,
                 output_active_func=ActiveFunction.sigmoid, cost_func=CostFunction.logistic):
        """
        BP神经网络
        :param alpha:                学习率，用于梯度下降
        :param threshold:            分类阈值，设置为None时默认使用 S vs S
        :param iter_times:           迭代次数
        :param output_neuron_num:    输出层神经元，为1表示二分类
        :param output_active_func:   输出层激活函数，可选sigmoid，relu，tanh
        :param cost_func:            损失函数，可选logistics损失和平方损失
        """
        super(NeuralNetwork, self).__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.iter_times = iter_times
        self.cost_func = cost_func
#        if self.cost_func != CostFunction.logistic:
#            raise CostFunctionError("暂时只支持Logistic损失")
        self.input_neuron_num = None
        self.output_neuron_num = output_neuron_num
        if self.output_neuron_num != 1:
            raise NeuralNetworkParamError("暂时只支持二分类，即一个输出层神经元")
        self.output_active_func = output_active_func
        self.w_list = []
        self.b_list = []
        self.z_list = []
        self.a_list = []
        self.delta_list = []
        self.layers_num = 1
        self.layers = [(self.output_neuron_num, self.output_active_func)]

    def _clear_hidden_layer(self):
        # 此处将输出层和隐含层放到一起了，layers不包括输入层
        self.layers_num = 1
        self.layers = [(self.output_neuron_num, self.output_active_func)]

    def _clear_init(self):
        self.w_list = []
        self.b_list = []
        self.z_list = []
        self.a_list = []
        self.delta_list = []

    def clear_all(self):
        self._clear_hidden_layer()
        self._clear_init()

    def _model_init(self):
        """
        - 模型初始化，每次add_layer之后模型必须初始化
            - 神经网络输入层神经元个数就是特征数，每个特征就是一个神经元
            - 权重 w_list
                - 用二维数组的列表存储, List[np.2darray]，长度为: self.hidden_layer_num + 1
                - 列表中第i个二维数组存储了 第i层(输入层为第一层）指向第 i + 1 层的权重
                - 第i个二维数组，行数表示第i+1层的神经元数目，列数表示第i层的神经元数目，
                - 第p行q列，表示i层第q个神经元指向i+1层第p个神经元的权重
                例： 如果最后一层隐含层有3个神经元，输出层有一个神经元，那么w_list[-1].shape = (1, 3)
            - 偏移项 b_list
                - 一维数组的列表存储偏移项，List[np.ndarrary](注意此处是列向量）,shape=(n,1),n为第i+1层神经元数目，
                - 长度为：self.hidden_layer_num + 1
                - 第i个数组表示了第 i+1 层的偏移项，长度等于该层的神经元数目
            - 线性值 z_list
                - 二维数组的列表，List[np.2darray]，z_list[i].shape = (n, m)，n为第i+1层的神经元数目，m为样本数目
            - 激活值 a_list，同z_list
            - 误差 delta_list，同z_list
        :return:
        """
        self.input_neuron_num = self.variable_num

        for i in range(self.layers_num):
            # i=0对应着输入层
            if i == 0:
                neuron_num_this_layer = self.input_neuron_num
            else:
                neuron_num_this_layer = self.layers[i - 1][0]

            neuron_num_next_layer = self.layers[i][0]

            self.w_list.append(np.random.rand(neuron_num_next_layer, neuron_num_this_layer))
            self.b_list.append(np.random.rand(neuron_num_next_layer, 1))
            self.z_list.append(np.ones((neuron_num_next_layer, self.sample_num)))
            self.a_list.append(np.ones((neuron_num_next_layer, self.sample_num)))
            self.delta_list.append(np.ones((neuron_num_next_layer, self.sample_num)))

    def add_layer(self, neuron_num, active_func=ActiveFunction.sigmoid):
        self.layers_num += 1
        self.layers.insert(0, (neuron_num, active_func))

    def add_some_layers(self, layer_num, neuron_num, active_func=ActiveFunction.sigmoid):
        for i in range(layer_num):
            self.add_layer(neuron_num, active_func)

    def fit(self, x, y):
        if self.layers_num == 1:
            print("""
            Warning: Now the Model Have No Hidden Layer, If You Want Add Some, Please Use 
                     NeuralNetwork().add_layer(self, neuron_num, active_func=ActiveFunction.sigmoid) or
                     NeuralNetwork().add_some_layers(self, layer_num, neuron_num, active_func=ActiveFunction.sigmoid)
            """)
        self._init(x, y)
        if self.label_type != LabelType.binary:
            raise LabelTypeError("暂时只支持二分类")
        self._model_init()
        for i in range(self.iter_times):
            self._forward_transfer(self.x)
            self._error_of_output_layer()
            self._error_of_hidden_layer()
            self._update_grad()

    @staticmethod
    def _sigmoid(arr):
        return 1 / (1 + np.exp(- arr))

    @staticmethod
    def _sigmoid_inv(arr):
        # 导数
        a = NeuralNetwork._sigmoid(arr)
        return a * (1 - a)

    @staticmethod
    def _tan_h(arr):
        return np.tanh(arr)

    @staticmethod
    def _tan_h_inv(arr):
        # 导数
        return 1 - (NeuralNetwork._tan_h(arr))**2

    @staticmethod
    def _relu(arr):
        func = lambda x: x if x >= 0 else 0
        vfunc = np.vectorize(func)
        return vfunc(arr)

    @staticmethod
    def _relu_inv(arr):
        # 导数
        func = lambda x: 1 if x > 0 else 0
        vfunc = np.vectorize(func)
        return vfunc(arr)

    def _forward_transfer(self, x):
        for layer_id in range(self.layers_num):
            # neuron_num = self.hidden_layers[layer_id][0]
            active_func = self.layers[layer_id][1]

            if layer_id == 0:
                # (n(i+1) x n(i)) *  (n(i), m) + (n(i+1), 1)，其中n(i)为前neuron_num，m为样本数
                self.z_list[layer_id] = np.dot(self.w_list[layer_id], x.T) + self.b_list[layer_id]
            else:
                self.z_list[layer_id] = np.dot(self.w_list[layer_id], self.a_list[layer_id - 1]) + self.b_list[layer_id]

            if active_func == ActiveFunction.sigmoid:
                self.a_list[layer_id] = self._sigmoid(self.z_list[layer_id])
            elif active_func == ActiveFunction.tanh:
                self.a_list[layer_id] = self._tan_h(self.z_list[layer_id])
            else:
                self.a_list[layer_id] = self._relu(self.z_list[layer_id])

    def _error_of_output_layer(self):
        """
        计算输出层误差
        """
        # partial_a 为损失函数对输出层结果的偏导，当输出层只有一个神经元时，其长度为 1 x m(m为样本数)
        if self.cost_func == CostFunction.logistic:
            self.partial_a = - (self.y / self.a_list[-1] - (1 - self.y) / (1 - self.a_list[-1]))
        elif self.cost_func == CostFunction.square:
            self.partial_a = - 2 * (self.y - self.a_list[-1])
        else:
            # TODO: 加入其他的损失函数
            raise CostFunctionError("不支持其他的损失函数")

        # delta_list[-1] 长度也为 (1 x m)
        if self.layers[-1][1] == ActiveFunction.sigmoid:
            self.delta_list[-1] = np.multiply(self.partial_a, self._sigmoid_inv(self.z_list[-1]))
        elif self.layers[-1][1] == ActiveFunction.tanh:
            self.delta_list[-1] = np.multiply(self.partial_a, self._tan_h_inv(self.z_list[-1]))
        else:
            self.delta_list[-1] = np.multiply(self.partial_a, self._relu_inv(self.z_list[-1]))

    def _error_of_hidden_layer(self):
        """
        - 必须在运行 self._error_of_output_layer() 之后
        - 计算隐含层误差
        """
        for i in range(self.layers_num - 2, -1, -1):
            # 注意，这里是-2不是-1，因为输出层的误差已经计算完成
            active_func = self.layers[i][1]
            if active_func == ActiveFunction.sigmoid:
                self.delta_list[i] = np.multiply(np.dot(self.w_list[i + 1].T, self.delta_list[i + 1]),
                                                 self._sigmoid_inv(self.z_list[i]))
            elif active_func == ActiveFunction.tanh:
                self.delta_list[i] = np.multiply(np.dot(self.w_list[i + 1].T, self.delta_list[i + 1]),
                                                 self._tan_h_inv(self.z_list[i]))
            else:
                self.delta_list[i] = np.multiply(np.dot(self.w_list[i + 1].T, self.delta_list[i + 1]),
                                                 self._relu_inv(self.z_list[i]))

    def _update_grad(self):
        """
        - 必须运行在 self._error_or_hidden_layer() 之后
        - 计算每层参数的梯度，并且进行更新
        """
        for i in range(self.layers_num - 1, -1, -1):
            neuron_num = self.layers[i][0]
            self.b_list[i] -= self.alpha * np.mean(self.delta_list[i], axis=1).reshape(neuron_num, 1)
            if i != 0:
                self.w_list[i] -= self.alpha / self.sample_num * np.dot(self.delta_list[i], self.a_list[i-1].T)
            else:
                self.w_list[i] -= self.alpha / self.sample_num * np.dot(self.delta_list[i], self.x)

    def predict(self, x):
        y_prob = self.predict_prob(x)
        return np.array([1 if i >= self.threshold else 0 for i in y_prob])

    # TODO: S vs S 的方法自动分类，无需给定 threshold
    def auto_threshold(self):
        pass

    def predict_prob(self, x):
        if self.input_neuron_num is None:
            raise ModelNotFittedError

        if x.shape[1] != self.variable_num:
            raise FeatureNumberMismatchError
        self._forward_transfer(x)
        return self.a_list[-1].ravel()

    def score(self, x, y):
        y_predict = self.predict(x)
        return classify_f1(y_predict, y)

    def classify_plot(self, x, y, title=""):
        classify_plot(self.new(self.alpha, self.threshold, self.iter_times, self.output_neuron_num,
                               self.output_active_func, self.cost_func, self.layers),
                      self.x, self.y, x, y, title=self.__doc__ + title)

    @classmethod
    def new(cls, alpha, threshold, iter_times, output_neuron_num, output_active_func, cost_func, layers):
        new_cls = cls(alpha, threshold, iter_times, output_neuron_num, output_active_func, cost_func)
        new_cls.layers = layers
        new_cls.layers_num = len(layers)
        return new_cls

    def auc_plot(self, x, y):
        predict_y = self.predict_prob(x)
        classify_roc_plot(predict_y, y)
