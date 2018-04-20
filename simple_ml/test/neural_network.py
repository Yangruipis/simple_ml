# -*- coding:utf-8 -*-

from simple_ml.base.base_error import ModelNotFittedError, CostFunctionError, NeuralNetworkParamError
from simple_ml.base.base_model import BaseClassifier
from simple_ml.base.base_enum import LabelType, CostFunction, ActiveFunction
import numpy as np
from simple_ml.evaluation import classify_f1, classify_f1_macro


class NeuralNetwork(BaseClassifier):

    def __init__(self, alpha, output_neuron_num=1, output_active_func=ActiveFunction.sigmoid, cost_func=CostFunction.logistic):
        super(NeuralNetwork, self).__init__()
        self.alpha = alpha
        self.cost_func = cost_func
        if self.cost_func != CostFunction.logistic:
            raise CostFunctionError("暂时只支持Logistic损失")
        self.input_neuron_num = None
        self.output_neuron_num = output_neuron_num
        if self.output_neuron_num != 1:
            raise NeuralNetworkParamError("暂时只支持二分类，即一个输出层神经元")
        self.output_active_func = output_active_func
        self.clear_all()

    def clear_hidden_layer(self):
        self.hidden_layers_num = 0
        self.hidden_layers = []

    def clear_init(self):
        self.w_list = []
        self.b_list = []
        self.z_list = []
        self.a_list = []
        self.delta_list = []

    def clear_all(self):
        self.clear_hidden_layer()
        self.clear_init()

    def model_init(self):
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
        # 此处将输出层和隐含层放到一起了
        self.hidden_layers.append((self.output_neuron_num, self.output_active_func))
        self.hidden_layers_num += 1

        for i in range(self.hidden_layers_num):
            # i=0对应着输入层
            if i == 0:
                neuron_num_this_layer = self.input_neuron_num
            else:
                neuron_num_this_layer = self.hidden_layers[i - 1][0]

            neuron_num_next_layer = self.hidden_layers[i][0]

            self.w_list.append(np.random.rand(neuron_num_next_layer, neuron_num_this_layer))
            self.b_list.append(np.random.rand(neuron_num_next_layer, 1))
            self.z_list.append(np.ones((neuron_num_next_layer, self.sample_num)))
            self.a_list.append(np.ones((neuron_num_next_layer, self.sample_num)))
            self.delta_list.append(np.ones((neuron_num_next_layer, self.sample_num)))

    def add_layer(self, neuron_num, active_func=ActiveFunction.sigmoid):
        self.hidden_layers_num += 1
        self.hidden_layers.append((neuron_num, active_func))

    def add_some_layers(self, layer_num, neuron_num, active_func=ActiveFunction.sigmoid):
        for i in range(layer_num):
            self.add_layer(neuron_num, active_func)

    def fit(self, x, y):
        self._init(x, y)
        self.model_init()

    @staticmethod
    def _sigmoid(arr):
        return 1 / (1 + np.exp(- arr))

    @staticmethod
    def _sigmoid_inv(arr):
        return np.log(arr / (1 - arr))

    @staticmethod
    def _tan_h(arr):
        return np.tanh(arr)

    @staticmethod
    def _tan_h_inv(arr):
        return 1 / 2 * np.log((1 + arr) / (1 - arr))

    @staticmethod
    def _ReLu(arr):
        return np.array([i if i >= 0 else 0 for i in arr])

    @staticmethod
    def _ReLu_inv(arr):
        return NeuralNetwork._ReLu(arr)

    def _forward_transfer(self):
        for layer_id in range(self.hidden_layers_num):
            # neuron_num = self.hidden_layers[layer_id][0]
            active_func = self.hidden_layers[layer_id][1]

            if layer_id == 0:
                # (n(i+1) x n(i)) *  (n(i), m) + (n(i+1), 1)，其中n(i)为前neuron_num，m为样本数
                self.z_list[layer_id] = np.dot(self.w_list[layer_id], self.x.T) + self.b_list[layer_id]
            else:
                self.z_list[layer_id] = np.dot(self.w_list[layer_id], self.a_list[layer_id - 1]) + self.b_list[layer_id]

            if active_func == ActiveFunction.sigmoid:
                self.a_list[layer_id] = self._sigmoid(self.z_list[layer_id])
            elif active_func == ActiveFunction.tanh:
                self.a_list[layer_id] = self._tan_h(self.z_list[layer_id])
            else:
                self.a_list[layer_id] = self._ReLu(self.z_list[layer_id])

    def _error_of_output_layer(self):
        pass

    def _error_of_hidden_layer(self):
        pass

    def _get_grad(self):
        pass

    def _update_param(self):
        pass

    def predict(self, x):
        pass

    def score(self, x, y):
        pass
