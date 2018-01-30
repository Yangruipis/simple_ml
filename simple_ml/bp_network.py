# -*- coding:utf-8 -*-


from .my_classifier import *
from .score import *


class MyBPNetwork(MyClassifier):

    def __init__(self, alpha=0.1, iter_times=1000, hidden_layer=1, hidden_neuron=4):
        super(MyBPNetwork, self).__init__()
        self.alpha = alpha
        self.iterTimes = iter_times
        self.hide_layer_amount = hidden_layer               # 多少层隐含层
        self.neuron_amount = hidden_neuron                 # 隐含层和输入层每层多少个神经元（简化问题，假设隐含层和输入层元数目一样）
        # 简化问题，假设输出层只有一个神经元
        self.__clear()
        self._init_w_and_b()

    def __clear(self):
        # 三维数组存储权重，第一维表示第几层(包括了输入和输出层，没有则为0)，第二维表示起点，第三维表示终点
        # 隐含层权重（没有以输入层为终点的权重）
        self.w_hide = np.ndarray((self.hide_layer_amount, self.neuron_amount, self.neuron_amount))
        self.b_hide = np.zeros((self.hide_layer_amount, self.neuron_amount))                     # 隐含层阈值（输入层没有阈值）
        self.z_hide = np.zeros((self.hide_layer_amount, self.neuron_amount))                     # 隐含层线性值（输入层没有线性结果）
        self.a_hide = np.zeros((self.hide_layer_amount, self.neuron_amount))                     # 隐含层激活

        self.w_output = np.zeros(self.neuron_amount)
        self.b_output = None
        self.z_output = None
        self.a_output = None

        self.delta_output = None                              # 输出层的误差
        self.delta_hidden = np.zeros((self.hide_layer_amount, self.neuron_amount))            # 隐含层的误差

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 - np.exp(-x))

    @staticmethod
    def _sigmoid_inv(p):
        return np.log(1 / (1 - p))

    def _init_w_and_b(self):
        self.w_hide = np.array([np.random.normal(0, 1) for i in self.w_hide.ravel()]).reshape(self.w_hide.shape)
        self.b_hide = np.array([np.random.normal(0, 1) for i in self.b_hide.ravel()]).reshape(self.b_hide.shape)
        self.w_output = np.array([np.random.normal(0, 1) for i in self.w_output])
        self.b_output = np.random.normal(0, 1)

    def fit(self, x, y):
        self._init(x, y)
        self._fit(self.iterTimes, self.alpha)

    def _forward_transfer(self):
        """
        前向传递，根据w和b，计算出每一神经元的z和a
        """
        for layer_id in range(self.hide_layer_amount):
            for neuron_id in range(self.neuron_amount):
                if layer_id == 0:
                    self.z_hide[layer_id, neuron_id] = np.dot(self.w_hide[layer_id, :, neuron_id], self.x_vector) + \
                                                       self.b_hide[layer_id, neuron_id]
                else:
                    self.z_hide[layer_id, neuron_id] = np.dot(self.w_hide[layer_id, :, neuron_id],
                                                              self.a_hide[layer_id-1, :]) + \
                                                       self.b_hide[layer_id, neuron_id]
                self.a_hide[layer_id, neuron_id] = self._sigmoid(self.z_hide[layer_id, neuron_id])

        self.z_output = np.dot(self.w_output, self.a_hide[-1, :]) + self.b_output
        self.a_output = self._sigmoid(self.z_output)

    def _cal_output_error(self):
        """
        根据公式1计算输出层的误差
        假设使用平方损失，则 J = (y_i - a_i)^2
        J' = -2(y_i - a_i)
        delta_output = -2(y_i - a_output) sigmiodInv(z_output)
        """
        self.delta_output = -2 * (self.y_vector - self.a_output) * self._sigmoid(self.z_output)

    def _cal_hidden_error(self):
        """
        根据公式2计算隐含层误差
        """

        # step1. 计算最后一层 hiden layer 的误差
        self.delta_hidden[-1, :] = np.multiply(self.w_output * self.delta_output, self.z_hide[-1, :])

        # step. 计算其余各层的 误差

        for layer in range(self.hide_layer_amount-1):
            self.delta_hidden[layer, :] = np.multiply(np.dot(self.w_hide[layer+1, :, :].T,
                                                             self.delta_hidden[layer+1, :]), self.z_hide[layer, :])

    def _update_param(self, alpha):
        """
        根据误差更新w和b
        - b的更新梯度就是误差
        - w的更新梯度如下计算
        """
        self.w_hide_update = self.w_hide.copy()
        for layer in range(self.hide_layer_amount):
            if layer != 0:
                self.w_hide_update[layer, :, :] = self.delta_hidden[layer, :] * self.a_hide[layer-1].reshape(-1, 1)
            else:
                self.w_hide_update[layer, :, :] = self.delta_hidden[layer, :] * self.x_vector.reshape(-1, 1)
        self.w_output_update = self.delta_output * self.a_hide[-1, :]

        for layer in range(self.hide_layer_amount):
            for i in range(self.neuron_amount):
                for j in range(self.neuron_amount):
                    self.w_hide[layer, i, j] -= alpha * self.w_hide_update[layer, i, j]
                self.b_hide[layer, i] -= alpha * self.delta_hidden[layer, i]

        self.w_output = self.w_output - alpha * self.w_output_update
        self.b_output = self.b_output - alpha * self.delta_output

    def _fit(self, iter_times, alpha):
        """
        随机梯度下降求解
        :param iter_times: 迭代次数
        :param alpha:      更新步长
        :return:
        """
        for i in range(iter_times):
            rand_int = np.random.randint(0, self.x.shape[0])
            self.x_vector = self.x[rand_int]
            self.y_vector = self.y[rand_int]

            self._forward_transfer()
            self._cal_output_error()
            self._cal_hidden_error()
            self._update_param(alpha)

    def predict(self, x):
        if self.b_output is None:
            raise ModelNotFittedError

        return np.array(list(map(self._predict_single, x)))

    def _predict_single(self, x):
        z_hide_new = self.z_hide.copy()
        a_hide_new = self.a_hide.copy()
        for layer_id in range(self.hide_layer_amount):
            for neuron_id in range(self.neuron_amount):
                if layer_id == 0:
                    z_hide_new[layer_id, neuron_id] = np.dot(self.w_hide[layer_id, :, neuron_id], x) \
                                                      + self.b_hide[layer_id, neuron_id]
                else:
                    z_hide_new[layer_id, neuron_id] = np.dot(self.w_hide[layer_id, :, neuron_id],
                             a_hide_new[layer_id-1, :]) + self.b_hide[layer_id, neuron_id]
                a_hide_new[layer_id, neuron_id] = self._sigmoid(z_hide_new[layer_id, neuron_id])

        z_output_new = np.dot(self.w_output, a_hide_new[-1, :]) + self.b_output
        a_output_new = self._sigmoid(z_output_new)
        return a_output_new

    def score(self, x, y):
        y_predict = self.predict(x)
        if self.label_type == LabelType.binary:
            return classify_f1(y_predict, y)
        else:
            return classify_f1_macro(y_predict, y)
