# -*- coding:utf-8 -*-

from .base import *
from .base_enum import KernelType
from .classify_plot import classify_plot
from .score import *
from .base_error import KernelTypeError, KernelMissParameterError, FeatureNumberMismatchError

class MySVM(MyClassifier):

    def __init__(self, c, tol, precision, max_iter, kernel_type, **kwargs):
        """
        可变参数kwargs存储核函数的参数
        param:
            C           软间隔支持向量机参数（越大越迫使所有样本满足约束）
            tol         误差容忍度（越大越不准确，但是省时间）
            precision   alpha结果精度
            max_iter    外循环最大迭代次数
            KERNEL_type 核函数类型:
                        linear（无需提供参数，相当于没有用核函数）
                        polynomial(需提供参数：d)
                        gassian(需提供参数：sigma)
                        laplace(需提供参数：sigma)
                        sigmoid(需提供参数：beta, theta)
        """
        super(MySVM, self).__init__()
        self.c = c
        self.tol = tol
        self.precision = precision
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self._check_kernel(kwargs)       # 检查核函数是否提供了对应参数，并存入self
        self.b = 0                       # 超平面偏移项
        self.error = None
        self.alphas = None
        self.kernel_mat = None

    def _check_kernel(self, kwargs_dict):
        if self.kernel_type == KernelType.gassian or self.kernel_type == KernelType.laplace:
            if 'sigma' in kwargs_dict:
                raise KernelMissParameterError("高斯核或拉普拉斯核必须申明带宽sigma参数, sigma>0")
            self.sigma = kwargs_dict['sigma']
        elif self.kernel_type == KernelType.polynomial:
            if 'd' in kwargs_dict:
                raise KernelMissParameterError("多项式核必须申明幂指数d参数, d>=1")
            self.d = kwargs_dict['d']
        elif self.kernel_type == KernelType.sigmoid:
            if 'beta' in kwargs_dict:
                raise KernelMissParameterError("sigmoid核必须申明beta参数, beta>0")
            if 'theta' in kwargs_dict:
                raise KernelMissParameterError("sigmoid核必须申明theta参数, theta<0")
            self.beta = kwargs_dict['beta']
            self.theta = kwargs_dict['theta']
        else:
            raise KernelTypeError

    def clear(self):
        self.b = 0
        # 误差缓存表N*2，第一列为更新状态（0-未更新，1-已更新），第二列为缓存值
        self.error = np.zeros((self.sample_num, 2))
        # 对偶问题所需优化的目标变量向量
        self.alphas = np.zeros(self.sample_num)
        self.kernel_mat = None

    def fit(self, x, y):
        """
        两种求解方法：
            1. SOM（本文直接实现该方法）
            2. 二次优化求解QP方法（调用python cvxopt包，参考http://tullo.ch/articles/svm-py/）
        """
        self._init(x, y)
        self.clear()
        # self._clear()
        # 事先计算出核函数矩阵，避免高维下的计算问题
        self.kernel_mat = self._cal_kernel_matrix()
        entire_set = True
        self._smo_outer(entire_set)

    def _cal_kernel_matrix(self):
        if self.kernel_type == KernelType.linear:
            return np.dot(self.x, self.x.T)
        elif self.kernel_type == KernelType.polynomial:
            return np.dot(self.x, self.x.T)**self.d
        elif self.kernel_type == KernelType.sigmoid:
            return np.tanh(self.beta * np.dot(self.x, self.x.T) + self.theta)

        if self.kernel_type == KernelType.gassian:
            kernel_func = lambda x_i, x_j: np.exp(- np.sum((x_i - x_j)**2) / (2 * self.sigma**2))
        elif self.kernel_type == KernelType.laplace:
            kernel_func = lambda x_i, x_j: np.exp(- np.sqrt(np.sum((x_i - x_j)**2)) / self.sigma)
        else:
            return

        kernel_mat = np.zeros((self.sample_num, self.sample_num))
        for i in range(self.sample_num):
            for j in range(i, self.sample_num):
                kernel_mat[i, j] = kernel_func(self.x[i], self.x[j])
                if i!=j: kernel_mat[j, i] = kernel_mat[i,j]
        return kernel_mat

    def _selected_alpha_j(self, i, error_i):
        """
        SOM 算法
        根据定下来的alpha_i, 选择最合适的alpha_j，使得违背KKT的程度最大，从而优化目标函数值变动幅度最大，更快的收敛

        - 如果是第一次选择，则随机抽取一个样本
        - 如果不是第一次选择，则通过迭代，找到|error_i - error_j|最大的样本

        """
        self.error[i] = [1, error_i]
        candidates = list(np.nonzero(self.error[:, 0]))[0]

        iter_i = i
        iter_error = error_i
        max_diff = 0

        if len(candidates) > 1:
            for candidate in candidates:
                if candidate != i:
                    error_candidate = self._cal_error(candidate)
                    if abs(error_candidate - iter_error) > max_diff:
                        max_diff = abs(error_candidate - iter_error)
                        iter_i = candidate
                        iter_error = error_candidate
            return iter_i, iter_error
        else:
            to_choose = list(range(self.sample_num))
            to_choose.remove(i)
            np.random.seed(918)
            j = np.random.choice(to_choose)
            error_j = self._cal_error(j)
            return j, error_j

    def _cal_error(self, i):
        """
        计算残差：
        Error_i = f(x_i) - y_i
        y_i * Error_i = y_i * f(x_i) - y_i^2
                      = y_i * f(x_i) - 1
        f(x_i) = w^T x_i + b
               = sum_{j=1}^N alpha_j * y_j * kernelMat[:, i][j]
        """
        f_x_i = np.dot(np.multiply(self.alphas, self.y),  self.kernel_mat[:, i]) + self.b    # 记住你这边出错了，当时忘记加b
        error_i = f_x_i - self.y[i]
        return error_i

    def _update_error(self, i):
        error = self._cal_error(i)
        self.error[i] = [1, error]

    def _smo_inner(self, i):
        """
        ref: 刘亮亮ppt
             http://blog.csdn.net/zouxy09/article/details/17292011
             http://jonchar.net/notebooks/SVM/

        开始进行迭代，注意要尽量打破KKT条件，从而更快的收敛
        根据软间隔向量机的KKT条件，以及y_i * Error_i = y_i * f(x_i) - y_i^2 = y_i * f(x_i) - 1，可以得到：
        1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
        2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
        3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
        """
        error_i = self._cal_error(i)

        if (self.y[i] * error_i < -self.tol) and self.alphas[i] < self.c \
           or (self.y[i] * error_i > self.tol) and self.alphas[i] > 0:

            # step1. 选择合适的J
            j, error_j = self._selected_alpha_j(i, error_i)
            alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

            # step2. 计算上界和下界
            if self.y[i] != self.y[j]:
                low = max(0, self.alphas[j] - self.alphas[i])
                high = min(self.c, self.c + self.alphas[j] - self.alphas[i])
            else:
                low = max(0, self.alphas[i] + self.alphas[j] - self.c)
                high = min(self.c, self.alphas[i] + self.alphas[j])
            if low == high:
                return 0

            # step3. 计算i和j的相似度eta
            eta = -(self.kernel_mat[i, i] + self.kernel_mat[j, j] - 2*self.kernel_mat[i, j])
            if eta >= 0:
                return 0

            # step4. 根据eta更新j（这一步推导很复杂，主要根据替换alpha_i后的拉格朗日函数求导得到，见同一文件夹下的pdf）
            self.alphas[j] += self.y[j]*(error_j - error_i) / eta

            # step5. 对上下边界约束进行修剪（最优解必须在方框内的两条直线上取得）
            if self.alphas[j] > high:
                self.alphas[j] = high
            if self.alphas[j] < low:
                self.alphas[j] = low

            # step6. 收敛到一定精度后退出
            if abs(self.alphas[j] - alpha_j_old) <= self.precision:
                self._update_error(j)
                return 0

            # step7. 更新第alpha_i
            self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

            # step8. 更新b值，公式推导见刘亮亮ppt
            b1 = self.b - error_i - self.y[i]*(self.alphas[i] - alpha_i_old)*self.kernel_mat[i, i] - \
                      self.y[j]*(self.alphas[j] - alpha_j_old)*self.kernel_mat[i, j]

            b2 = self.b - error_j - self.y[i]*(self.alphas[i] - alpha_i_old)*self.kernel_mat[i, j] - \
                      self.y[j]*(self.alphas[j] - alpha_j_old)*self.kernel_mat[j, j]

            if 0 < self.alphas[i] < self.c:
                self.b = b1
            elif 0 < self.alphas[j] < self.c:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2
            self._update_error(j)
            self._update_error(i)
            return 1
        return 0

    def _smo_outer(self, entire_set=True):
        iter_count = 0
        alpha_pair_count = 0
        # 跳出条件：
        #       1. 达到最大迭代次数
        #       2. alpha不在变动，此时全部满足KKT条件
        while (iter_count < self.max_iter) and (alpha_pair_count > 0 or entire_set):
            alpha_pair_count = 0
            if entire_set:
                for i in range(self.sample_num):
                    alpha_pair_count += self._smo_inner(i)
                print('---iter:%d entire set, alpha pairs changed:%d' % (iter_count, alpha_pair_count))
                iter_count += 1
            else:
                nonbound_alphas = list(np.nonzero((self.alphas > 0) * (self.alphas < self.c)))[0]
                for i in nonbound_alphas:
                    alpha_pair_count += self._smo_inner(i)
                print('---iter:%d non boundary, alpha pairs changed:%d' % (iter_count, alpha_pair_count))
                iter_count += 1

            if entire_set:
                entire_set = False
            elif alpha_pair_count == 0:
                entire_set = True

        print('Finished')

    def predict(self, x):
        if self.error is None:
            raise ModelNotFittedError

        # step1. 找到支持向量对应的坐标
        support_vector_index = list(np.nonzero(self.alphas > 0))[0]

        # step2. 获取支持向量
        support_vector_x = self.x[support_vector_index]
        support_vector_y = self.y[support_vector_index]
        support_vector_alphas = self.alphas[support_vector_index]

        # step3. 计算预测值向量
        pred_y_value = np.array(list(map(lambda i: self._predict_single(i, support_vector_x, support_vector_y, support_vector_alphas), x)))
        pred_y_binary = np.array([np.sign(i) for i in pred_y_value])
        return pred_y_binary

    def _predict_single(self, x, support_vector_x, support_vector_y, support_vector_alphas):
        kernel_vector = self._cal_kernel_vector(support_vector_x, x, self.kernel_type)
        return np.dot(kernel_vector, np.multiply(support_vector_y, support_vector_alphas)) + self.b

    def _cal_kernel_vector(self, x_mat, x_vector, kernel_type):
        """
        根据输入的矩阵和行向量，计算kernel值
        """
        if x_mat.shape[1] == len(x_vector):
            raise FeatureNumberMismatchError
        if kernel_type == KernelType.linear:
            return np.dot(x_mat, x_vector.reshape(-1, 1)).ravel()
        elif kernel_type == KernelType.polynomial:
            return np.dot(x_mat, x_vector.reshape(-1, 1)).ravel()**self.d
        elif kernel_type == KernelType.sigmoid:
            return np.tanh(self.beta * np.dot(x_mat, x_vector.reshape(-1, 1)).ravel() + self.theta)

        if kernel_type == KernelType.gassian:
            kernel_func = lambda x_i, x_j: np.exp(- np.sum((x_i - x_j)**2) / (2 * self.sigma**2))
        elif kernel_type == KernelType.laplace:
            kernel_func = lambda x_i, x_j: np.exp(- np.sqrt(np.sum((x_i - x_j)**2)) / self.sigma)
        else:
            return

        kernel_vector = np.zeros(x_mat.shape[0])
        for i in range(x_mat.shape[0]):
            kernel_vector[i] = kernel_func(x_mat[i], x_vector)
        return kernel_vector

    def score(self, x, y):
        y_predict = self.predict(x)
        y_predict_binary = np.array([0 if i == -1 else i for i in y_predict])
        y_true_binary = np.array([0 if i == -1 else i for i in self.y])
        return classify_f1(y_predict_binary, y_true_binary)

    def classify_plot(self, x, y):
        classify_plot(self, self.x, self.y, x, y, title='My Support Vector Machine')
