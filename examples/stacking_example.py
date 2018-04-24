# -*- coding:utf-8 -*-

from simple_ml.classify_data import *
from simple_ml.data_handle import train_test_split
from simple_ml import *

def wine_example():
    x, y = get_wine()
    #x = x[(y == 2) | (y == 1)]
    #y = y[(y == 2) | (y == 1)]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3, 91)

    model_list = [LogisticRegression(), NaiveBayes()]     # , SVM(kernel_type=KernelTbbype.gaussian, sigma=1)]

    stack = Stacking(model_list, k_folder=5)
    stack.fit(x_train, y_train)
    print(stack.score(x_test, y_test))
    print(stack.score_mat)
    stack.classify_plot(x_test, y_test)

if __name__ == '__main__':
    wine_example()