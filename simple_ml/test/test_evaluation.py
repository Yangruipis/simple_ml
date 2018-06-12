# -*- coding:utf-8 -*-

import unittest
from simple_ml.evaluation import *
import numpy as np
from simple_ml.base.base_enum import *
from simple_ml.base.base_error import *

class TestEvaluation(unittest.TestCase):

    def test_regression_plot(self):
        x_train = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9],
                            [10, 11, 12]])
        y_train = np.array([0.1, 0.2, 0.4, 0.6])
        x_test = np.array([[1, 3, 4, 5]])
        y_test = np.array([0.12])
        self.assertRaises(FeatureNumberMismatchError, regression_plot, x_train, y_train, x_test, y_test)

        x_test = np.array([[1,3,4]])
        y_test = np.array([0.12, 0.13])
        self.assertRaises(SampleNumberMismatchError, regression_plot, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    unittest.main()

