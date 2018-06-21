# -*- coding:utf-8 -*-

import unittest
from simple_ml.evaluation import *
from simple_ml.evaluation import _check_input, _get_binary_confusion_matrix, _gen_binary_pairs
import numpy as np
from numpy.testing import assert_array_equal
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
        self.assertRaises(FeatureNumberMismatchError, regression_plot, x_train, y_train, x_test, y_test, y_test)

        x_test = np.array([[1,3,4]])
        y_test = np.array([0.12, 0.13])
        self.assertRaises(SampleNumberMismatchError, regression_plot, x_train, y_train, x_test, y_test, y_test)

    def test_check_input(self):
        y_true = np.array([[1, 2, 3], [4, 5, 6]])
        y_predict = y_true.copy()
        self.assertRaises(InputTypeError, _check_input, y_predict, y_true)
        y_true = np.array([1, 2])
        y_predict = np.array([1, 2, 3])
        self.assertRaises(LabelLengthMismatchError, _check_input, y_predict, y_true)

    def test_confusion_matrix(self):
        y1 = np.array([1, 0, 0, 1])
        y2 = np.array([1, 0, 0, 2])
        self.assertRaises(ParamInputError, _get_binary_confusion_matrix, y1, y2)
        y2 = np.array([1, 0, 0, 0])
        confusion_matrix = _get_binary_confusion_matrix(y1, y2)
        assert_array_equal(confusion_matrix, np.array([[1, 1],
                                                       [0, 2]]))

    def test_classify_accuracy(self):
        y1 = np.array([1, 0, 0, 1])
        y2 = np.array([1, 0, 1, 1])
        score = classify_accuracy(y1, y2)
        self.assertEqual(score, 0.75)

    def test_classify_precision(self):
        y1 = np.array([1, 1, 0, 0, 1])
        y2 = np.array([1, 0, 1, 0, 0])
        score = classify_precision(y1, y2)
        self.assertEqual(score, 1/3)

    def test_classify_recall(self):
        y1 = np.array([1, 1, 0, 0, 1])
        y2 = np.array([1, 0, 1, 0, 0])
        score = classify_recall(y1, y2)
        self.assertEqual(score, 0.5)


if __name__ == '__main__':
    unittest.main()
