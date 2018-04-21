# -*- coding:utf-8 -*-

import unittest
from simple_ml.data_handle import *
import numpy as np
from simple_ml.base.base_enum import LabelType, ConMissingHandle, DisMissingHandle


class TestDataHandle(unittest.TestCase):

    def test_read_string(self):

        a = "1,2.0,3 \n 0,3.1,否 \n 1,0,2"
        res = read_string(a, header=False, index=False)
        self.assertEqual(len(res), 3)
        self.assertEqual(len(res[0]), 3)
        for i in range(3):
            self.assertIsInstance(res[i][0], int)
            self.assertIsInstance(res[i][1], float)
            self.assertIsInstance(res[i][2], str)

    def test_encoding(self):
        inp = [[1, 2.0, '1'], [0, np.nan, '2'], [1, 2.0, '1']]
        res = number_encoder(inp)
        self.assertEqual(len(res), 3)
        self.assertEqual(len(res[0]), 3)
        for i in range(3):
            for j in range(3):
                self.assertIsInstance(res[i][j], float)

    def test_get_type(self):
        np.random.seed(918)
        a1 = np.random.rand(10)
        a2 = np.random.choice([1, 2], 10, replace=True)
        a3 = np.random.choice([3, 4, 5], 10, replace=True)
        arr = np.column_stack((a1, a2, a3))
        res = get_type(arr)
        self.assertEqual(res[0], LabelType.continuous)
        self.assertEqual(res[1], LabelType.binary)
        self.assertEqual(res[2], LabelType.multi_class)

    def test_one_hot_encoder(self):
        np.random.seed(918)
        a1 = np.random.rand(10)
        a2 = np.random.choice([1, 2], 10, replace=True)
        a3 = np.random.choice([3, 4, 5], 10, replace=True)
        arr = np.column_stack((a1, a2, a3))
        types = get_type(arr)
        res = one_hot_encoder(arr, types)
        self.assertEqual(res.shape[0], 10)
        self.assertEqual(res.shape[1], 4)

    def test_missing_value_handle(self):
        np.random.seed(918)
        a1 = np.random.rand(10)
        a2 = np.random.choice([1, 2], 10, replace=True)
        a3 = np.random.choice([3, 4, 5], 10, replace=True)
        arr = np.column_stack((a1, a2, a3))
        arr[3, 0] = np.nan
        arr[4, 1] = np.nan
        arr[6, 2] = np.nan
        types = get_type(arr)
        res = missing_value_handle(arr, types)
        self.assertIsInstance(res[3, 0], float)
        self.assertIsInstance(arr[4, 1], float)
        self.assertIsInstance(arr[6, 2], float)
        res = missing_value_handle(arr, types, continuous_method=ConMissingHandle.sample_drop,
                                   discrete_method=DisMissingHandle.sample_drop)
        self.assertEqual(res.shape, (7, 3))

    def test_abnormal_handle(self):
        np.random.seed(918)
        a1 = np.random.rand(50)
        a2 = np.random.choice([1, 2], 50, replace=True)
        a3 = np.random.choice([3, 4, 5], 50, replace=True)
        arr = np.column_stack((a1, a2, a3))
        arr[3, 0] = np.nan
        arr[4, 0] = 100
        arr[6, 0] = -100
        types = get_type(arr)
        res = abnormal_handle(arr, types)
        self.assertLess(res[4, 0], 100)
        self.assertLess(-100, res[6, 0])


if __name__ == '__main__':
    unittest.main()
