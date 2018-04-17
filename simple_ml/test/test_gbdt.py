# -*- coding:utf-8 -*-

import unittest
from simple_ml.ensemble import CARTForGBDT, GBDT
from simple_ml.base.base_error import *
import numpy as np


class TESTGBDT(unittest.TestCase):

    def test_cart_importance(self):
        np.random.seed(918)
        x = np.random.rand(11, 5)
        y = np.random.rand(10)
        cart = CARTForGBDT()
        self.assertRaises(FeatureNumberMismatchError, cart.fit, x, y)

        x = np.random.choice([0, 1], 50).reshape(10, 5)
        cart.fit(x, y)
        self.assertIsNotNone(cart.importance)
        self.assertNotEqual(0, cart.importance.any())
        self.assertIsNotNone(cart.leaf_node_list)

    def test_GBDT_fit(self):
        np.random.seed(918)
        x = np.random.choice([0, 1], 50).reshape(10, 5)
        y = np.random.rand(10)
        gbdt = GBDT()
        gbdt.fit(x, y)
        predict = gbdt.predict(x)
        self.assertEqual(len(predict), len(y))

    def test_GBDT_importance(self):
        np.random.seed(918)
        x = np.array([[1, 1, 0],
                      [1, 1, 0],
                      [0, 1, 0],
                      [0, 1, 1]])
        y = np.array([0.85, 0.91, 0.2, 0.15])
        gbdt = GBDT()
        gbdt.fit(x, y)
        selected = gbdt.feature_select(1)
        self.assertEqual(len(selected), 1)
        self.assertNotEqual(selected[0], 0)


if __name__ == '__main__':
    unittest.main()
