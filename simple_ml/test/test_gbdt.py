# -*- coding:utf-8 -*-

import unittest
from simple_ml.ensemble import CART, BaseGBDT
import numpy as np
from simple_ml.base.base_error import *


class TESTGBDT(unittest.TestCase):

    def assertArrayAlmostEqual(self, y1, y2):
        if len(y1) != len(y2):
            raise self.failureException("Not The Same Length")
        count = 0
        for i, y in enumerate(y1):
            if y == y2[i] or abs(y - y2[i]) <= 0.1*y:
                count += 1
        if count < 2*len(y1)//3:
            raise self.failureException("Array Not Almost Equal")

    def test_cart_importance(self):
        np.random.seed(918)
        x = np.random.rand(10, 5)
        y = np.random.rand(10)
        cart = CART()
        self.assertRaises(FeatureTypeError, cart.fit, x, y)

        x = np.random.choice([0, 1], 50).reshape(10, 5)
        cart.fit(x, y)
        self.assertIsNotNone(cart.importance)
        self.assertNotEqual(0, cart.importance.any())
        self.assertIsNotNone(cart.leaf_node_list)

    def test_GBDT_fit(self):
        np.random.seed(918)
        x = np.random.choice([0, 1], 50).reshape(10, 5)
        y = np.random.rand(10)
        gbdt = BaseGBDT()
        gbdt.fit(x, y)
        predict = gbdt.predict(x)
        self.assertArrayAlmostEqual(predict, y)

    def test_GBDT_importance(self):
        np.random.seed(918)
        x = np.array([[1, 1, 0],
                      [1, 1, 0],
                      [0, 1, 0],
                      [0, 1, 1]])
        y = np.array([0.85, 0.91, 0.2, 0.15])
        gbdt = BaseGBDT()
        gbdt.fit(x, y)
        selected = gbdt.feature_select(1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0], 0)


if __name__ == '__main__':
    unittest.main()
