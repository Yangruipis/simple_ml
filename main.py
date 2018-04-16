# -*- coding:utf-8 -*-


from simple_ml.tree import CART
import numpy as np
X = np.array([[1,1.1],
              [1,2.0],
              [0,3.0],
              [0,2.2]])
y = np.array([1,1,0,0])
cart = CART(min_samples_leaf=1)
cart.fit(X, y)
print(cart.predict(np.array([[1,2], [3,4]])))
