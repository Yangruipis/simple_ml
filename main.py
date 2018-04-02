# -*- coding:utf-8 -*-

# from simple_ml.bp_network import *
# from simple_ml.knn import *
# from simple_ml.cluster import *
# from simple_ml.logistic import *
# from simple_ml.naive_bayes import *
# from simple_ml.svm import *
# from simple_ml.tree import *

import numpy as np
from simple_ml.bayes import MyBayesMinimumRisk

X = np.array([[2,1],
             [0,3],
             [3,0],
             [1,2],
             [2,0],
              [0,1.5]])
y = np.array([1,0,1,0,1,0])
bme = MyBayesMinimumRisk(np.array([[0,10,1], [1,0,2]]))
bme.fit(X, y)
print(bme.predict(X))