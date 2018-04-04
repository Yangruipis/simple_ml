# -*- coding:utf-8 -*-

# from simple_ml.bp_network import *
# from simple_ml.knn import *
# from simple_ml.cluster import *
# from simple_ml.logistic import *
# from simple_ml.naive_bayes import *
# from simple_ml.svm import *
# from simple_ml.tree import *


from simple_ml.pca import *

pca = SuperPCA(1)
a = np.array([[1,3,2], [3,5,1], [4,7,3], [1,2,0], [0,2,1]])
print(pca.fit_transform(a.T))
print(pca.explain_ratio)

