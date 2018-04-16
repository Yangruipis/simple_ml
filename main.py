# -*- coding:utf-8 -*-

# from simple_ml.bp_network import *
# from simple_ml.knn import *
# from simple_ml.cluster import *
# from simple_ml.logistic import *
# from simple_ml.naive_bayes import *
# from simple_ml.svm import *
# from simple_ml.tree import *


from simple_ml.knn import *
from simple_ml.classify_data import get_iris

knn_test = KNN(K=3, distance_type=DisType.CosSim)
X, y = get_iris()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

knn_test.fit(X_train, y_train)
print(knn_test.predict(X_test))
print(knn_test.score(X_test, y_test))