__version__ = "1.1.2"

from simple_ml.bayes import *
from simple_ml.ensemble import *
from simple_ml.knn import *
from simple_ml.logistic import *
from simple_ml.neural_network import *
from simple_ml.svm import *
from simple_ml.tree import *
from simple_ml.pca import *
from simple_ml.cluster import *
from simple_ml.data_handle import *
from simple_ml.feature_select import *
from simple_ml.evaluation import *
from simple_ml.auto import *


# __all__ = [
#     'NaiveBayes',
#     'BayesMinimumError',
#     'BayesMinimumRisk',
#     'RandomForest',
#     'AdaBoost',
#     'LogisticRegression',
#     'Lasso',
#     'Ridge',
#     'NeuralNetwork',
#     'KNN',
#     'SVM',
#     'ID3',
#     'CART',
#     'GBDT',
#     'PCA',
#     'SuperPCA',
#     'KMeans',
#     'Hierarchical',
# ]


BINARY_CLASSIFY_MODEL = [
    NaiveBayes,
    BayesMinimumError,
    BayesMinimumRisk,
    RandomForest,
    AdaBoost,
    LogisticRegression,
    Lasso,
    Ridge,
    NeuralNetwork,
    KNN,
    SVM,
    ID3,
    CART,
]


MULTI_CLASSIFY_MODEL = [
    NaiveBayes,
    BayesMinimumRisk,
    BayesMinimumError,
    RandomForest,
    KNN,
    CART
]


REGRESSION_MODEL = [
    CART,
    GBDT,
]
