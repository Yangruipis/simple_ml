# -*- coding:utf-8 -*-

from simple_ml.feature_select import *
from simple_ml.classify_data import get_wine

def wine_example():
    x, y = get_wine()
    x = x[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]

    _filter = Filter(FilterType.corr, 3)
    x_filter = _filter.fit_transform(x, y)
    print(x_filter.shape)

    _filter = Filter(FilterType.var, 3)
    x_filter = _filter.fit_transform(x, y)
    print(x_filter.shape)

    _filter = Filter(FilterType.entropy, 3)
    x_filter = _filter.fit_transform(x, y)
    print(x_filter.shape)

    embedded = Embedded(3, EmbeddedType.Lasso)
    x_embedded = embedded.fit_transform(x, y)
    print(x_embedded.shape) # lasso后稀疏到只有两个值非0，因此只输出了两个特征

    # GBDT暂时只支持离散特征
    embedded = Embedded(3, EmbeddedType.GBDT)
    x = np.random.choice([0, 1], 50).reshape(10, 5)
    y = np.random.rand(10)
    x_embedded = embedded.fit_transform(x, y)
    print(x_embedded, y)


if __name__ == '__main__':
    wine_example()
