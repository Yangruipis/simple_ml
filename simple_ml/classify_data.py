
from sklearn import datasets


def get_iris():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    return x, y


def get_moon(samples):
    moons = datasets.make_moons(samples)
    return moons[0], moons[1]