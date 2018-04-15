
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def get_iris():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    return x, y


def get_wine():
    wine = datasets.load_wine()
    standard = StandardScaler()

    x = wine.data
    y = wine.target
    x = standard.fit_transform(x)
    return x, y


def get_moon(samples):
    moons = datasets.make_moons(samples)
    return moons[0], moons[1]


if __name__ == '__main__':
    x, y = get_wine()
    print(x, y)