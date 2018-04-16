from distutils.core import setup


setup(
    name="simple_ml",
    version="1.0",
    author="Ray Yang",
    author_email="yangruipis@163.com",
    description=("A simple machine learning algorithm implementation"),
    license="MIT",

    url="https://github.com/Yangruipis/simple_ml",
    packages=['simple_ml', 'simple_ml.base'],
    package_dir = {'': 'lib'},
    install_requires=['numpy>=1.10', 'setuptools>=16.0', 'matplotlib>2.0.0'],


)