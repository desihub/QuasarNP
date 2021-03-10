from setuptools import setup

setup(
    name='quasarnp',
    version='0.1.0',
    description='Numpy Implementation of QuasarNet',
    url='https://github.com/dylanagreen/QuasarNP',
    author='Dylan Green',
    author_email='dylanag@uci.edu',
    license='MIT',
    install_requires=['numpy', 'scipy', 'fitsio', 'h5py'],
    zip_safe=False,
)