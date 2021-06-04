import os, sys, glob, re
from setuptools import setup, find_packages

def _get_version():
    line = open('quasarnp/_version.py').readline().strip()
    m = re.match("__version__\s*=\s*'(.*)'", line)
    if m is None:
        print('ERROR: Unable to parse version from: {}'.format(line))
        version = 'unknown'
    else:
        version = m.groups()[0]

    return version

setup_keywords = dict(
    name='quasarnp',
    version=_get_version(),
    description='Numpy Implementation of QuasarNet',
    url='https://github.com/dylanagreen/QuasarNP',
    author='Dylan Green',
    author_email='dylanag@uci.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'fitsio', 'h5py'],
    zip_safe=False,
)

setup(**setup_keywords)