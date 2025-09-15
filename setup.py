import re
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
    version=_get_version(),
    packages=find_packages(),
)

setup(**setup_keywords)