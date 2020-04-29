import sys

from setuptools import setup

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(name='rl_exp',
    install_requires=[
    ],
    description="rl_exp",
    author="Alexander Penkin",
    url='https://github.com/SSS135/rl_exp',
    author_email="sss13594@gmail.com",
    version="0.1",
    packages=['rl_exp'],
    zip_safe=False)
