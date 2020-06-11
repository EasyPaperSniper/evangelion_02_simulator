#!/usr/bin/env python

import sys

for arg in sys.argv:
    if arg in ('upload', 'register', 'testarg'):
        print('This package is not meant to be uploaded.')
        sys.exit(-1)

from distutils.core import setup
from setuptools import find_packages
print(find_packages())

install_requires = [
        'torch',
        'pybullet',
        'gym',
        'xmltodict',
        'numpy',
        'opencv-python',
        'tensorboard',
        'openvr',
        'deepdish',
        'transformations',
        'matplotlib',
        'ax-platform==0.1.9',
        'cma'
      ]

setup(name='unitree_robotics_env',
      packages=find_packages(),
      version='0.0.1',
      description='Simulation and Hardware environments for A1 and Aliengo by Unitree Robotics.',
      url='NONE',
      author='Tianyu EasyPaperSniper Li',
      author_email='bzdlity@gmail.com',
      maintainer='Tianyu Li',
      maintainer_email='tianyul@fb.com',
      install_requires=install_requires,
      include_package_data=True,
      zip_safe=False,
     )