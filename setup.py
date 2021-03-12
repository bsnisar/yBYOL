import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = [l for l in f.read().splitlines() if not l.startswith('--index')]

setup(
    name='parastash',
    packages=find_packages(exclude=['prepare']),
    version='0.0.6',
    author='bsnisar',
    author_email='bogdan.sns@gmail.com',
    url='https://github.com/bsnisar/parastash',
    install_requires=required,
)
