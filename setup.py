"""
setup with
python setup.py install
"""

from setuptools import setup

setup(
    name='gptree',
    version='0.0',
    install_requires=[
        'numpy',
        'scikit-learn',
        'binarytree',
        'typing',
        'tqdm',
    ]
)