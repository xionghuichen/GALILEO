from distutils.core import setup
from setuptools import find_packages

setup(
    name='galileo',
    packages=find_packages(),
    version='0.0.1',
    description='Adversarial Counterfactual Environment Model Learning',
    long_description=open('./README.md').read(),
    author='Xiong-Hui Chen',
    author_email='chenxh@lamda.nju.edu.cn',
    zip_safe=True,
    license='MIT'
)
