#!/usr/bin/env python

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="featurewiz",
    version="0.0.55",
    author="Ram Seshadri",
    author_email="rsesha2001@yahoo.com",
    description="Select Best Features from your data set - any size - now with XGBoost!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/featurewiz",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "ipython",
        "jupyter",
        "xgboost==0.90",
        "pandas>=1.3.4",
        "matplotlib",
        "seaborn",
        "scikit-learn==0.23.2",
        "networkx",
        "category_encoders",
        "xlrd",
        "imbalanced-learn>=0.7",
        "tqdm",
        "dask==2021.11.0",
        "distributed==2021.11.0",
        "dask-ml==2021.10.17",
        "dask-xgboost==0.2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
