#!/usr/bin/env python

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="featurewiz",
    version="0.0.16",
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
        "xgboost>=1.1.1",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn>=0.23.1",
        "networkx",
        "category_encoders",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
