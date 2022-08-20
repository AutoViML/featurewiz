#!/usr/bin/env python

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="featurewiz",
    version="0.1.96",
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
        "xgboost>=1.5.1",
        "pandas>=1.3.4",
        "matplotlib",
        "seaborn",
        "scikit-learn~=0.24",
        "networkx>=2.8.1",
        "category_encoders>=2.4.0",
        "xlrd>=2.0.0",
        "imbalanced-learn>=0.7",
        "tqdm>=4.62.0",
        "dask>=2021.11.0",
        "lightgbm>=3.2.1",
        "distributed>=2021.11.0",
        "feather-format>=0.4.1",
        "pyarrow==7.0.0",
        "fsspec>=0.3.3",
        "imbalanced-ensemble>=0.1.7",
        #"lazytransform>=0.78",
        "Pillow~=9.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
