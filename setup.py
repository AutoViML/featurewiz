#!/usr/bin/env python

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="featurewiz",
    version="0.5.5",
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
        "xgboost>=1.5,<=1.6.2",
        "pandas>=1.3.4,<2.0",
        "matplotlib",
        "seaborn",
        "scikit-learn>=0.24,<=1.2.2",
        "networkx>=2.6.2",
        "category_encoders>=2.6.2",
        "xlrd>=2.0.0",
        "tqdm>=4.61.1",
        "dask>=2021.11.0",
        "lightgbm>=3.2.1",
        "distributed>=2021.11.0",
        "feather-format>=0.4.1",
        "pyarrow>=7.0.0",
        "fsspec>=0.3.3",
        "scipy<1.11.0",
        "Pillow>=9.0.0",
        "lazytransform>=1.14",
        "tensorflow>=2.5",
        "numexpr>=2.7.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
