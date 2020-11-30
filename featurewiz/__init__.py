# -*- coding: utf-8 -*-
################################################################################
#     featurewiz - fast feature selection using one line of code
#     Python v3.6+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# Version
from .__version__ import __version__
from .featurewiz import featurewiz

if __name__ == "__main__":
    version_number = __version__
    print("""Running featurewiz version: %s. Call using:
                 features = featurewiz(dataname, target, corr_limit=0.70,
                                verbose=2, sep=',', header=0)""" %version_number)
else:
    version_number = __version__
    print("""Imported featurewiz version: %s. Call using:
                 features = featurewiz(dataname, target, corr_limit=0.70,
                                verbose=2,  sep=',', header=0)""" %version_number)
################################################################################
