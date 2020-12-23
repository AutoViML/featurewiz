# -*- coding: utf-8 -*-
################################################################################
#     featurewiz - fast feature selection using one line of code
#     Python v3.6+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# Version
from .__version__ import __version__
from .featurewiz import featurewiz, convert_all_object_columns_to_numeric
from .featurewiz import split_one_field_into_many, add_aggregate_primitive_features
from .featurewiz import create_time_series_features
if __name__ == "__main__":
    version_number = __version__
    print("""Running featurewiz version: %s. Call by using:
                 features = featurewiz(dataname, target, corr_limit=0.70,
                                verbose=2, sep=',', header=0,
                                add_features='', cat_encoders='')""" %version_number)
else:
    version_number = __version__
    print("""Imported featurewiz version: %s. Call by using:
                 features = featurewiz(dataname, target, corr_limit=0.70,
                                verbose=2,  sep=',', header=0,
                                add_features='', cat_encoders='')""" %version_number)
################################################################################
