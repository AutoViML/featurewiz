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
    print("""Running featurewiz: Auto_ViML's feature engg and selection library. Version=%s
output_tuple = featurewiz(dataname, target, corr_limit=0.70,
                    verbose=2, sep=',', header=0, test_data='',
                    feature_engg='', category_encoders='')
Let featurewiz add features! Set feature_engg as: 'interactions' or 'groupby' or 'target'
Instead, you can also choose your own category_encoders from list below:
['HashingEncoder', 'SumEncoder', 'PolynomialEncoder', 'BackwardDifferenceEncoder',
'OneHotEncoder', 'HelmertEncoder', 'OrdinalEncoder', 'FrequencyEncoder', 'BaseNEncoder',
'TargetEncoder', 'CatBoostEncoder', 'WOEEncoder', 'JamesSteinEncoder']
                                """ %version_number)
else:
    version_number = __version__
    print("""Imported featurewiz: Auto_ViML's feature engg and selection library. Version=%s
output_tuple = featurewiz(dataname, target, corr_limit=0.70,
                    verbose=2,  sep=',', header=0, test_data='',
                    feature_engg='', category_encoders='')
Let featurewiz add features! Set feature_engg as: 'interactions' or 'groupby' or 'target'
Instead, you can also choose your own category_encoders from list below:
['HashingEncoder', 'SumEncoder', 'PolynomialEncoder', 'BackwardDifferenceEncoder',
'OneHotEncoder', 'HelmertEncoder', 'OrdinalEncoder', 'FrequencyEncoder', 'BaseNEncoder',
'TargetEncoder', 'CatBoostEncoder', 'WOEEncoder', 'JamesSteinEncoder']
""" %version_number)
################################################################################
