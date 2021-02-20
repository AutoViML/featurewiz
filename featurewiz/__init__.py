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
from .featurewiz import fe_create_time_series_features
################################################################################
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""%s featurewiz: Auto_ViML's feature engg and selection library. Version=%s
output = featurewiz(dataname, target, corr_limit=0.70,
                    verbose=2, sep=',', header=0, test_data='',
                    feature_engg='', category_encoders='')
Let featurewiz add features to your data! Set 'feature_engg' as: 'interactions' or 'groupby' or 'target'
                                """ %(module_type, version_number))
################################################################################
