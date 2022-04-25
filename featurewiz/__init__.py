# -*- coding: utf-8 -*-
################################################################################
#     featurewiz - advanced feature engineering and best features selection in single line of code
#     Python v3.6+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# Version
from .__version__ import __version__
from .featurewiz import featurewiz, FE_convert_all_object_columns_to_numeric
from .featurewiz import FE_split_one_field_into_many, FE_add_groupby_features_aggregated_to_dataframe
from .featurewiz import FE_create_time_series_features, FE_start_end_date_time_features
from .featurewiz import classify_features
from .featurewiz import classify_columns,FE_combine_rare_categories
from .featurewiz import FE_count_rows_for_all_columns_by_group
from .featurewiz import FE_add_age_by_date_col, FE_split_add_column, FE_get_latest_values_based_on_date_column
from .featurewiz import FE_capping_outliers_beyond_IQR_Range
from .featurewiz import EDA_classify_and_return_cols_by_type, EDA_classify_features_for_deep_learning
from .featurewiz import FE_create_categorical_feature_crosses, EDA_find_skewed_variables
from .featurewiz import FE_kmeans_resampler, FE_find_and_cap_outliers, EDA_find_outliers
from .featurewiz import split_data_n_ways, FE_concatenate_multiple_columns
from .featurewiz import FE_discretize_numeric_variables
from .ml_models import simple_XGBoost_model, simple_LightGBM_model, complex_XGBoost_model, complex_LightGBM_model,data_transform
from .my_encoders import My_LabelEncoder, Groupby_Aggregator, My_LabelEncoder_Pipe
from .my_encoders import Rare_Class_Combiner, Rare_Class_Combiner_Pipe
from .sulov_method import FE_remove_variables_using_SULOV_method
from .featurewiz import FE_transform_numeric_columns, FE_create_interaction_vars
from .stacking_models import Stacking_Classifier, Blending_Regressor, Stacking_Regressor, stacking_models_list
from .featurewiz import EDA_binning_numeric_column_displaying_bins, FE_add_lagged_targets_by_date_category
from .featurewiz import FE_convert_mixed_datatypes_to_string, FE_drop_rows_with_infinity
from .featurewiz import EDA_find_remove_columns_with_infinity, FE_split_list_into_columns
from .featurewiz import FeatureWiz
################################################################################
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""%s version = %s.
from featurewiz import FeatureWiz
wiz = FeatureWiz(verbose=1)
X_train_selected = wiz.fit_transform(X_train, y_train)
X_test_selected = wiz.transform(X_test)
wiz.features  ### provides a list of selected features ###
                                """ %(module_type, version_number))
################################################################################
