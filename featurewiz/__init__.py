# -*- coding: utf-8 -*-
################################################################################
#     featurewiz - advanced feature engineering and best features selection in single line of code
#     Python v3.6+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# Version
from .__version__ import __version__
from .featurewiz import featurewiz
from .featurewiz import FE_split_one_field_into_many, FE_add_groupby_features_aggregated_to_dataframe
from .featurewiz import FE_start_end_date_time_features
from .featurewiz import classify_features
from .featurewiz import classify_columns,FE_combine_rare_categories
from .featurewiz import FE_count_rows_for_all_columns_by_group
from .featurewiz import FE_add_age_by_date_col, FE_split_add_column, FE_get_latest_values_based_on_date_column
from .featurewiz import FE_capping_outliers_beyond_IQR_Range
from .featurewiz import EDA_classify_and_return_cols_by_type, EDA_classify_features_for_deep_learning
from .featurewiz import FE_create_categorical_feature_crosses, EDA_find_skewed_variables
from .featurewiz import FE_kmeans_resampler, FE_find_and_cap_outliers, EDA_find_outliers
from .featurewiz import split_data_n_ways, FE_concatenate_multiple_columns
from .featurewiz import FE_discretize_numeric_variables, reduce_mem_usage
from .ml_models import simple_XGBoost_model, simple_LightGBM_model, complex_XGBoost_model, complex_LightGBM_model,data_transform
from .my_encoders import My_LabelEncoder, Groupby_Aggregator, My_LabelEncoder_Pipe, Ranking_Aggregator, DateTime_Transformer
from .my_encoders import Rare_Class_Combiner, Rare_Class_Combiner_Pipe, FE_create_time_series_features, Binning_Transformer
from .my_encoders import Column_Names_Transformer, FE_convert_all_object_columns_to_numeric, Numeric_Transformer
from .my_encoders import TS_Lagging_Transformer, TS_Fourier_Transformer, TS_Trend_Seasonality_Transformer
from .my_encoders import TS_Lagging_Transformer_Pipe, TS_Fourier_Transformer_Pipe

from .sulov_method import FE_remove_variables_using_SULOV_method
from .featurewiz import FE_transform_numeric_columns_to_bins, FE_create_interaction_vars
from .stacking_models import Stacking_Classifier, Blending_Regressor, Stacking_Regressor, stacking_models_list
from .featurewiz import EDA_binning_numeric_column_displaying_bins, FE_calculate_duration_from_timestamp
from .featurewiz import FE_convert_mixed_datatypes_to_string, FE_drop_rows_with_infinity
from .featurewiz import EDA_find_remove_columns_with_infinity, FE_split_list_into_columns
from .featurewiz import EDA_remove_special_chars, FE_remove_commas_in_numerics
from .featurewiz import EDA_randomly_select_rows_from_dataframe, remove_duplicate_cols_in_dataset

from .featurewiz import FeatureWiz
################################################################################
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""%s %s version. Select nrows to a small number when running on huge datasets.
output = featurewiz(dataname, target, corr_limit=0.90, verbose=2, sep=',', 
		header=0, test_data='',feature_engg='', category_encoders='',
		dask_xgboost_flag=False, nrows=None, skip_sulov=False, skip_xgboost=False)
Create new features via 'feature_engg' flag : ['interactions','groupby','target']
""" %(module_type, version_number))
################################################################################
