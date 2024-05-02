##############################################################################
#Copyright 2019 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#################################################################################
##### This project is not an official Google project. It is not supported by ####
##### Google and Google specifically disclaims all warranties as to its quality,#
##### merchantability, or fitness for a particular purpose.  ####################
#################################################################################
import numpy as np
np.random.seed(99)
import random
random.seed(42)
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
###########   This is from category_encoders Library ################################################
from category_encoders import HashingEncoder, SumEncoder, PolynomialEncoder, BackwardDifferenceEncoder
from category_encoders import OneHotEncoder, HelmertEncoder, OrdinalEncoder, CountEncoder, BaseNEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder, JamesSteinEncoder
from category_encoders.glmm import GLMMEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.wrapper import PolynomialWrapper
from .encoders import FrequencyEncoder
from .sulov_method import FE_remove_variables_using_SULOV_method
from .classify_method import classify_columns, EDA_find_remove_columns_with_infinity
from .ml_models import analyze_problem_type, get_sample_weight_array, check_if_GPU_exists
from .my_encoders import Groupby_Aggregator, My_LabelEncoder_Pipe, My_LabelEncoder
from .my_encoders import Rare_Class_Combiner, Rare_Class_Combiner_Pipe, FE_create_time_series_features
from .my_encoders import Column_Names_Transformer
from .auto_encoders import DenoisingAutoEncoder, VariationalAutoEncoder, GANAugmenter, GAN
from .auto_encoders import dae_hyperparam_selection, vae_hyperparam_selection, CNNAutoEncoder
from .stacking_models import get_class_distribution

from . import settings
settings.init()
################################################################################
#### The warnings from Sklearn are so annoying that I have to shut it off #######
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
################################################################################
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import logging
####################################################################################
import re
import pdb
import pprint
from itertools import cycle, combinations
from collections import defaultdict, OrderedDict
import time
import sys
import xlrd
import statsmodels
from io import BytesIO
import base64
from functools import reduce
import copy
import dask
import dask.dataframe as dd
#import dask_xgboost
import xgboost
from dask.distributed import Client, progress
import psutil
import json
from sklearn.model_selection import train_test_split
from .my_encoders import FE_convert_all_object_columns_to_numeric
#######################################################################################################
def classify_features(dfte, depVar, verbose=0):
    dfte = copy.deepcopy(dfte)
    if isinstance(depVar, list):
        orig_preds = [x for x in list(dfte) if x not in depVar]
    else:
        orig_preds = [x for x in list(dfte) if x not in [depVar]]
    #################    CLASSIFY  COLUMNS   HERE    ######################
    var_df = classify_columns(dfte[orig_preds], verbose)
    #####       Classify Columns   ################
    IDcols = var_df['id_vars']
    discrete_string_vars = var_df['nlp_vars']+var_df['discrete_string_vars']
    cols_delete = var_df['cols_delete']
    bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
    int_vars = var_df['int_vars']
    categorical_vars = var_df['cat_vars'] + var_df['factor_vars'] + int_vars + bool_vars
    date_vars = var_df['date_vars']
    if len(var_df['continuous_vars'])==0 and len(int_vars)>0:
        continuous_vars = var_df['int_vars']
        categorical_vars = left_subtract(categorical_vars, int_vars)
        int_vars = []
    else:
        continuous_vars = var_df['continuous_vars']
    preds = [x for x in orig_preds if x not in IDcols+cols_delete+discrete_string_vars]
    if len(IDcols+cols_delete+discrete_string_vars) == 0:
        print('        No variables were removed since no ID or low-information variables found in data set')
    else:
        print('        %d variable(s) to be removed since ID or low-information variables'
                                %len(IDcols+cols_delete+discrete_string_vars))
        if len(IDcols+cols_delete+discrete_string_vars) <= 30:
            print('    \tvariables removed = %s' %(IDcols+cols_delete+discrete_string_vars))
        else:
            print('    \tmore than %s variables to be removed; too many to print...' %len(IDcols+cols_delete+discrete_string_vars))
    #############  Check if there are too many columns to visualize  ################
    ppt = pprint.PrettyPrinter(indent=4)
    if verbose >= 2 and len(cols_list) <= max_cols_analyzed:
        marthas_columns(dft,verbose)
    if verbose==1 and len(cols_list) <= max_cols_analyzed:
        print("   Columns to delete:")
        ppt.pprint('   %s' % cols_delete)
        print("   Boolean variables %s ")
        ppt.pprint('   %s' % bool_vars)
        print("   Categorical variables %s ")
        ppt.pprint('   %s' % categorical_vars)
        print("   Continuous variables %s " )
        ppt.pprint('   %s' % continuous_vars)
        print("   Discrete string variables %s " )
        ppt.pprint('   %s' % discrete_string_vars)
        print("   Date and time variables %s " )
        ppt.pprint('   %s' % date_vars)
        print("   ID variables %s ")
        ppt.pprint('   %s' % IDcols)
        print("   Target variable %s ")
        ppt.pprint('   %s' % depVar)
    elif verbose==1 and len(cols_list) > max_cols_analyzed:
        print('   Total columns > %d, too numerous to list.' %max_cols_analyzed)
    features_dict = dict([('IDcols',IDcols),('cols_delete',cols_delete),('bool_vars',bool_vars),(
                            'categorical_vars',categorical_vars),
                        ('continuous_vars',continuous_vars),('discrete_string_vars',discrete_string_vars),
                        ('date_vars',date_vars)])
    return features_dict
#######################################################################################################
def marthas_columns(data,verbose=0):
    """
    This program is named  in honor of my one of students who came up with the idea for it.
    It's a neat way of printing data types and information compared to the boring describe() function in Pandas.
    """
    data = data[:]
    print('Data Set Shape: %d rows, %d cols' % data.shape)
    if data.shape[1] > 30:
        print('Too many columns to print')
    else:
        if verbose==1:
            print('    Additional details on columns:')
            for col in data.columns:
                print('\t* %s:\t%d missing, %d uniques, most common: %s' % (
                        col,
                        data[col].isnull().sum(),
                        data[col].nunique(),
                        data[col].value_counts().head(2).to_dict()
                    ))
            print('--------------------------------------------------------------------')
################################################################################
######### NEW And FAST WAY to CLASSIFY COLUMNS IN A DATA SET #######
################################################################################
#################################################################################
def lenopenreadlines(filename):
    with open(filename) as f:
        return len(f.readlines())
#########################################################################################
from collections import Counter
import time
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
##################################################################################
def load_file_dataframe(dataname, sep=",", header=0, verbose=0, 
                    nrows=None, parse_dates=False, target='', is_test_flag=False):
    start_time = time.time()
    ### This is where you have to make sure target is not empty #####
    if not isinstance(dataname,str):
        dfte = copy.deepcopy(dataname)
        if isinstance(target, str):
            if not is_test_flag:
                if len(target) == 0:
                    modelt = 'Clustering'
                    print('featurewiz does not work on clustering or unsupervised problems. Returning...')
                    return dataname
                else:
                    modelt, _ = analyze_problem_type(dataname[target], target)
            else:
                ### For test data, just check the target value which will be given as odeltype ##
                modelt = copy.deepcopy(target)
        else:
            ### Target is a list or None ############
            if not is_test_flag:
                if target is None or len(target) == 0:
                    modelt = 'Clustering'
                    print('featurewiz does not work on clustering or unsupervised problems. Returning...')
                    return dataname
                else:
                    modelt, _ = analyze_problem_type(dataname[target], target)
            else:
                ## For test data, the modeltype is given in the target variable 
                modelt = copy.deepcopy(target)
    ###########################  This is where we load file or data frame ###############
    elif isinstance(dataname,str):
        if dataname == '':
            print('    No file given. Continuing...')
            return None
        #### this means they have given file name as a string to load the file #####
        codex = ['ascii', 'utf-8', 'iso-8859-1', 'cp1252', 'latin1']
        ## this is the total number of rows in df  ###
        ###############################################################################
        if dataname != '' and dataname.endswith(('csv')):
            try:
                ### You can read the entire data into pandas first and then stratify split it ##
                ###   If you don't stratify it, then you will have less classes than 2 error ###
                dfte = pd.read_csv(dataname, sep=sep, header=header, encoding=None, 
                                parse_dates=parse_dates)
                print('    pandas default encoder does not work for this file. Trying other encoders...')
            except:
                for code in codex:
                    try:
                        dfte = pd.read_csv(dataname, sep=sep, header=header, 
                                        encoding=code, parse_dates=parse_dates)
                        break
                    except:
                        continue
            ######### If the file is not loadable, then give an error message #########
            try:
                print('    Shape of your Data Set loaded: %s' %(dfte.shape,))
            except:
                print('    File not loadable. Please check your file path or encoding format and try again.')
                return dataname
        elif dataname.endswith(('xlsx','xls','txt')):
            #### It's very important to get header rows in Excel since people put headers anywhere in Excel#
            dfte = pd.read_excel(dataname,header=header, parse_dates=parse_dates)
        elif dataname.endswith(('gzip', 'bz2', 'zip', 'xz')):
            print('    Reading compressed file...')
            try:
                #### Dont use skip_function in zip files #####
                compression = 'infer'
                dfte = pd.read_csv(dataname, sep=sep, header=header, encoding=None,
                                compression=compression, parse_dates=parse_dates) 
            except:
                print('    Could not read compressed file. Please unzip and try again...')
                return dataname
    elif isinstance(dataname, pd.DataFrame):        
        dfte = copy.deepcopy(dataname)
    else:
        print('Dataname input must be a filename with path to that file or a Dataframe')
        return None
    ######################### This is where you sample rows ############################
    #### this means they now have a dataframe and you must sample it correctly #####
    #### Now that you have read the file, you must sample it ############
    ####################################################################################
    if not nrows is None:
        if nrows < dfte.shape[0]:
            if modelt == 'Regression':
                dfte = dfte[:nrows]
                print('        sequentially select %s max_rows from dataset %d...' %(nrows, dfte.shape[0]))
            else:
                test_size = 1 - (nrows/dfte.shape[0])
                print('        stratified split %d rows from given %s' %(nrows, dfte.shape[0]))
                dfte, _ = train_test_split(dfte, test_size=test_size, stratify=dfte[target],
                                shuffle=True, random_state=99)
    if len(np.array(list(dfte))[dfte.columns.duplicated()]) > 0:
        print('You have duplicate column names in your data set. Removing duplicate columns now...')
        dfte = dfte[list(dfte.columns[~dfte.columns.duplicated(keep='first')])]
    return dfte
##########################################################################################
##### This function loads a time series data and sets the index as a time series
def load_dask_data(filename, sep, ):
    """
    This function loads a given filename into a dask dataframe.
    If the input is a pandas DataFrame, it converts it into a dask dataframe.
    Note that filename should contain the full path to the file.
    """
    n_workers = get_cpu_worker_count()
    if isinstance(filename, str):
            dft = dd.read_csv(filename, blocksize='default')
            print('    Too big to fit into pandas. Hence loaded file %s into a Dask dataframe ...' % filename)
    else:
        ### If filename is not a string, it must be a dataframe and can be loaded
        dft =   dd.from_pandas(filename, npartitions=n_workers)
    return dft
##################################################################################
# Removes duplicates from a list to return unique values - USED ONLYONCE
def find_remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
#################################################################################
import copy
def FE_drop_rows_with_infinity(df, cols_list, fill_value=None):
    """
    This feature engineering function will fill infinite values in your data with a fill_value.
    You might need this function during deep_learning models where infinite values don't work.
    You can also leave the fill_value as None which means we will drop the rows with infinity.
    This function checks for both negative and positive infinity values to fill or remove.
    """
    # first you must drop rows that have inf in them ####
    print('    Shape of dataset initial: %s' %(df.shape[0]))
    corr_list_copy = copy.deepcopy(cols_list)
    init_rows = df.shape[0]
    if fill_value:
        for col in corr_list_copy:
            ### Capping using the n largest value based on n given in input.
            maxval = df[col].max()  ## what is the maximum value in this column?
            minval = df[col].min()
            if maxval == np.inf:
                sorted_list = sorted(df[col].unique())
                ### find the n_smallest values after the maximum value based on given input n
                next_best_value_index = sorted_list.index(np.inf) - 1
                capped_value = sorted_list[next_best_value_index]
                df.loc[df[col]==maxval, col] =  capped_value ## maximum values are now capped
            if minval == -np.inf:
                sorted_list = sorted(df[col].unique())
                ### find the n_smallest values after the maximum value based on given input n
                next_best_value_index = sorted_list.index(-np.inf)+1
                capped_value = sorted_list[next_best_value_index]
                df.loc[df[col]==minval, col] =  capped_value ## maximum values are now capped
        print('        capped all rows with infinite values in data')
    else:
        for col in corr_list_copy:
            df = df[df[col]!=np.inf]
            df = df[df[col]!=-np.inf]
        dropped_rows = init_rows - df.shape[0]
        print('        dropped %d rows due to infinite values in data' %dropped_rows)
        print('    Shape of dataset after dropping rows: %s' %(df.shape[0]))
    ###  Double check that all columns have been fixed ###############
    cols_with_infinity = EDA_find_remove_columns_with_infinity(df)
    if cols_with_infinity:
        print('    There are still %d columns with infinite values. Returning...' %len(cols_with_infinity))
    else:
        print('    There are no more columns with infinite values')
    return df
##################################################################################
def count_freq_in_list(lst):
    """
    This counts the frequency of items in a list but MAINTAINS the order of appearance of items.
    This order is very important when you are doing certain functions. Hence this function!
    """
    temp=np.unique(lst)
    result = []
    for i in temp:
        result.append((i,lst.count(i)))
    return result
###############################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#################################################################################
def return_factorized_dict(ls):
    """
    ######  Factorize any list of values in a data frame using this neat function
    if your data has any NaN's it automatically marks it as -1 and returns that for NaN's
    Returns a dictionary mapping previous values with new values.
    """
    factos = pd.unique(pd.factorize(ls)[0])
    categs = pd.unique(pd.factorize(ls)[1])
    if -1 in factos:
        categs = np.insert(categs,np.where(factos==-1)[0][0],np.nan)
    return dict(zip(categs,factos))
###################################################################################
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from .databunch import DataBunch
from .encoders import FrequencyEncoder

from sklearn.model_selection import train_test_split
def featurewiz(dataname, target, corr_limit=0.8, verbose=0, sep=",", header=0,
            test_data='', feature_engg='', category_encoders='', dask_xgboost_flag=False,
            nrows=None, skip_sulov=False, skip_xgboost=False,  **kwargs):
    """
    #################################################################################
    ###############           F E A T U R E   W I Z A R D          ##################
    ################  featurewiz library developed by Ram Seshadri  #################
    # featurewiz utilizes SULOV METHOD which is a fast method for feature selection #
    #####  SULOV also means Searching for Uncorrelated List Of Variables (:-)  ######
    ###############     A L L   R I G H T S  R E S E R V E D         ################
    #################################################################################
    Featurewiz is the main module of this library. You will create features and select
    the best features using the SULOV method and permutation based XGB feature importance.
    It returns a list of important features from your dataframe after feature engineering.
    Since we do label encoding, you can send both categorical and numeric vars.
    You can also send in features with NaN's in them.
    #################################################################################
    Inputs:
        dataname: training data set you want to input. dataname could be a datapath+filename or a dataframe. 
            featurewiz will detect whether your input is a filename or a dataframe and load it automatically.
        target: name of the target variable in the data set. Also known as dependent variable.
        corr_limit: if you want to set your own threshold for removing variables as
            highly correlated, then give it here. The default is 0.7 which means variables less
            than -0.7 and greater than 0.7 in pearson's correlation will be candidates for removal.
        verbose: This has 3 possible states:
            0 limited output. Great for running this silently and getting fast results.
            1 more verbiage. Great for knowing how results were and making changes to flags in input.
            2 SULOV charts and output. Great for finding out what happens under the hood for SULOV method.
        test_data: If you want to transform test data in the same way you are transforming dataname, you can.
            test_data could be the name of a datapath+filename or a dataframe. featurewiz will detect whether
                your input is a filename or a dataframe and load it automatically. Default is empty string.
        feature_engg: You can let featurewiz select its best encoders for your data set by setting this flag
            for adding feature engineering. There are three choices. You can choose one, two or all three in a list.
            'interactions': This will add interaction features to your data such as x1*x2, x2*x3, x1**2, x2**2, etc.
            'groupby': This will generate Group By features to your numeric vars by grouping all categorical vars.
            'target':  This will encode & transform all your categorical features using certain target encoders.
            Default is empty string (which means no additional feature engineering to be performed)
        category_encoders: Instead of above method, you can choose your own kind of category encoders from below.
            Recommend you do not use more than two of these.
                            Featurewiz will automatically select only two from your list.
            Default is empty string (which means no encoding of your categorical features)
                ['HashingEncoder', 'SumEncoder', 'PolynomialEncoder', 'BackwardDifferenceEncoder',
                'OneHotEncoder', 'HelmertEncoder', 'OrdinalEncoder', 'FrequencyEncoder', 'BaseNEncoder',
                'TargetEncoder', 'CatBoostEncoder', 'WOEEncoder', 'JamesSteinEncoder']
        dask_xgboost_flag: default = False. This flag enables DASK by default so that you can process large
            data sets faster using parallel processing. It detects the number of CPUs and GPU's in your machine
            automatically and sets the num of workers for DASK. It also uses DASK XGBoost to run it.
        nrows: default = None: None means all rows will be utilized. If you want to sample "N" rows, set nrows=N.
        skip_sulov: a new flag to skip SULOV method. It will automatically go straight to recursive xgboost.
        skip_xgboost: a new flag to skip recursive xgboost.
    ########           Featurewiz Output           #############################
    Output: Tuple
    Featurewiz can output either a list of features or one dataframe or two depending on what you send in.
        1. features: featurewiz will return just a list of important features
                     in your data if you send in just a dataset.
        2. trainm: modified train dataframe is the dataframe that is modified
                        with engineered and selected features from dataname.
        3. testm: modified test dataframe is the dataframe that is modified with
                    engineered and selected features from test_data
    """
    if verbose:
        print('############################################################################################')
        print('############       F A S T   F E A T U R E  E N G G    A N D    S E L E C T I O N ! ########')
        print("# Be judicious with featurewiz. Don't use it to create too many un-interpretable features! #")
        print('############################################################################################')
    print('featurewiz has selected %s as the correlation limit. Change this limit to fit your needs...' %corr_limit)
    if not nrows is None:
        print('ALERT: nrows=%s. Hence featurewiz will randomly sample that many rows.' %nrows)
        print('    Change nrows=None if you want all rows...')
    ### set all the defaults here ##############################################
    dataname = copy.deepcopy(dataname)
    max_nums = 30
    max_cats = 15
    maxrows = 10000
    RANDOM_SEED = 42
    mem_limit = 500 ### amount of memory consumed by pandas df before reducing_mem function called
    ############################################################################
    cat_encoders_list = list(settings.cat_encoders_names.keys())
    ### Just set defaults here which can be overridden by user input ####
    cat_vars = []
    if kwargs:
        for key, value in zip(kwargs.keys(), kwargs.values()):
            print('You supplied %s = %s' %(key, value))
            ###### Now test the next set of kwargs ###
            if key == 'cat_vars':
                if isinstance(value, list):
                    cat_vars = value
                elif isinstance(value, str):
                    cat_vars = value
                else:
                    print('cat vars must be a list or a string')
                    return
    ######################################################################################
    #####      MAKING FEATURE_TYPE AND FEATURE_GEN SELECTIONS HERE           #############
    ######################################################################################
    feature_generators = ['interactions', 'groupby', 'target']
    feature_gen = ''
    if feature_engg:
        if isinstance(feature_engg, str):
            if feature_engg in feature_generators:
                feature_gen = [feature_engg]
            else:
                print('feature engg types must be one of three strings: %s' %feature_generators)
                return
        elif isinstance(feature_engg, list):
            feature_gen = copy.deepcopy(feature_engg)
    else:
        print('    Skipping feature engineering since no feature_engg input...')
    #######     Now start doing the pre-processing   ###################
    feature_type = ''
    if category_encoders:
        if isinstance(category_encoders, str):
            feature_type = [category_encoders]
        elif isinstance(category_encoders, list):
            feature_type = category_encoders[:2] ### Only two will be allowed at a time
    else:
        print('Skipping category encoding since no category encoders specified in input...')
    ######################################################################################
    ##################    L O A D     T R A I N   D A T A   ##############################
    ##########   dataname will be the name of the pandas version of train data      ######
    ##########           train will be the Dask version of train data               ######
    ######################################################################################
    if isinstance(dataname, str):
        #### This is where we get a filename as a string as an input #################
        if re.search(r'(.ftr)', dataname) or re.search(r'(.feather)', dataname):
            print("""**INFO: Feather format allowed. Loading feather formatted file...**""")
            import feather
            dataname = pd.read_feather(dataname, use_threads=True)
            train = load_dask_data(dataname, sep)
        else:
            if verbose:
                print("""**INFO: to increase file loading performance, convert huge `csv` files to `feather` format""")
                print("""**INFO: Use `df.reset_index(drop=True).to_feather("path/to/save/file.ftr")` to save file in feather format**""")
            if dask_xgboost_flag:
                try:
                    print('    Since dask_xgboost_flag is True, reducing memory size and loading into dask')
                    dataname = pd.read_csv(dataname, sep=sep, header=header, nrows=nrows)
                    if (dataname.memory_usage().sum()/1000000) > mem_limit:
                        dataname = reduce_mem_usage(dataname)
                    train = load_dask_data(dataname, sep)
                except:
                    print('File could not be loaded into dask. Check the path or filename and try again')
                    return None
            else:
                #### There is no dask flag so load it into a regular pandas dataframe ####
                train = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, 
                                    nrows=nrows, target=target)
                if (train.memory_usage().sum()/1000000) > mem_limit:
                    dataname = reduce_mem_usage(train)
                else:
                    dataname = copy.deepcopy(train)
    else:
        #### This is where we get a dataframe as an input #################
        if dask_xgboost_flag:
            if not nrows is None:
                dataname = dataname.sample(n=nrows, replace=True, random_state=9999)
                print('Sampling %s rows from dataframe given' %nrows)
            print('    Since dask_xgboost_flag is True, reducing memory size and loading into dask')
            if (dataname.memory_usage().sum()/1000000) > mem_limit:
                dataname = reduce_mem_usage(dataname)
            train = load_dask_data(dataname, sep)
        else:
            train = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, 
                            nrows=nrows, target=target)
            if (train.memory_usage().sum()/1000000) > mem_limit:
                dataname = reduce_mem_usage(train)
            else:
                dataname = copy.deepcopy(train)
    print('    Loaded train data. Shape = %s' %(dataname.shape,))
    ##################    L O A D    T E S T   D A T A      ######################
    dataname = remove_duplicate_cols_in_dataset(dataname)
 
    #### Convert mixed data types to string data type  ############################
    #dataname = FE_convert_mixed_datatypes_to_string(dataname)
    
    ######   XGBoost cannot handle special chars in column names ###########
    uniq = Column_Names_Transformer()
    dataname = uniq.fit_transform(dataname)
    new_col_names = uniq.new_column_names
    old_col_names = uniq.old_column_names
    special_char_flag = uniq.transformed_flag

    ### Suppose you have changed the names, thenn you must load it in dask again ##    
    if special_char_flag:
        if dask_xgboost_flag:
            train = load_dask_data(dataname, sep)

    ###### Now save the old and new columns in a dictionary to use them later ###
    col_name_mapper = dict(zip(new_col_names, old_col_names))
    col_name_replacer = {y: x for (x, y) in col_name_mapper.items()}
    item_replacer = col_name_replacer.get  # For faster gets.

    #### You need to change the target name if you have changed the column names ### 
    if special_char_flag:
        if isinstance(target, str):
            targets = [target]
            targets = [item_replacer(n, n) for n in targets]
            target = targets[0]
        else:
            targets = copy.deepcopy(target)
            target = [item_replacer(n, n) for n in targets]

    train_index = dataname.index
    
    if isinstance(target, str):
        if len(target) == 0:
            cols_list = list(dataname)
            settings.modeltype = 'Clustering'
            print('featurewiz does not work on clustering or unsupervised problems. Returning...')
            return old_col_names, dataname
        else:
            settings.modeltype, _ = analyze_problem_type(dataname[target], target)
            cols_list = left_subtract(list(dataname),target)
    else:
        ### Target is a list or None ############
        if target is None or len(target) == 0:
            cols_list = list(dataname)
            settings.modeltype = 'Clustering'
            print('featurewiz does not work on clustering or unsupervised problems. Returning...')
            return old_col_names, dataname
        else:
            settings.modeltype, _ = analyze_problem_type(dataname[target], target)
            cols_list = left_subtract(list(dataname),target)

    ######################################################################################
    ##################    L O A D      T E S T     D A T A   #############################
    ##########   test_data will be the name of the pandas version of test data      #####
    ##########   test will be the name of the dask dataframe version of test data   #####
    ######################################################################################
    if isinstance(test_data, str):
        if test_data != '':
            if re.search(r'(.ftr)', test_data):
                print("""**INFO: Feather format allowed. Loading feather file...**""")
                import feather
                test_data = pd.read_feather(test_data, use_threads=True)
                test = load_dask_data(test_data, sep)
            else:
                if verbose:
                    print("""**INFO: to increase file loading performance, convert huge `csv` files to `feather` format using `df.to_feather("path/to/save/file.feather")`**""")
                    print('**INFO: featurewiz can now read feather formatted files...***')
                ### only if test_data is a filename load this #####
                print('Loading test data filename = %s...' %test_data)
                if dask_xgboost_flag:
                    print('    Since dask_xgboost_flag is True, reducing memory size and loading into dask')
                    ### nrows does not apply to test data in the case of featurewiz ###############
                    test_data = load_file_dataframe(test_data, sep=sep, header=header, verbose=verbose,
                                     nrows=None, target=settings.modeltype, is_test_flag=True)
                    ### sometimes, test_data returns None if there is an error. ##########
                    if test_data is not None:
                        test_data = reduce_mem_usage(test_data)
                        ### test_data is the pandas dataframe object and test is dask dataframe object ##
                        test = load_dask_data(test_data, sep)
                else:
                    #### load the entire test dataframe - there is no limit applicable there #########
                    test_data = load_file_dataframe(test_data, sep=sep, header=header, 
                                    verbose=verbose, nrows=None, target=settings.modeltype, is_test_flag=True)
                    test = copy.deepcopy(test_data)
        else:
            print('No test data filename given...')
            test_data = None
            test = None
    else:
        print('loading the entire test dataframe - there is no nrows limit applicable #########')
        test_data = load_file_dataframe(test_data, sep=sep, header=header, 
                        verbose=verbose, nrows=None, target=settings.modeltype, is_test_flag=True)
        test = copy.deepcopy(test_data)
    ### sometimes, test_data returns None if there is an error. ##########
    if test_data is not None:
        test_data = remove_duplicate_cols_in_dataset(test_data)
        test_index = test_data.index
        print('    Loaded test data. Shape = %s' %(test_data.shape,))
        #######  Once again remove special chars in test data as well ###
        test_data = uniq.transform(test_data)

        ### Suppose you have changed the names, thenn you must load it in dask again ##    
        if special_char_flag:
            if dask_xgboost_flag:
                ### Re-load test into dask in case names have been changed ###
                test = load_dask_data(test_data, sep)
        ##### convert mixed data types to string ############
        #test_data = FE_convert_mixed_datatypes_to_string(test_data)
        #test = FE_convert_mixed_datatypes_to_string(test)
    #############    C L A S S I F Y    F E A T U R E S      ####################
    if nrows is None:
        nrows_limit = maxrows
    else:
        nrows_limit = int(min(nrows, maxrows))
    #### you can use targets as a list wherever you choose #####
    if isinstance(target, str):
        targets = [target]
    else:
        targets = copy.deepcopy(target)
    if dataname.shape[0] >= nrows_limit:
        print('Classifying features using a random sample of %s rows from dataset...' %nrows_limit)
        ##### you can use nrows_limit to select a small sample from data set ########################
        train_small = EDA_randomly_select_rows_from_dataframe(dataname, targets, nrows_limit, DS_LEN=dataname.shape[0])
        features_dict = classify_features(train_small, target)
    else:
        features_dict = classify_features(dataname, target)
    #### Now we have to drop certain cols that must be deleted #####################
    remove_cols = features_dict['discrete_string_vars'] + features_dict['cols_delete']
    if len(remove_cols) > 0:
        print('train data shape before dropping %d columns = %s' %(len(remove_cols), dataname.shape,))
        dataname.drop(remove_cols, axis=1, inplace=True)
        print('\ttrain data shape after dropping columns = %s' %(dataname.shape,))
        train = load_dask_data(dataname, sep)
        if not test_data is None:
            test_data.drop(remove_cols, axis=1, inplace=True)
            test = load_dask_data(test_data, sep)
    ################    Load data frame with date var features correctly this time ################
    if len(features_dict['date_vars']) > 0:
        print('Caution: Since there are date-time variables in dataset, it is best to load them using pandas')
        dask_xgboost_flag = False ### Set the dask flag to be False since it is now becoming Pandas dataframe 
        date_time_vars = features_dict['date_vars']
        dataname = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, 
                            nrows=nrows, parse_dates=date_time_vars, target=target)
        if (dataname.memory_usage().sum()/1000000) > mem_limit:
            dataname = reduce_mem_usage(dataname)
        train = load_dask_data(dataname, sep)
        if not test_data is None:
            ### You must load the entire test data - there is no limit there ##################
            ### test_data is the pandas dataframe object and test is dask dataframe object ##
            test_data = load_file_dataframe(test_data, sep=sep, header=header, verbose=verbose, 
                                 nrows=nrows, parse_dates=date_time_vars, target=settings.modeltype, is_test_flag=True )
            test = copy.deepcopy(test_data)
        else:
            test_data = None
            test = None
    else:
        train_index = dataname.index
        if test_data is not None:
            test_index = test_data.index
    ################   X G B O O S T      D E F A U L T S      ######################################
    #### If there are more than 30 categorical variables in a data set, it is worth reducing features.
    ####  Otherwise. XGBoost is pretty good at finding the best features whether cat or numeric !
    #################################################################################################

    start_time = time.time()
    n_splits = 5
    max_depth = 8
    ######################   I M P O R T A N T    D E F A U L T S ##############
    subsample =  0.7
    col_sub_sample = 0.7
    test_size = 0.2
    #print('test_size = %s' %test_size)
    seed = 1
    early_stopping = 5
    ####### All the default parameters are set up now #########
    kf = KFold(n_splits=n_splits)
    #########     G P U     P R O C E S S I N G      B E G I N S    ############
    ###### This is where we set the CPU and GPU parameters for XGBoost
    GPU_exists = check_if_GPU_exists(verbose)
    n_workers = get_cpu_worker_count()
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    cpu_tree_method = 'hist'
    tree_method = 'hist'
    n_estimators = 100
    cpu_params['nthread'] = -1
    cpu_params['tree_method'] = 'hist'
    cpu_params['eta'] = 0.01
    cpu_params['subsample'] = 0.5
    cpu_params['grow_policy'] = 'depthwise' #'lossguide'
    cpu_params['n_estimators'] = n_estimators
    cpu_params['max_depth'] = max_depth
    cpu_params['max_leaves'] = 0
    cpu_params['verbosity'] = 0
    cpu_params['gpu_id'] = 0
    cpu_params['updater'] = 'grow_colmaker'
    cpu_params['predictor'] = 'cpu_predictor'
    cpu_params['num_parallel_tree'] = 1
    if GPU_exists:
        ### This has been fixed ###
        tree_method = 'gpu_hist'
        param['nthread'] = -1
        param['tree_method'] = 'gpu_hist'
        param['eta'] = 0.01
        param['subsample'] = 0.5
        param['grow_policy'] = 'depthwise' # 'lossguide' # 
        param['n_estimators'] = n_estimators
        param['max_depth'] = max_depth
        param['max_leaves'] = 0
        param['verbosity'] = 0
        param['gpu_id'] = 0
        param['updater'] = 'grow_gpu_hist' #'prune'
        param['predictor'] = 'gpu_predictor'
        param['num_parallel_tree'] = 1
        gpuid = 0
        if verbose:
            print('    Tuning XGBoost using GPU hyper-parameters. This will take time...')
    else:
        param = copy.deepcopy(cpu_params)
        gpuid = None
        if verbose:
            print('    Tuning XGBoost using CPU hyper-parameters. This will take time...')
    #################################################################################
    #############   D E T E C T  SINGLE OR MULTI-LABEL PROBLEM      #################
    #################################################################################
    if isinstance(target, str):
        target = [target]
        settings.multi_label = False
    else:
        if len(target) <= 1:
            settings.multi_label = False
        else:
            settings.multi_label = True
    #### You need to make sure only Single Label problems are handled in target encoding!
    if settings.multi_label:
        if verbose:
            print('Turning off Target encoding for multi-label problems like this data set...')
            print('    since Feature Engineering module cannot handle Multi Label Targets, turnoff target_enc_cat_features to False')
        target_enc_cat_features = False
    else:
        ## If target is specified in feature_gen then use it to Generate target encoded features
        target_enc_cat_features = 'target' in feature_gen
    ######################################################################################
    ########     C L A S S I F Y    V A R I A B L E S           ##########################
    ###### Now we detect the various types of variables to see how to convert them to numeric
    ######################################################################################
    date_cols = features_dict['date_vars']
    if len(features_dict['date_vars']) > 0:
        date_time_vars = copy.deepcopy(date_cols)
        #### Do this only if date time columns exist in your data set!
        date_col_mappers = {}
        for date_col in date_cols:
            print('Processing %s column for date time features....' %date_col)
            dataname, ts_adds = FE_create_time_series_features(dataname, date_col)
            date_col_mapper = dict([(x,date_col) for x in ts_adds])
            date_col_mappers.update(date_col_mapper)
            #print('    Adding %d column(s) from date-time column %s in train' %(len(date_col_adds_train),date_col))
            #train = train.join(date_df_train, rsuffix='2')
            if isinstance(test_data,str) or test_data is None:
                ### do nothing ####
                pass
            else:
                print('        Adding same time series features to test data...')
                test_data, _ = FE_create_time_series_features(test_data, date_col, ts_adds)
                #date_col_adds_test_data = left_subtract(date_df_test.columns.tolist(),date_col)
                ### Now time to remove the date time column from all further processing ##
                #test = test.join(date_df_test, rsuffix='2')
    ### Now time to continue with our further processing ##
    idcols = features_dict['IDcols']
    if isinstance(test_data,str) or test_data is None:
        pass
    else:
        test_ids = test_data[idcols]
    train_ids = dataname[idcols] ### this saves the ID columns of dataname
    if cat_vars:
        cols_in_both = [x for x in cat_vars if x in features_dict['cols_delete']]
        cat_vars = left_subtract(cat_vars, features_dict['cols_delete'])
        if len(cols_in_both) > 0:
            print('Removing %s columns(s) which are in both cols to be deleted and cat vars given as input' %cols_in_both)
    cols_to_remove = features_dict['cols_delete'] + idcols + features_dict['discrete_string_vars']
    print('Removing %d columns from further processing since ID or low information variables' %len(cols_to_remove))
    preds = [x for x in list(dataname) if x not in target+cols_to_remove]
    ###   This is where we sort the columns to make sure that the order of columns doesn't matter in selection ###########
    #preds = np.sort(preds)
    if verbose:
        print('    After removing redundant variables from further processing, features left = %d' %len(preds))
    numvars = dataname[preds].select_dtypes(include = 'number').columns.tolist()
    if len(numvars) > max_nums:
        if feature_gen:
            print('\nWarning: Too many extra features will be generated by featurewiz. This may take time...')
    if cat_vars:
        ### if input is given for cat_vars, use it!
        catvars = copy.deepcopy(cat_vars)
        numvars = left_subtract(preds, catvars)
    else:
        catvars = left_subtract(preds, numvars)
    if len(catvars) > max_cats:
        if feature_type:
            print('\nWarning: Too many extra features will be generated by category encoding. This may take time...')
    ######   C R E A T I N G    I N T X N  V A R S   F R O M   C A T  V A R S #####################
    if np.where('interactions' in feature_gen,True, False).tolist():
        if len(catvars) > 1:
            
            num_combos = len(list(combinations(catvars,2)))
            print('Adding %s interactions between %s categorical_vars %s...' %(
                                num_combos, len(catvars), catvars))
            dataname = FE_create_interaction_vars(dataname, catvars)
            train = FE_create_interaction_vars(train, catvars)
            catvars = left_subtract(dataname.columns.tolist(), numvars)
            catvars = left_subtract(catvars, target)
            preds =  left_subtract(dataname.columns.tolist(), target)
            if not test_data is None:
                test_data = FE_create_interaction_vars(test_data, catvars)
                test = FE_create_interaction_vars(test, catvars)
        else:
            if verbose:
                print('No interactions created for categorical vars since number less than 2')
    else:
        if verbose:
            print('No interactions created for categorical vars since feature engg does not specify it')
    ##### Now we need to re-set the catvars again since we have created new features #####
    rem_vars = copy.deepcopy(catvars)
    ########## Now we need to select the right model to run repeatedly ####
    if settings.modeltype != 'Regression':
        ##########################################################################
        ###########   L A B E L    E N C O D I N G   O F   T A R G E T   #########
        ##########################################################################
        ### This is to convert the target labels to proper numeric columns ######
        target_conversion_flag = False
        cat_targets = dataname[target].select_dtypes(include='object').columns.tolist() + dataname[target].select_dtypes(include='category').columns.tolist()
        copy_targets = copy.deepcopy(targets)
        for each_target in copy_targets:
                if cat_targets or sorted(np.unique(dataname[each_target].values))[0] != 0:
                    print('    target labels need to be converted...')
                    target_conversion_flag = True
        ### check if they are not starting from zero ##################
        copy_targets = copy.deepcopy(target)
        for each_target in copy_targets:
            if target_conversion_flag:
                mlb = My_LabelEncoder()
                dataname[each_target] = mlb.fit_transform(dataname[each_target])
                try:
                    ## After converting train, just load it into dask again ##
                    train[each_target] = dd.from_pandas(dataname[each_target], npartitions=n_workers)
                except:
                    print('Could not convert dask dataframe target into numeric. Check your input. Continuing...')
                if test_data is not None:
                    if each_target in test_data.columns:
                        test_data[each_target] = mlb.transform(test_data[each_target])
                        try:
                            ## After converting test, just load it into dask again ##
                            test[each_target] = dd.from_pandas(test_data[each_target], npartitions=n_workers)
                        except:
                            print('Could not convert dask dataframe target into numeric. Check your input. Continuing...')
                print('Completed label encoding of target variable = %s' %each_target)
                print('How model predictions need to be transformed for %s:\n\t%s' %(each_target, mlb.inverse_transformer))

    ######################################################################################
    ######    B E F O R E    U S I N G    D A T A B U N C H  C H E C K ###################
    ######################################################################################
    ## Before using DataBunch check if certain encoders work with certain kind of data!
    if feature_type:
        final_cat_encoders = feature_type
    else:
        final_cat_encoders = []
    if settings.modeltype == 'Multi_Classification':
        ### you must put a Polynomial Wrapper on the cat_encoder in case the model is multi-class
        if final_cat_encoders:
            final_cat_encoders = [PolynomialWrapper(x) for x in final_cat_encoders if x in settings.target_encoders_names]
    elif settings.modeltype == 'Regression':
        if final_cat_encoders:
            if 'WOEEncoder' in final_cat_encoders:
                print('Removing WOEEncoder from list of encoders since it cannot be used for this Regression problem.')
            final_cat_encoders = [x for x in final_cat_encoders if x != 'WOEEncoder' ]
    ######################################################################################
    ######    F E A T U R E    E N G G    U S I N G    D A T A B U N C H  ###################
    ######################################################################################
    if feature_gen or feature_type:
        if isinstance(test_data, str) or test_data is None:
            print('    Starting feature engineering...Since no test data is given, splitting train into two...')
            if settings.multi_label:
                ### if it is a multi_label problem, leave target as it is - a list!
                X_train, X_test, y_train, y_test = train_test_split(dataname[preds],
                                                                dataname[target],
                                                                test_size=0.2,
                                                                random_state=RANDOM_SEED)
            else:
                ### if it not a multi_label problem, make target as target[0]
                X_train, X_test, y_train, y_test = train_test_split(dataname[preds],
                                                            dataname[target[0]],
                                                            test_size=0.2,
                                                            random_state=RANDOM_SEED)
        else:
            print('    Starting feature engineering...Since test data is given, using train and test...')
            X_train = dataname[preds]
            if settings.multi_label:
                y_train = dataname[target]
            else:
                y_train = dataname[target[0]]
            X_test = test_data[preds]
            try:
                y_test = test_data[target]
            except:
                y_test = None
        X_train_index = X_train.index
        X_test_index = X_test.index
        
        ##################################################################################################
        ###### Category_Encoders does not work with Dask - so don't send in Dask dataframes to DataBunch!
        ##################################################################################################
        data_tuple = DataBunch(X_train=X_train,
                    y_train=y_train,
                    X_test=X_test, # be sure to specify X_test, because the encoder needs all dataset to work.
                    cat_features = catvars,
                    clean_and_encod_data=True,
                    cat_encoder_names=final_cat_encoders, # final list of Encoders selected
                    clean_nan=True, # fillnan
                    num_generator_features=np.where('interactions' in feature_gen,True, False).tolist(), # Generate interaction Num Features
                    group_generator_features=np.where('groupby' in feature_gen,True, False).tolist(), # Generate groupby Features
                    target_enc_cat_features=target_enc_cat_features,# Generate target encoded features
                    normalization=False,
                    random_state=RANDOM_SEED,
                    )
        #### Now you can process the tuple this way #########
        if type(y_train) == dask.dataframe.core.DataFrame:
            ### since y_train is dask df and data_tuple.X_train is a pandas df, you can't merge them.
            y_train = y_train.compute()  ### remember you first have to convert them to a pandas df
        data1 = pd.concat([data_tuple.X_train, y_train], axis=1) ### data_tuple does not have a y_train, remember!
        
        if isinstance(test_data, str) or test_data is None:
            ### Since you have done a train_test_split using randomized split, you need to put it back again.
            if type(y_test) == dask.dataframe.core.DataFrame:
                ### since y_train is dask df and data_tuple.X_train is a pandas df, you can't merge them.
                y_test = y_test.compute()  ### remember you first have to convert them to a pandas df
            data2 = pd.concat([data_tuple.X_test, y_test], axis=1)
            dataname = pd.concat([data1, data2])
            ### Sometimes there are duplicate values in index when you append. So just remove duplicates here
            dataname = dataname[~dataname.index.duplicated()]
            dataname = dataname.reindex(train_index)
            print('    Completed feature engineering. Shape of Train (with target) = %s' %(dataname.shape,))
        else:
            try:
                if type(y_test) == dask.dataframe.core.DataFrame:
                    ### since y_train is dask df and data_tuple.X_train is a pandas df, you can't merge them.
                    y_test = y_test.compute()  ### remember you first have to convert them to a pandas df
                test_data = pd.concat([data_tuple.X_test, y_test], axis=1)
            except:
                test_data = copy.deepcopy(data_tuple.X_test)
            ### Sometimes there are duplicate values in index when you append. So just remove duplicates here
            test_data = test_data[~test_data.index.duplicated()]
            test_data = test_data.reindex(test_index)
            dataname = copy.deepcopy(data1)
            print('    Completed feature engineering. Shape of Test (with target) = %s' %(test_data.shape,))
        #################################################################################################
        ###### Train and Test are currently pandas data frames even if dask_xgboost_flag is True ########
        ######   That is because we combined them after feature engg to using Category_Encoders  ########
        #################################################################################################
        preds = [x for x in list(dataname) if x not in target]
        numvars = dataname[preds].select_dtypes(include = 'number').columns.tolist()
        if cat_vars:
            #### if cat_vars input is given, use it!
            catvars = copy.deepcopy(cat_vars)
            numvars = left_subtract(preds, catvars)
        else:
            catvars = left_subtract(preds, numvars)
    ######################   I M P O R T A N T ##############################################
    important_cats = copy.deepcopy(catvars)
    data_dim = int((len(dataname)*dataname.shape[1])/1e6)
    ################################################################################################
    ############     S   U  L  O   V       M   E   T   H   O  D      ###############################
    #### If the data dimension is less than 5o Million then do SULOV - otherwise skip it! #########
    ################################################################################################
    
    cols_with_infinity = EDA_find_remove_columns_with_infinity(dataname)
    # first you must drop rows that have inf in them ####
    if cols_with_infinity:
        print('Dropped %d columns which contain infinity in dataset' %len(cols_with_infinity))
        #dataname = FE_drop_rows_with_infinity(dataname, cols_with_infinity, fill_value=True)
        dataname = dataname.drop(cols_with_infinity, axis=1)
        print('%s' %cols_with_infinity)
        if isinstance(test_data,str) or test_data is None:
            pass
        else:
            test_data = test_data.drop(cols_with_infinity, axis=1)
            print('     dropped %s columns with infinity from test data...' %len(cols_with_infinity))
        numvars = left_subtract(numvars, cols_with_infinity)
        print('     numeric features left = %s' %len(numvars))
    #######  This is where you start the SULOV process ##################################    
    start_time1 = time.time()
    if len(numvars) > 1 and not skip_sulov:
        if data_dim < 50:
            try:
                final_list = FE_remove_variables_using_SULOV_method(dataname,numvars,settings.modeltype,target,
                             corr_limit,verbose, dask_xgboost_flag)
            except:
                print('    SULOV method is erroring. Continuing ...')
                final_list = copy.deepcopy(numvars)
        else:
                print('    Running SULOV on smaller dataset sample since data size %s m > 50 m. Continuing ...' %int(data_dim))
                if settings.modeltype != 'Regression':
                    data_temp = dataname.sample(n=10000, replace=True, random_state=99)
                else:
                    data_temp = dataname[:10000]
                final_list = FE_remove_variables_using_SULOV_method(data_temp,numvars,settings.modeltype,target,
                             corr_limit,verbose, dask_xgboost_flag)
                del data_temp
    elif skip_sulov:
        print('    Skipping SULOV method. Continuing ...')
        final_list = copy.deepcopy(numvars)
    else:
        print('    Skipping SULOV method since there are no continuous vars. Continuing ...')
        final_list = copy.deepcopy(numvars)
    ####### This is where you draw how featurewiz works when the verbose = 2 ###########
    print('Time taken for SULOV method = %0.0f seconds' %(time.time()-start_time1))
    #### Now we create interaction variables between categorical features ########
    if verbose:
        print('    Adding %s categorical variables to reduced numeric variables  of %d' %(
                                len(important_cats),len(final_list)))
    if  isinstance(final_list,np.ndarray):
        final_list = final_list.tolist()
    preds = final_list+important_cats
    if verbose and len(preds) <= 30:
        print('Final list of selected %s vars after SULOV = %s' %(len(preds), preds))
    else:
        print('Finally %s vars selected after SULOV' %(len(preds)))
    #######You must convert category variables into integers ###############    
    print('Converting all features to numeric before sending to XGBoost...')
    if isinstance(target, str):
        dataname = dataname[preds+[target]]
    else:
        dataname = dataname[preds+target]
    
    if not test_data is None:
        test_data = test_data[preds]
    if len(important_cats) > 0:
        dataname, test_data, error_columns = FE_convert_all_object_columns_to_numeric(dataname,  test_data, preds)
        important_cats = left_subtract(important_cats, error_columns)
        if len(error_columns) > 0:
            print('    removing %s object columns that could not be converted to numeric' %len(error_columns))
            preds = list(set(preds)-set(error_columns))
            dataname.drop(error_columns, axis=1, inplace=True)
        else:
            print('    there were no mixed data types or object columns that errored. Data is all numeric...')
        print('Shape of train data after adding missing values flags = %s' %(dataname.shape,) )
        preds = [x for x in list(dataname) if x not in targets]
        if not test_data is None:
            if len(error_columns) > 0:
                test_data.drop(error_columns, axis=1, inplace=True)
            print('    Shape of test data after adding missing values flags  = %s' %(test_data.shape,) )
    
    if not skip_xgboost:
        ### This is where we perform the recursive XGBoost method ##############
        if verbose:
            print('#######################################################################################')
            print('#####    R E C U R S I V E   X G B O O S T : F E A T U R E   S E L E C T I O N  #######')
            print('#######################################################################################')
        
        #################################################################################################
        ########   Now if dask_xgboost_flag is True, convert pandas dfs back to Dask Dataframes     #####
        #################################################################################################
        if dask_xgboost_flag:
            ### we reload the dataframes into dask since columns may have been dropped ##
            if verbose:
                print('    using DASK XGBoost')  
            train = load_dask_data(dataname, sep)
            if not test_data is None:
                test = load_dask_data(test_data, sep)
        else:
            ### we reload the dataframes into dask since columns may have been dropped ##
            if verbose:
                print('    using regular XGBoost') 
            train = copy.deepcopy(dataname)
            test = copy.deepcopy(test_data)
        ########  Conversion completed for train and test data ##########
        #### If Category Encoding took place, these cat variables are no longer needed in Train. So remove them!
        if feature_gen or feature_type:
            print('Since %s category encoding is done, dropping original categorical vars from predictors...' %feature_gen)
            preds = left_subtract(preds, catvars)
        #### Now we process the numeric  values through DASK XGBoost repeatedly ###################
        start_time2 = time.time()
        if dask_xgboost_flag:
            important_features = FE_perform_recursive_xgboost(train, target, 
                                settings.modeltype, settings.multi_label, dask_xgboost_flag, verbose)
        else:
            important_features = FE_perform_recursive_xgboost(dataname, target, 
                                settings.modeltype, settings.multi_label, dask_xgboost_flag, verbose)
        ######    E    N     D      O  F      X  G  B  O  O  S  T    S E L E C T I O N ##############
        print('    Completed XGBoost feature selection in %0.0f seconds' %(time.time()-start_time2))
        if len(idcols) > 0:
            print('    Alert: No ID variables %s are included in selected features' %idcols)
      
    else:
        print('Skipping Recursive XGBoost method. Continuing ...')
        important_features = copy.deepcopy(preds)

    if verbose:
        print("#######################################################################################")
        print("#####          F E A T U R E   S E L E C T I O N   C O M P L E T E D            #######")
        print("#######################################################################################")
    dicto = {}
    missing_flags1 = [{x:x[:-13]} for x in important_features if 'Missing_Flag' in x]
    for each_flag in missing_flags1:
        print('Alert: Dont forget to add a missing flag to %s to create %s column' %(list(each_flag.values())[0], list(each_flag.keys())[0]))
        dicto.update(each_flag)
    if len(dicto) > 0:
        important_features = [dicto.get(item,item)  for item in important_features]
        important_features = list(set(important_features))
    if len(important_features) <= 30:
        print('Selected %d important features:\n%s' %(len(important_features), important_features))
    else:
        print('Selected %d important features. Too many to print...' %len(important_features))
    numvars = [x for x in numvars if x in important_features]
    important_cats = [x for x in important_cats if x in important_features]
    print('Total Time taken for featurewiz selection = %0.0f seconds' %(time.time()-start_time))
    #### Now change the feature names back to original feature names ################
    item_replacer = col_name_mapper.get  # For faster gets.
    ##########################################################################
    ### You select the features with the same old names as before here #######
    ##########################################################################
    ## In one case, column names get changed in train but not in test since it test is not available.
    if isinstance(test_data, str) or test_data is None:
        print('Output contains a list of %s important features and a train dataframe' %len(important_features))
    else:
        print('Output contains two dataframes: train and test with %d important features.' %len(important_features))
    if feature_gen or feature_type:
        if isinstance(test_data, str) or test_data is None:
            ### if feature engg is performed, id columns are dropped. Hence they must rejoin here.
            dataname = pd.concat([train_ids, dataname], axis=1)
            if isinstance(target, str):
                return important_features, dataname[important_features+[target]]
            else:
                return important_features, dataname[important_features+target]
        else:
            ### if feature engg is performed, id columns are dropped. Hence they must rejoin here.
            dataname = pd.concat([train_ids, dataname], axis=1)
            test_data = pd.concat([test_ids, test_data], axis=1)
            if isinstance(target, str):
                return dataname[important_features+[target]], test_data[important_features]
            else:
                return dataname[important_features+target], test_data[important_features]
    else:
        ### You select the features with the same old names as before #######
        old_important_features = copy.deepcopy(important_features)
        if len(date_cols) > 0:
            date_replacer = date_col_mappers.get  # For faster gets.
            important_features1 = [date_replacer(n, n) for n in important_features]
        else:
            important_features1 = [item_replacer(n, n) for n in important_features]
        important_features = find_remove_duplicates(important_features1)
        if old_important_features == important_features:
            ## Don't drop the old target since there is only one target here ###
            pass
        else:
            if len(old_important_features) == len(important_features):
                ### You just move the values from the new names to the old feature names ##
                dataname[important_features] = dataname[old_important_features]
                if isinstance(test_data, str) or test_data is None:
                    pass
                else:
                    #### if there is test data transfer values to it ###
                    test_data[important_features] = test_data[old_important_features]
            else:
                ### first try to return with the new important features, if that fails return with old features
                try:
                    print('There are special chars in column names. Please remove them and try again.')
                    if isinstance(test_data, str) or test_data is None:
                        return important_features, dataname[important_features]
                    else:
                        return dataname[important_features], test_data[important_features]
                except:
                    print('There are special chars in column names. Returning with important features and train.')
                    if isinstance(test_data, str) or test_data is None:
                        return old_important_features, dataname[old_important_features]
                    else:
                        return dataname[old_important_features], test_data[old_important_features]   
        
        old_target = copy.deepcopy(target)
        if isinstance(target, str):
            target = item_replacer(target, target)
            targets = [target]
        else:
            target = [item_replacer(n, n) for n in target]
            targets = copy.deepcopy(target)

        if old_target == target:
            ## Don't drop the old target since there is only one target here ###
            pass
        else:
            ### you don't need drop the cols that have changed since only a few are selected #######
            if isinstance(target, str):
                dataname[target] = dataname[old_target]
            else:
                copy_targets = copy.deepcopy(targets)
                copy_old = copy.deepcopy(old_target)
                for each_target, each_old_target in zip(copy_targets, copy_old):
                    dataname[each_target] = dataname[each_old_target]

        #### This is where we check whether to return test data or not ######
        try:
            if isinstance(test_data, str) or test_data is None:
                if feature_gen or feature_type:
                    ### if feature engg is performed, id columns are dropped. Hence they must rejoin here.
                    dataname = pd.concat([train_ids, dataname], axis=1)
                return important_features, dataname[important_features+target]
            else:
                ## This is for test data existing ####
                if feature_gen or feature_type:
                    ### if feature engg is performed, id columns are dropped. Hence they must rejoin here.
                    dataname = pd.concat([train_ids, dataname], axis=1)
                    test_data = pd.concat([test_ids, test_data], axis=1)
                ### You select the features with the same old names as before #######
                return dataname[important_features+targets], test_data[important_features]
        except:
            print('Warning: Returning with important features and train. Please re-check your outputs.')
            return important_features, dataname[important_features+targets]
################################################################################
def FE_perform_recursive_xgboost(train, target, model_type, multi_label_type,
                            dask_xgboost_flag=False, verbose=0):
    """
    Perform recursive XGBoost to identify most important features in an all-
        numeric dataset. If the dataset is not numeric, it will give an error.
    
    Inputs:
        X_train: a pandas dataframe containing all-numeric features
        y_train: a pandas Series or Dataframe containing all-numeric features.
        model_type: can be one of "Classification" or "Regression". If it is 
            multi-class, you can also use 'Multi-Classification" as input.
        multi_label_type: can be one of "Single_Label" or "Multi_Label".
        dask_xgboost_flag: False by default. You can set it to True if your dataset
            is a Dask dataframe and Series.
        
    output:
        features: a list of top features in dataset.
    
    """
    ### TO DO: target can be multi-label - need to test for it
    ### TO DO: train can be DASK dataframe - need to test for it
    ### we need to use the name train_p to start and then change it to X_train 
    train_p = train.drop(target, axis=1)
    ### train is already a dask dataframe -> you can leave it as it is
    y_train = train[target]
    cpu_tree_method = 'hist'
    cols_sel = train_p.columns.tolist()
    ######## Do the Dask settings here  #######
    if dask_xgboost_flag:
        if verbose:
            print('    Dask version = %s' %dask.__version__)
        ### You can remove dask_ml here since you don't do any split of train test here ##########
        #from dask_ml.model_selection import train_test_split
        ### check available memory and allocate at least 1GB of it in the Client in DASK #########
        n_workers = get_cpu_worker_count()
        ### Avoid reducing the free memory - leave it as big as it wants to be ###
        memory_free = str(max(1, int(psutil.virtual_memory()[0]/(n_workers*1e9))))+'GB'
        print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(
                                get_cpu_worker_count(), memory_free))
        client = Client(n_workers= n_workers, threads_per_worker=1, processes=True, silence_logs=50,
                        memory_limit=memory_free)
        print('Dask client configuration: %s' %client)
        import gc
        client.run(gc.collect) 
        #train_p = client.persist(train_p)
    important_features = []
    #######################################################################
    #####   This is for DASK XGB Regressor and XGB Classifier problems ####
    #######################################################################
    if settings.multi_label:
        ### only regular xgboost with multi-output can work well in multi-label settings #
        dask_xgboost_flag = False 
    bst_models = []

    #########   This is for DASK Dataframes XGBoost training ####################
    try:
        xgb.set_config(verbosity=0)
    except:
        ## Some cases, this errors, so pass ###
        pass
    ####  Limit the number of iterations here ######
    if train_p.shape[1] <= 10:
        iter_limit = 2
    else:
        iter_limit = int(train_p.shape[1]/5+0.5)
    ######################   I M P O R T A N T ##############################################
    ###### This top_num decides how many top_n features XGB selects in each iteration.
    ####  There a total of 5 iterations. Hence 5x10 means maximum 50 features will be selected.
    #####  If there are more than 50 variables, then maximum 25% of its variables will be selected
    if len(cols_sel) <= 50:
        #top_num = 10
        top_num = int(max(2, len(cols_sel)*0.25))
    else:
        ### the maximum number of variables will be 25% of preds which means we divide by 4 and get 25% here
        ### The five iterations result in 10% being chosen in each iteration. Hence max 50% of variables!
        top_num = int(len(cols_sel)*0.20)
    if verbose:
        print('    Taking top %s features per iteration...' %top_num)
    try:
        for i in range(0,train_p.shape[1],iter_limit):
            start_time1 = time.time()
            imp_feats = []
            if train_p.shape[1]-i < iter_limit:
                X_train = train_p.iloc[:,i:]
                cols_sel = X_train.columns.tolist()
            else:
                X_train = train_p[list(train_p.columns.values)[i:train_p.shape[1]]]
                cols_sel = X_train.columns.tolist()
            ##### This is where you repeat the training and finding feature importances
            if dask_xgboost_flag:
                rows = X_train.compute().shape[0]
            else:
                rows = X_train.shape[0]
            if rows >= 100000:
                num_rounds = 20
            else:
                num_rounds = 100
            if i == 0:
                if verbose:
                    print('    Number of booster rounds = %s' %num_rounds)
            if train_p.shape[1]-i < 2:
                ### If there is just one variable left, then just skip it #####
                continue
            #### The target must always be  numeric ##
            if model_type == 'Regression':
                objective = 'reg:squarederror'
                params = {'objective': objective, 
                                "silent":True, "verbosity": 0, 'min_child_weight': 0.5}
            else:
                #### This is for Classifiers only ##########                    
                if model_type == 'Binary_Classification':
                    objective = 'binary:logistic'
                    num_class = 1
                    params = {'objective': objective, 'num_class': num_class,
                                    "silent":True,  "verbosity": 0,  'min_child_weight': 0.5}
                else:
                    objective = 'multi:softmax'
                    try:
                        ### This is in case target is a list ###
                        num_class  =  int(np.max(np.unique(train[target]))+1)
                    except:
                        ### This is in case target is a string ###
                        num_class  =  int(np.max(train[target].unique())+1)
                    params = {'objective': objective, 
                                    "silent":True, "verbosity": 0,   'min_child_weight': 0.5, 'num_class': num_class}
            ############################################################################################################
            ######### This is where we find out whether to use single or multi-label for xgboost #######################
            ############################################################################################################
            
            if multi_label_type:
                if model_type == 'Regression':
                    clf = XGBRegressor(n_jobs=-1, n_estimators=100, max_depth=4, random_state=99)
                    clf.set_params(**params)
                    bst = MultiOutputRegressor(clf)
                else:
                    clf = XGBClassifier(n_jobs=-1, n_estimators=100, max_depth=4, random_state=99)
                    clf.set_params(**params)
                    bst = MultiOutputClassifier(clf)
                bst.fit(X_train, y_train)
            else:
                if not dask_xgboost_flag:
                    ################################################################################
                    #########  Training Regular XGBoost on pandas dataframes only ##################
                    ################################################################################
                    #print('cols order: ',cols_sel)
                    try:
                        #### now this training via bst works well for both xgboost 0.0.90 as well as 1.5.1 ##                        
                        dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True, feature_names=cols_sel)
                        bst = xgb.train(params, dtrain, num_boost_round=num_rounds)                
                                    
                    except Exception as error_msg:
                        
                        print('Regular XGBoost is crashing due to: %s' %error_msg)
                        if model_type == 'Regression':
                            params = {'tree_method': cpu_tree_method, 'gpu_id': None}
                        else:
                            params = {'tree_method': cpu_tree_method,'num_class': num_class, 'gpu_id': None}
                        print(error_msg)
                        bst = xgb.train(params, dtrain, num_boost_round=num_rounds)                
                else:
                    ################################################################################
                    ##########   Training XGBoost model using dask_xgboost #########################
                    ################################################################################
                    ### the dtrain syntax can only be used xgboost 1.50 or greater. Dont use it until then.
                    ### use the next line for new xgboost version 1.5.1 abd higher #########
                    try:
                        #### SYNTAX BELOW WORKS WELL. BUT YOU CANNOT DO EVALS WITH DASK XGBOOST AS OF NOW ####
                        #dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
                        #bst = xgb.dask.train(client, params, dtrain, num_boost_round=num_rounds)
                        bst = dask_xgboost_training(X_train, y_train, params)
                    except Exception as error_msg:
                        if model_type == 'Regression':
                            params = {'tree_method': cpu_tree_method}
                        else:
                            params = {'tree_method': cpu_tree_method,'num_class': num_class}
                        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=cols_sel)
                        bst = xgb.dask.train(client=client, params=params, dtrain=dtrain, num_boost_round=num_rounds)
                        print(error_msg)
            ################################################################################
            if not dask_xgboost_flag:
                bst_models.append(bst)
            else:
                bst_models.append(bst['booster'])                
            ##### to get the params of an xgboost booster object you have to do the following steps:
            if verbose >= 3:
                if not dask_xgboost_flag :
                    print('Regular XGBoost model parameters:\n')
                    config = json.loads(bst.save_config())
                else:
                    print('Dask XGBoost model parameters:\n')
                    boo = bst['booster']
                    config = json.loads(boo.save_config())
                print(config)
            #### use this next one for dask_xgboost old ############### 
            if multi_label_type:
                imp_feats = dict(zip(X_train.columns, bst.estimators_[0].feature_importances_))
            else:
                if not dask_xgboost_flag:
                    imp_feats = bst.get_score(fmap='', importance_type='total_gain')
                else:
                    imp_feats = bst['booster'].get_score(fmap='', importance_type='total_gain')
            ### skip the next statement since it is duplicating the work of sort_values ##
            #imp_feats = dict(sorted(imp_feats.items(),reverse=True, key=lambda item: item[1]))
            ### doing this for single-label is a little different from multi_label_type #########
            
            #imp_feats = model_xgb.get_booster().get_score(importance_type='gain')
            #print('%d iteration: imp_feats = %s' %(i+1,imp_feats))
            if len(pd.Series(imp_feats)[pd.Series(imp_feats).sort_values(ascending=False)/pd.Series(imp_feats).values.max()>=0.5]) > 1:
                print_feats = (pd.Series(imp_feats)[pd.Series(imp_feats).sort_values(ascending=False)/pd.Series(imp_feats).values.max()>=0.5]).index.tolist()
                if len(print_feats) < top_num:
                    print_feats = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
                if len(print_feats) <= 30 and verbose:
                    print('        Selected: %s' %print_feats)
                important_features += print_feats
            else:
                print_feats = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
                if len(print_feats) <= 30 and verbose:
                    print('        Selected: %s' %pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist())
                important_features += print_feats
            #######  order this in the same order in which they were collected ######
            important_features = list(OrderedDict.fromkeys(important_features))
            if dask_xgboost_flag:
                if verbose >= 2:
                    print('            Time taken for DASK XGBoost feature selection = %0.0f seconds' %(time.time()-start_time1))
            else:
                if verbose >= 2:
                    print('            Time taken for regular XGBoost feature selection = %0.0f seconds' %(time.time()-start_time1))
        #### plot all the feature importances in a grid ###########
        
        if verbose >= 2:
            if multi_label_type:
                draw_feature_importances_multi_label(bst_models, dask_xgboost_flag)
            else:
                draw_feature_importances_single_label(bst_models, dask_xgboost_flag)
    except Exception as e:
        if dask_xgboost_flag:
            print('Dask XGBoost is crashing due to %s. Returning with currently selected features...' %e)
        else:
            print('Regular XGBoost is crashing due to %s. Returning with currently selected features...' %e)
        important_features = copy.deepcopy(cols_sel)
    return important_features
################################################################################
import multiprocessing
def get_cpu_worker_count():
    return multiprocessing.cpu_count()
#############################################################################################
from itertools import combinations
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
import xgboost
def draw_feature_importances_multi_label(bst_models, dask_xgboost_flag=False):
    rows = int(len(bst_models)/2 + 0.5)
    colus = 2
    fig, ax = plt.subplots(rows, colus)
    fig.set_size_inches(min(colus*5,20),rows*5)
    fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
    fig.subplots_adjust(wspace=0.5) ### This controls the space between columns
    counter = 0
    if rows == 1:
        ax = ax.reshape(-1,1).T
    for k in np.arange(rows):
        for l in np.arange(colus):
            if counter < len(bst_models):
                try:
                    bst_booster = bst_models[counter].estimators_[0]
                    ax1 = xgboost.plot_importance(bst_booster, height=0.8, show_values=False,
                            importance_type='gain', max_num_features=10, ax=ax[k][l])
                    ax1.set_title('Multi_label: Top 10 features for first label: round %s' %(counter+1))
                except:
                    pass
            counter += 1
    plt.show();
########################################################################################
def draw_feature_importances_single_label(bst_models, dask_xgboost_flag=False):
    rows = int(len(bst_models)/2 + 0.5)
    colus = 2
    fig, ax = plt.subplots(rows, colus)
    fig.set_size_inches(min(colus*5,20),rows*5)
    fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
    fig.subplots_adjust(wspace=0.5) ### This controls the space between columns
    counter = 0
    if rows == 1:
        ax = ax.reshape(-1,1).T
    for k in np.arange(rows):
        for l in np.arange(colus):
            if counter < len(bst_models):
                try:
                    bst_booster = bst_models[counter]
                    ax1 = xgboost.plot_importance(bst_booster, height=0.8, show_values=False,
                            importance_type='gain', max_num_features=10, ax=ax[k][l])
                    ax1.set_title('Top 10 features with XGB model %s' %(counter+1))
                except:
                    pass
            counter += 1
    plt.show();
######################################################################################
def reduce_mem_usage(df):
    """
    #####################################################################
    Greatly indebted to :
    https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
        for this function to reduce memory usage.
    #####################################################################
    It is a bit slow as it iterates through all the columns of a dataframe and modifies data types
        to reduce memory usage. But it has been shown to reduce memory usage by 65% or so.       
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if type(df) == dask.dataframe.core.DataFrame:
        start_mem = start_mem.compute()
    print('    Caution: We will try to reduce the memory usage of dataframe from {:.2f} MB'.format(start_mem))
    cols = df.columns
    if type(df) == dask.dataframe.core.DataFrame:
        cols = cols.tolist()

    for col in cols:
        col_type = df[col].dtype
        if col_type != object:
            try:
                c_min = df[col].min()
                c_max = df[col].max()
            except:
                continue
            if type(df) == dask.dataframe.core.DataFrame:
                c_min = c_min.compute()
                c_max = c_max.compute()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                try:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                except:
                    continue
        else:
            df[col] = df[col].astype('category')

    #######  Results after memory usage function ###################
    end_mem = df.memory_usage().sum() / 1024**2
    if type(df) == dask.dataframe.core.DataFrame:
        end_mem = end_mem.compute()
    print(f'        by {(100 * (start_mem - end_mem) / start_mem):.1f}%. Memory usage after is: {end_mem:.2f} MB')    
    return df
##################################################################################
def FE_start_end_date_time_features(smalldf, startTime, endTime, splitter_date_string="/",splitter_hour_string=":"):
    """
    FE stands for Feature Engineering - it means this function performs feature engineering
    ######################################################################################
    This function is used when you have start and end date time stamps in your dataset.
        - If there is no start and end time features, don't use it. Both must be present!
        - this module will create additional features for such fields.
        - you must provide a start date time stamp field and an end date time stamp field
    Otherwise, you are better off using the FE_create_date_time_features() module in this library.

    Inputs:
    smalldf: Dataframe containing your date time fields
    startTime: this is hopefully a string field which converts to a date time stamp easily. Make sure it is a string.
    endTime: this also must be a string field which converts to a date time stamp easily. Make sure it is a string.
    splitter_date_string: usually there is a string such as '/' or '.' between day/month/year etc. Default is assumed / here.
    splitter_hour_string: usually there is a string such as ':' or '.' between hour:min:sec etc. Default is assumed : here.

    Outputs:
    The original pandas dataframe with additional fields created by splitting the start and end time fields
    ######################################################################################
    """
    smalldf = smalldf.copy()
    add_cols = []
    date_time_variable_flag = False
    if smalldf[startTime].dtype in ['datetime64[ns]','datetime16[ns]','datetime32[ns]']:
        print('%s variable is a date-time variable' %startTime)
        date_time_variable_flag = True
    if date_time_variable_flag:
        view_days = 'processing'+startTime+'_elapsed_days'
        smalldf[view_days] = (smalldf[endTime] - smalldf[startTime]).astype('timedelta64[s]')/(60*60*24)
        smalldf[view_days] = smalldf[view_days].astype(int)
        add_cols.append(view_days)
        view_time = 'processing'+startTime+'_elapsed_time'
        smalldf[view_time] = (smalldf[endTime] - smalldf[startTime]).astype('timedelta64[s]').values
        add_cols.append(view_time)
    else:
        start_date = 'processing'+startTime+'_start_date'
        smalldf[start_date] = smalldf[startTime].map(lambda x: x.split(" ")[0])
        add_cols.append(start_date)
        try:
            start_time = 'processing'+startTime+'_start_time'
            smalldf[start_time] = smalldf[startTime].map(lambda x: x.split(" ")[1])
            add_cols.append(start_time)
        except:
            ### there is no hour-minutes part of this date time stamp field. You can just skip it if it is not there
            pass
        end_date = 'processing'+endTime+'_end_date'
        smalldf[end_date] = smalldf[endTime].map(lambda x: x.split(" ")[0])
        add_cols.append(end_date)
        try:
            end_time = 'processing'+endTime+'_end_time'
            smalldf[end_time] = smalldf[endTime].map(lambda x: x.split(" ")[1])
            add_cols.append(end_time)
        except:
            ### there is no hour-minutes part of this date time stamp field. You can just skip it if it is not there
            pass
        view_days = 'processing'+startTime+'_elapsed_days'
        smalldf[view_days] = (pd.to_datetime(smalldf[end_date]) - pd.to_datetime(smalldf[start_date])).values.astype(int)
        add_cols.append(view_days)
        try:
            view_time = 'processing'+startTime+'_elapsed_time'
            smalldf[view_time] = (pd.to_datetime(smalldf[end_time]) - pd.to_datetime(smalldf[start_time])).astype('timedelta64[s]').values
            add_cols.append(view_time)
        except:
            ### In some date time fields this gives an error so skip it in that case
            pass
        #### The reason we chose endTime here is that startTime is usually taken care of by another library. So better to do this alone.
        year = 'processing'+endTime+'_end_year'
        smalldf[year] = smalldf[end_date].map(lambda x: str(x).split(splitter_date_string)[0]).values
        add_cols.append(year)
        #### The reason we chose endTime here is that startTime is usually taken care of by another library. So better to do this alone.
        month = 'processing'+endTime+'_end_month'
        smalldf[month] = smalldf[end_date].map(lambda x: str(x).split(splitter_date_string)[1]).values
        add_cols.append(month)
        try:
            #### The reason we chose endTime here is that startTime is usually taken care of by another library. So better to do this alone.
            daynum = 'processing'+endTime+'_end_day_number'
            smalldf[daynum] = smalldf[end_date].map(lambda x: str(x).split(splitter_date_string)[2]).values
            add_cols.append(daynum)
        except:
            ### In some date time fields the day number is not there. If not, just skip it ####
            pass
        #### In some date time fields, the hour and minute is not there, so skip it in that case if it errors!
        try:
            start_hour = 'processing'+startTime+'_start_hour'
            smalldf[start_hour] = smalldf[start_time].map(lambda x: str(x).split(splitter_hour_string)[0]).values
            add_cols.append(start_hour)
            start_min = 'processing'+startTime+'_start_hour'
            smalldf[start_min] = smalldf[start_time].map(lambda x: str(x).split(splitter_hour_string)[1]).values
            add_cols.append(start_min)
        except:
            ### If it errors, skip it
            pass
        #### Check if there is a weekday and weekends in date time columns using endTime only
        weekday_num = 'processing'+endTime+'_end_weekday_number'
        smalldf[weekday_num] = pd.to_datetime(smalldf[end_date]).dt.weekday.values
        add_cols.append(weekday_num)
        weekend = 'processing'+endTime+'_end_weekend_flag'
        smalldf[weekend] = smalldf[weekday_num].map(lambda x: 1 if x in[5,6] else 0)
        add_cols.append(weekend)
    #### If everything works well, there should be 13 new columns added by module. All the best!
    print('%d columns added using start date=%s and end date=%s processing...' %(len(add_cols),startTime,endTime))
    return smalldf
###########################################################################
def FE_split_one_field_into_many(df_in, field, splitter, filler, new_names_list='', add_count_field=False):
    """
    FE stands for Feature Engineering - it means this function performs feature engineering
    ######################################################################################
    This function takes any data frame field (string variables only) and splits
    it into as many fields as you want in the new_names_list.

    Inputs:
        dft: pandas DataFrame
        field: name of string column that you want to split using the splitter string specified
        splitter: specify what string to split on using the splitter argument.
        filler: You can also fill Null values that may happen due to your splitting by specifying a filler.
        new_names_list: If no new_names_list is given, then we use the name of the field itself to create new columns.
        add_count_field: False (default). If True, it will count the number of items in
            the "field" column before the split. This may be needed in nested dictionary fields.

    Outputs:
        dft: original dataframe with additional columns created by splitting the field.
        new_names_list: the list of new columns created by this function
    ######################################################################################
    """
    df_field = df_in[field].values
    df = copy.deepcopy(df_in)
    ### First copy  whatever is in that field so we can save it for later ###
    ### Remember that fillna only works at dataframe level! ###
    df[[field]] = df[[field]].fillna(filler)
    if add_count_field:
        ### there will be one extra field created when we count the number of contents in each field ###
        max_things = df[field].map(lambda x: len(x.split(splitter))).max() + 1
    else:
        max_things = df[field].map(lambda x: len(x.split(splitter))).max()
    if len(new_names_list) == 0:
        print('    Max. columns created by splitting %s field is %d.' %(
                            field,max_things))
    else:
        if not max_things == len(new_names_list):
            print("""    Max. columns created by splitting %s field is %d but you have given %d
                            variable names only. Selecting first %d""" %(
                        field,max_things,len(new_names_list),len(new_names_list)))
    ### This creates a new field that counts the number of things that are in that field.
    if add_count_field:
        #### this counts the number of contents after splitting each row which varies. Hence it helps.
        num_products_viewed = 'Content_Count_in_'+field
        df[num_products_viewed] = df[field].map(lambda x: len(x.split(splitter))).values
    ### Clean up the field such that it has the right number of split chars otherwise add to it
    ### This fills up the field with empty strings between each splitter. You can't do much about it.
    #### Leave this as it is. It is not something you can do right now. It works.
    fill_string = splitter + filler
    df[field] = df[field].map(lambda x: x+fill_string*(max_things-len(x.split(splitter))) if len(
                                    x.split(splitter)) < max_things else x)
    ###### Now you create new fields by split the one large field ########
    if isinstance(new_names_list, str):
        if new_names_list == '':
            new_names_list = [field+'_'+str(i) for i in range(1,max_things+1)]
        else:
            new_names_list = [new_names_list]
    ### First fill empty spaces or NaNs with filler ###
    df.loc[df[field] == splitter, field] = filler
    for i in range(len(new_names_list)):
        try:
            df[new_names_list[i]] = df[field].map(lambda x: x.split(splitter)[i]
                                          if splitter in x else filler)
        except:
            df[new_names_list[i]] = filler
            continue
    ### there is really nothing you can do to fill up since they are filled with empty strings.
    #### Leave this as it is. It is not something you can do right now. It works.
    df[field] = df_field
    return df, new_names_list
###########################################################################
def FE_add_groupby_features_aggregated_to_dataframe(train,
                    agg_types,  groupby_columns, ignore_variables, test=""):
    """
    FE stands for Feature Engineering. This function performs feature engineering on data.
    ######################################################################################
    ###   This function is a very fast function that will compute aggregates for numerics
    ###   It returns original dataframe with added features from numeric variables aggregated
    ###   What do you mean aggregate? aggregates can be "count, "mean", "median", etc.
    ###   What do you aggregrate? all numeric columns in your data
    ###   What do you groupby? one groupby column at a time or multiple columns one by one
    ###     -- if you give it a list of columns, it will execute the grouping one by one
    ###   What is the ignore_variables for? it will ignore these variables from grouping.
    ###   Make sure to reduce correlated features using FE_remove_variables_using_SULOV_method()
    ######################################################################################
    ### Inputs:
    ###   train: Just sent in the data frame where you want aggregated features for.
    ###   agg_types: list of computational types: 'mean','median','count', 
    ###                     'max', 'min', 'sum', etc.
    ###         One caveat: these agg_types must be found in the following agg_func of 
    ###                   numpy or pandas groupby statement.
    ###         List of aggregates available: {'count','sum','mean','mad','median','min','max',
    ###               'mode','abs', 'prod','std','var','sem','skew','kurt',
    ###                'quantile','cumsum','cumprod','cummax','cummin'}
    ###   groupby_columns: can be a string representing a single column or a list of 
    ###                     multiple columns
    ###               - it will groupby all the numeric features using one groupby column 
    ###                    at a time in a loop.
    ###   ignore_variables: list of variables to ignore among numeric variables in
    ###                data since they may be ID variables.
    ### Outputs:
    ###     Returns the original dataframe with additional features created by this function.
    ######################################################################################
    """
    trainx = copy.deepcopy(train)
    testx = copy.deepcopy(test)
    if isinstance(groupby_columns, str):
        groupby_columns = [groupby_columns]
    numerics = trainx.select_dtypes(include='number').columns.tolist()
    numerics = [x for x in numerics if x not in ignore_variables]
    MGB = Groupby_Aggregator(categoricals=groupby_columns,
            aggregates=agg_types, numerics=numerics)
    train_copy = MGB.fit_transform(trainx)
    if isinstance(testx, str) or testx is None:
        test_copy = testx
    else:
        test_copy = MGB.transform(testx)
    ### return the dataframes ###########
    return train_copy, test_copy
#####################################################################################################
def FE_combine_rare_categories(train_df, categorical_features, test_df=""):
    """
    In this function, we will select all rare classes having representation <1% of population and
    group them together under a new label called 'RARE'. We will apply this on train and test (optional)
    """
    train_df = copy.deepcopy(train_df)
    test_df = copy.deepcopy(test_df)
    train_df[categorical_features] = train_df[categorical_features].apply(
            lambda x: x.mask(x.map(x.value_counts())< (0.01*train_df.shape[0]), 'RARE'))
    for col in categorical_features:
        vals = list(train_df[col].unique())
        if isinstance(test_df, str) or test_df is None:
            return train_df, test_df
        else:
            test_df[col] = test_df[col].apply(lambda x: 'RARE' if x not in vals else x)
            return train_df, test_df

#####################################################################################################
def FE_get_latest_values_based_on_date_column(dft, id_col, date_col, cols, ascending=False):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    ######################################################################################
    This function gets you the latest values of the columns in cols from a date column date_col.

    Inputs:
    dft: dataframe, pandas
    id_col: you need to provide an ID column to groupby the cols and then sort them by date_col.
    date_col: this must be a valid pandas date-time column. If it is a string column,
           make sure you change it to a date-time column.
          It sorts each group by the latest date (descending) and selects that top row.
    cols: these are the list of columns you want their latest value based on the date-col you specify.
         These cols can be any type of column: numeric or string.
    ascending: Set this as True or False depending on whether you want smallest or biggest on top.

    Outputs:
    Returns a dataframe that is smaller than input dataframe since it groups cols by ID_column.
    ######################################################################################
    Beware! You will get a dataframe that has fewer cols than your input with fewer rows than input.
    """
    dft = copy.deepcopy(dft)
    try:
        if isinstance(cols, str):
            cols = [cols]
        train_add = dft.groupby([id_col], sort=False).apply(lambda x: x.sort_values([date_col],
                                                                        ascending=ascending))
        train_add = train_add[cols].reset_index()
        train_add = train_add.groupby(id_col).head(1).reset_index(drop=True).drop('level_1',axis=1)
    except:
        print('    Error in getting latest status of columns based on %s. Returning...' %date_col)
        return dft
    return train_add
#################################################################################
from functools import reduce
def FE_split_add_column(dft, col, splitter=',', action='add'):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    ######################################################################################
    This function will split a column's values based on a splitter you specify and
    will either add them or concatenate them as you specify in the action argument.

    Inputs:
    dft: pandas DataFrame
    col: name of column that you want to split into its constituent parts. It must be a string column.
    splitter: splitter can be any string that is found in your column and that you want to split by.
    action: can be any one of following: {'add', 'subtract', 'multiply', 'divide', 'concat', 'concatenate'}
    ################################################################################
    Returns a dataframe with a new column that is a modification of the old column
    """
    dft = copy.deepcopy(dft)
    new_col = col + '_split_apply'
    print('Creating column = %s using split_add feature engineering...' %new_col)
    if action in ['+','-','*','/','add','subtract','multiply','divide']:
        if action in ['add','+']:
            sign = '+'
        elif action in ['-', 'subtract']:
            sign = '-'
        elif action in ['*', 'multiply']:
            sign = '*'
        elif action in ['/', 'divide']:
            sign = '/'
        else:
            sign = '+'
        # using reduce to compute sum of list
        try:
            trainx = dft[col].astype(str)
            trainx = trainx.map(lambda x:  0 if x is np.nan else 0 if x == '' else x.split(splitter)).map(
                                lambda listx: [int(x) if x != '' else 0 for x in listx ] if isinstance(listx,list) else [0,0])
            dft[new_col] = trainx.map(lambda lis: reduce(lambda a,b : eval('a'+sign+'b'), lis) if isinstance(lis,list) else 0).values
        except:
            print('    Error: returning without creating new column')
            return dft
    elif action in ['concat','concatenate']:
        try:
            dft[new_col] = dft[col].map(lambda x:  " " if x is np.nan else " " if x == '' else x.split(splitter)).map(
                            lambda listx: np.concatenate([str(x) if x != '' else " " for x in listx] if isinstance(listx,list) else " ")).values
        except:
            print('    Error: returning without creating new column')
    else:
        print('Could not perform action. Please check your inputs and try again')
        return dft
    return dft
################################################################################
def FE_add_age_by_date_col(dft, date_col, age_format):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    ######################################################################################
    This handy function gets you age from the date_col to today. It can be counted in months or years or days.
    ######################################################################################
    It returns the same dataframe with an extra column added that gives you age
    """
    if not age_format in ['M','D','Y']:
        print('Age is not given in right format. Must be one of D, Y or M')
        return dft
    new_date_col = 'last_'+date_col+'_in_months'
    try:
        now = pd.Timestamp('now')
        dft[date_col] = pd.to_datetime(dft[date_col], format='%y-%m-%d')
        dft[date_col] = dft[date_col].where(dft[date_col] < now, dft[date_col] -  np.timedelta64(100, age_format))
        if age_format == 'M':
            dft[new_date_col] = (now - dft[date_col]).astype('<m8[M]')
        elif age_format == 'Y':
            dft[new_date_col] = (now - dft[date_col]).astype('<m8[Y]')
        elif age_format == 'D':
            dft[new_date_col] = (now - dft[date_col]).astype('<m8[D]')
    except:
        print('    Error in date formatting. Please check your input and try again')
    return dft
#################################################################################
def FE_count_rows_for_all_columns_by_group(dft, id_col):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    ######################################################################################
    This handy function gives you a count of all rows by groups based on id_col in your dataframe.
    Remember that it counts only non-null rows. Hence it is a different count than other count function.
    ######################################################################################
    It returns a dataframe with id_col as the index and a bunch of new columns that give you counts of groups.
    """
    new_col = 'row_count_'
    if isinstance(id_col, str):
        groupby_columns =  [id_col]
    else:
        groupby_columns = copy.deepcopy(id_col)
    grouped_count = dft.groupby(groupby_columns).count().add_prefix(new_col)
    return grouped_count
#################################################################################
def count_rows_by_group_incl_nulls(dft, id_col):
    """
    ######################################################################################
    This function gives you the count of all the rows including null rows in your data.
    It returns a dataframe with id_col as the index and the counts of rows (incl null rows) as a new column
    ######################################################################################
    """
    new_col = 'row_count_incl_null_rows'
    if isinstance(id_col, str):
        groupby_columns =  [id_col]
    else:
        groupby_columns = copy.deepcopy(id_col)
    ### len gives you count of all the rows including null rows in your data
    grouped_len = dft.groupby(groupby_columns).apply(len)
    grouped_val = grouped_len.values
    grouped_len = pd.DataFrame(grouped_val, columns=[new_col],index=grouped_len.index)
    return grouped_len
#################################################################################
# Can we see if a feature or features has some outliers and how can we cap them?
from collections import Counter
def FE_capping_outliers_beyond_IQR_Range(df, features, cap_at_nth_largest=5, IQR_multiplier=1.5,
                                         drop=False, verbose=False):
    """
    FE at the beginning of function name stands for Feature Engineering. FE functions add or drop features.
    #########################################################################################
    Typically we think of outliers as being observations beyond the 1.5 Inter Quartile Range (IQR)
    But this function will allow you to cap any observation that is multiple of IQR range, such as 1.5, 2, etc.
    In addition, this utility helps you select the value to cap it at.
    The value to be capped is based on "n" that you input.
    n represents the nth_largest number below the maximum value to cap at!
    Notice that it does not put a floor under minimums. You have to do that yourself.
    "cap_at_nth_largest" specifies the max number below the largest (max) number in your column to cap that feature.
    Optionally, you can drop certain observations that have too many outliers in at least 3 columns.
    #########################################################################################
    Inputs:
    df : pandas DataFrame
    features: a single column or a list of columns in your DataFrame
    cap_at_nth_largest: default is 5 = you can set it to any integer such as 1, 2, 3, 4, 5, etc.
    IQR_multiplier: default is 1.5 = but you can set it to any float value such as 1, 1.25. 1.5, 2.0, etc.

    Outputs:
    df: pandas DataFrame
    It returns the same dataframe as you input unless you change drop to True in the input argument.

    Optionally, it can drop certain rows that have too many outliers in at least 3 columns simultaneously.
    If drop=True, it will return a smaller number of rows in your dataframe than what you sent in. Be careful!
    #########################################################################################
    """
    outlier_indices = []
    df = df.copy(deep=True)
    if isinstance(features, str):
        features = [features]
    # iterate over features(columns)
    for col in features:
        ### this is how the column looks now before capping outliers
        if verbose:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
            df[col].plot(kind='box', title = '%s before capping outliers' %col, ax=ax1)
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step using multiplier
        outlier_step = IQR_multiplier * IQR

        lower_limit = Q1 - outlier_step
        upper_limit = Q3 + outlier_step

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < lower_limit) | (df[col] > upper_limit )].index

        ### Capping using the n largest value based on n given in input.
        maxval = df[col].max()  ## what is the maximum value in this column?
        num_maxs = df[df[col]==maxval].shape[0] ## number of rows that have max value
        ### find the n_largest values after the maximum value based on given input n
        num_largest_after_max = num_maxs + cap_at_nth_largest
        capped_value = df[col].nlargest(num_largest_after_max).iloc[-1] ## this is the value we cap it against
        df.loc[df[col]==maxval, col] =  capped_value ## maximum values are now capped
        ### you are now good to go - you can show how they are capped using before and after pics
        if verbose:
            df[col].plot(kind='box', title = '%s after capping outliers' %col, ax=ax2)
            plt.show()

        # Let's save the list of outliers and see if there are some with outliers in multiple columns
        outlier_indices.extend(outlier_list_col)

    # select certain observations containing more than one outlier in 2 columns or more. We can drop them!
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 3 )
    ### now drop these rows altogether ####
    if drop:
        print('Shape of dataframe before outliers being dropped: %s' %(df.shape,))
        number_of_rows = df.shape[0]
        df = df.drop(multiple_outliers, axis=0)
        print('Shape of dataframe after outliers being dropped: %s' %(df.shape,))
        print('\nNumber_of_rows with multiple outliers in more than 3 columns which were dropped = %d' %(number_of_rows-df.shape[0]))
    return df
#################################################################################
def EDA_classify_and_return_cols_by_type(df1):
    """
    EDA stands for Exploratory data analysis. This function performs EDA - hence the name
    ########################################################################################
    This handy function classifies your columns into different types : make sure you send only predictors.
    Beware sending target column into the dataframe. You don't want to start modifying it.
    #####################################################################################
    It returns a list of categorical columns, integer cols and float columns in that order.
    """
    ### Let's find all the categorical excluding integer columns in dataset: unfortunately not all integers are categorical!
    catcols = df1.select_dtypes(include='object').columns.tolist() + df1.select_dtypes(include='category').columns.tolist()
    cats = copy.deepcopy(catcols)
    nlpcols = []
    for each_cat in cats:
        try:
            if df1[[each_cat]].fillna('missing').map(len).mean() >= 40:
                nlpcols.append(each_cat)
                catcols.remove(each_cat)
        except:
            continue
    intcols = df1.select_dtypes(include='integer').columns.tolist()
    # let's find all the float numeric columns in data
    floatcols = df1.select_dtypes(include='float').columns.tolist()
    return catcols, intcols, floatcols, nlpcols
############################################################################################
def EDA_classify_features_for_deep_learning(train, target, idcols):
    """
    ######################################################################################
    This is a simple method of classifying features into 4 types: cats, integers, floats and NLPs
    This is needed for deep learning problems where we need fewer types of variables to transform.
    ######################################################################################
    """
    ### Test Labeler is a very important dictionary that will help transform test data same as train ####
    test_labeler = defaultdict(list)

    #### all columns are features except the target column and the folds column ###
    if isinstance(target, str):
        features = [x for x in list(train) if x not in [target]+idcols]
    else:
        ### in this case target is a list and hence can be added to idcols
        features = [x for x in list(train) if x not in target+idcols]

    ### first find all the types of columns in your data set ####
    cats, ints, floats, nlps = EDA_classify_and_return_cols_by_type(train[features])

    numeric_features = ints + floats
    categoricals_features = copy.deepcopy(cats)
    nlp_features = copy.deepcopy(nlps)

    test_labeler['categoricals_features'] = categoricals_features
    test_labeler['numeric_features'] = numeric_features
    test_labeler['nlp_features'] = nlp_features

    return cats, ints, floats, nlps
#############################################################################################
from itertools import combinations
def FE_create_categorical_feature_crosses(dfc, cats):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    ######################################################################################
    This creates feature crosses for each pair of categorical variables in cats.
    The number of features created will be n*(n-1)/2 which means 3 cat features will create
    3*2/2 = 3 new features. You must be careful with this function so it doesn't create too many.

    Inputs:
    dfc : dataframe containing all the features
    cats: a list of categorical features in the dataframe above (dfc)

    Outputs:
    dfc: returns the dataframe with newly added features. Original features are untouched.

    ######################################################################################
    Usage:
    dfc = FE_create_feature_crosses(dfc, cats)
    """
    dfc = copy.deepcopy(dfc)
    combos = list(combinations(cats, 2))
    for cat1, cat2 in combos:
        dfc.loc[:,cat1+'_cross_'+cat2] = dfc[cat1].astype(str)+" "+dfc[cat2].astype(str)
    return dfc
#############################################################################################
from scipy.stats import probplot,skew
def EDA_find_skewed_variables(dft, skew_limit=1.1):
    """
    EDA stands for Exploratory Data Analysis : this function performs EDA
    ######################################################################################
    This function finds all the highly skewed float (continuous) variables in your DataFrame
    It selects them based on the skew_limit you set: anything over skew 1.1 is the default setting.
    ######################################################################################
    Inputs:
    df: pandas DataFrame
    skew_limit: default 1.1 = anything over this limit and it detects it as a highly skewed var.

    Outputs:
    list of a variables found that have high skew in data set.
    ######################################################################################
    You can use FE_capping_outliers_beyond_IQR_Range() function to cap outliers in these variables.
    """
    skewed_vars = []
    conti = dft.select_dtypes(include='float').columns.tolist()
    for each_conti in conti:
        skew_val=round(dft[each_conti].skew(), 1)
        if skew_val >= skew_limit:
            skewed_vars.append(each_conti)
    print('Found %d skewed variables in data based on skew_limit >= %s' %(len(skewed_vars),skew_limit))
    return skewed_vars
#############################################################################################
def is_outlier(dataframe, thresh=3.5):
    if len(dataframe.shape) == 1:
        dataframe = dataframe[:,None]
    median = np.median(dataframe, axis=0)
    diff = np.sum((dataframe - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

def EDA_find_outliers(df, col, thresh=5):
    """
    """
    ####### Finds Outliers and marks them as 'True' if they are outliers
    ####### Dataframe refers to the input dataframe and threshold refers to how far from the median a value is
    ####### I am using the Median Absolute Deviation Method (MADD) to find Outliers here
    mask_outliers = is_outlier(df[col],thresh=thresh).astype(int)
    return df.iloc[np.where(mask_outliers>0)]
###################################################################################
def outlier_determine_threshold(df, col):
    """
    This function automatically determines the right threshold for the dataframe and column.
    Threshold is used to determine how many outliers we should detect in the series.
    A low threshold will result in too many outliers and a very high threshold will not find any.
    This loops until it finds less than 10 times or maximum 1% of data being outliers.
    """
    df = df.copy(deep=True)
    keep_looping = True
    number_of_loops = 1
    thresh = 5
    while keep_looping:
        if number_of_loops >= 10:
            break
        mask_outliers = is_outlier(df[col], thresh=thresh).astype(int)
        dfout_index = df.iloc[np.where(mask_outliers>0)].index
        pct_outliers = len(dfout_index)/len(df)
        if pct_outliers == 0:
            if thresh > 5:
                thresh = thresh - 5
            elif thresh == 5:
                return thresh
            else:
                thresh = thresh - 1
        elif  pct_outliers <= 0.01:
            keep_looping = False
        else:
            thresh_multiplier = int((pct_outliers/0.01)*0.5)
            thresh = thresh*thresh_multiplier
        number_of_loops += 1
    print('    %s Outlier threshold = %d' %(col, thresh))
    return thresh

from collections import Counter
def FE_find_and_cap_outliers(df, features, drop=False, verbose=False):
    """
    FE at the beginning of function name stands for Feature Engineering. FE functions add or drop features.
    #########################################################################################
    Typically we think of outliers as being observations beyond the 1.5 Inter Quartile Range (IQR)
    But this function will allow you to cap any observation using MADD method:
        MADD: Median Absolute Deviation Method - it's a fast and easy method to find outliers.
    In addition, this utility automatically selects the value to cap it at.
         -- The value to be capped is based on maximum 1% of data being outliers.
    It automatically determines how far away from median the data point needs to be for it to called an outlier.
         -- it uses a thresh number: the lower it is, more outliers. It starts at 5 or higher as threshold value.
    Notice that it does not use a lower bound to find too low outliers. That you have to do that yourself.
    #########################################################################################
    Inputs:
    df : pandas DataFrame
    features: a single column or a list of columns in your DataFrame
    cap_at_nth_largest: default is 5 = you can set it to any integer such as 1, 2, 3, 4, 5, etc.

    Outputs:
    df: pandas DataFrame
    It returns the same dataframe as you input unless you change drop to True in the input argument.

    Optionally, it can drop certain rows that have too many outliers in at least 3 columns simultaneously.
    If drop=True, it will return a smaller number of rows in your dataframe than what you sent in. Be careful!
    #########################################################################################
    """
    df = df.copy(deep=True)
    outlier_indices = []
    idcol = 'idcol'
    df[idcol] = range(len(df))
    if isinstance(features, str):
        features = [features]
    # iterate over features(columns)
    for col in features:
        # Determine a list of indices of outliers for feature col
        thresh = outlier_determine_threshold(df, col)
        mask_outliers = is_outlier(df[col], thresh=thresh).astype(int)
        dfout_index = df.iloc[np.where(mask_outliers>0)].index

        df['anomaly1'] = 0
        df.loc[dfout_index ,'anomaly1'] = 1

        ### this is how the column looks now before capping outliers
        if verbose:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
            colors = {0:'blue', 1:'red'}
            ax1.scatter(df[idcol], df[col], c=df["anomaly1"].apply(lambda x: colors[x]))
            ax1.set_xlabel('Row ID')
            ax1.set_ylabel('Target values')
            ax1.set_title('%s before capping outliers' %col)

        capped_value = df.loc[dfout_index, col].min() ## this is the value we cap it against
        df.loc[dfout_index, col] =  capped_value ## maximum values are now capped
        ### you are now good to go - you can show how they are capped using before and after pics
        if verbose:
            colors = {0:'blue', 1:'red'}
            ax2.scatter(df[idcol], df[col], c=df["anomaly1"].apply(lambda x: colors[x]))
            ax2.set_xlabel('Row ID')
            ax2.set_ylabel('Target values')
            ax2.set_title('%s after capping outliers' %col)

        # Let's save the list of outliers and see if there are some with outliers in multiple columns
        outlier_indices.extend(dfout_index)

    # select certain observations containing more than one outlier in 2 columns or more. We can drop them!
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 3 )
    ### now drop these rows altogether ####
    df = df.drop([idcol,'anomaly1'], axis=1)
    if drop:
        print('Shape of dataframe before outliers being dropped: %s' %(df.shape,))
        number_of_rows = df.shape[0]
        df = df.drop(multiple_outliers, axis=0)
        print('Shape of dataframe after outliers being dropped: %s' %(df.shape,))
        print('\nNumber_of_rows with multiple outliers in more than 3 columns which were dropped = %d' %(number_of_rows-df.shape[0]))
    return df
#################################################################################
import pandas as pd
import numpy as np
import pdb
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
#################################################################################
import copy
from sklearn.cluster import KMeans
def FE_kmeans_resampler(x_train, y_train, target, smote="", verbose=0):
    """
    This function converts a Regression problem into a Classification problem to enable SMOTE!
    This function needs Imbalanced-Learn library. Please pip install it first!
    It is a very simple way to send your x_train, y_train in and get back an oversampled x_train, y_train.
    Why is this needed in Machine Learning problems?
         In Imbalanced datasets, esp. skewed regression problems where the target variable is skewed, this is needed.
    Try this on your skewed Regression problems and see what results you get. It should be better.
    ----------
    Inputs
    ----------
    x_train : pandas dataframe: you must send in the data with predictors only.
    min_n_samples : int, default=5: min number of samples below which you combine bins
    bins : int, default=3: how many bins you want to split target into

    Outputs
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    x_train_c = copy.deepcopy(x_train)
    x_train_c[target] = y_train.values

    # Regression problem turned into Classification problem
    n_clusters = max(3, int(np.log10(len(y_train))) + 1)
    # Use KMeans to find natural clusters in your data
    km_model = KMeans(n_clusters=n_clusters,
                      n_init=5,
                      random_state=99)
    #### remember you must predict using only predictor variables!
    y_train_c = km_model.fit_predict(x_train)

    if verbose >= 1:
        print('Number of clusters created = %d' %n_clusters)

    #### Generate the over-sampled data
    #### ADASYN / SMOTE oversampling #####
    if isinstance(smote, str):
        x_train_ext, _ = oversample_SMOTE(x_train_c, y_train_c)
    else:
        x_train_ext, _ = smote.fit_resample(x_train_c, y_train_c)
    y_train_ext = x_train_ext[target].values
    x_train_ext = x_train_ext.drop(target, axis=1)
    return (x_train_ext, y_train_ext)

###################################################################################################

def oversample_SMOTE(X,y):
    #input DataFrame
    #X →Independent Variable in DataFrame\
    #y →dependent Variable in Pandas DataFrame format
    # Get the class distriubtion for perfoming relative sampling in the next line
    try:
        from imblearn.over_sampling import SVMSMOTE
    except:
        print('This function needs Imbalanced-Learn library. Please pip install it first and try again!')
        return
    class_weighted_rows = get_class_distribution(y)
    smote = SVMSMOTE( random_state=27,
                  sampling_strategy=class_weighted_rows)
    X, y = smote.fit_resample(X, y)
    return(X,y)

def oversample_ADASYN(X,y):
    #input DataFrame
    #X →Independent Variable in DataFrame\
    #y →dependent Variable in Pandas DataFrame format
    # Get the class distriubtion for perfoming relative sampling in the next line
    try:
        from imblearn.over_sampling import ADASYN
    except:
        print('This function needs Imbalanced-Learn library. Please pip install it first and try again!')
        return
    class_weighted_rows = get_class_distribution(y)
    # Your favourite oversampler
    smote = ADASYN(random_state=27,
                   sampling_strategy=class_weighted_rows)
    X, y = smote.fit_resample(X, y)
    return(X,y)
#############################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def split_data_n_ways(df, target, n_splits, test_size=0.2, modeltype=None,**kwargs):
    """
    Inputs:
    df: dataframe that you want to split
    target: the target variable in data frame (df)
    n_splits: number of ways in which you want to split the data frame (default=3)
    test_size: size of the test dataset: default is 0.2 But it splits this test into valid and test half.
    Hence you will get 10% of df as test and 10% of df as valid and remaining 80% as train
    ################   how it works ################################################
    You can split a dataframe three ways or six ways depending on your need. Three ways is:
    train, valid, test
    Six ways can be:
    X_train,y_train, X_valid, y_valid, X_test, y_test
    You will get a list containing these dataframes...depending on what you entered as number of splits
    Output: List of dataframes
    """
    if kwargs:
        for key, val in kwargs:
            if key == 'modeltype':
                key = val
            if key == 'test_size':
                test_size = val
    if modeltype is None:
        if isinstance(target, str):
            if df[target].dtype == float:
                modeltype = 'Regression'
            else:
                modeltype = 'Classification'
            target = [target]
        else:
            if df[target[0]].dtype == float:
                modeltype = 'Regression'
            else:
                 modeltype = 'Classification'
    preds = [x for x in list(df) if x not in target]
    print('Number of predictors in dataset: %d' %len(preds))
    list_of_dfs = []
    if modeltype == 'Regression':
        nums = int((1-test_size)*df.shape[0])
        train, testlarge = df[:nums], df[nums:]
    else:
        train, testlarge = train_test_split(df, test_size=test_size, random_state=42)
    list_of_dfs.append(train)
    if n_splits == 2:
        print('Returning a Tuple with two dataframes and shapes: (%s,%s)' %(train.shape, testlarge.shape))
        return train, testlarge
    elif modeltype == 'Regression' and n_splits == 3:
        nums2 = int(0.5*(testlarge.shape[0]))
        valid, test = testlarge[:nums2], testlarge[nums2:]
        print('Returning a Tuple with three dataframes and shapes: (%s,%s,%s)' %(train.shape, valid.shape, test.shape))
        return train, valid, test
    elif modeltype == 'Classification' and n_splits == 3:
        valid, test = train_test_split(testlarge, test_size=0.5, random_state=99)
        print('Returning a Tuple with three dataframes and shapes: (%s,%s,%s)' %(train.shape, valid.shape, test.shape))
        return train, valid, test
    #### Continue only if you need more than 3 splits ######
    if modeltype == 'Regression':
        nums2 = int(0.5*(df.shape[0] - nums))
        valid, test = testlarge[:nums2], testlarge[nums2:]
        if n_splits == 4:
            X_train, y_train, X_test, y_test = train[preds], train[target], testlarge[preds], testlarge[target]
            list_of_dfs = [X_train,y_train, X_test, y_test]
            print('Returning a Tuple with 4 dataframes: (%s %s %s %s)' %(X_train.shape,y_train.shape,
                                X_test.shape,y_test.shape))
            return list_of_dfs
        elif n_splits == 6:
            X_train, y_train, X_valid, y_valid, X_test, y_test = train[preds], train[target], valid[
                                    preds], valid[target], test[preds], test[target]
            list_of_dfs = [X_train,y_train, X_valid, y_valid, X_test, y_test]
            print('Returning a Tuple with six dataframes and shapes: (%s %s %s %s,%s,%s)' %(
                X_train.shape,y_train.shape, X_valid.shape,y_valid.shape,X_test.shape,y_test.shape))
            return list_of_dfs
        else:
            print('Number of splits must be 2, 3, 4 or 6')
            return
    else:
        if n_splits == 4:
            X_train, y_train, X_test, y_test = train[preds], train[target], testlarge[preds], testlarge[target]
            list_of_dfs = [X_train,y_train, X_test, y_test]
            print('Returning a Tuple with 4 dataframes: (%s %s %s %s)' %(X_train.shape,y_train.shape,
                                X_test.shape,y_test.shape))
            return list_of_dfs
        elif n_splits == 6:
            X_train, y_train, X_valid, y_valid, X_test, y_test = train[preds], train[target], valid[
                                    preds], valid[target], test[preds], test[target]
            print('Returning 4 dataframes:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            list_of_dfs = [X_train,y_train, X_valid, y_valid, X_test, y_test]
            print('Returning a Tuple with six dataframes and shapes: (%s %s %s %s,%s,%s)' %(
                X_train.shape,y_train.shape, X_valid.shape,y_valid.shape,X_test.shape,y_test.shape))
            return list_of_dfs
        else:
            print('Number of splits must be 2, 3, 4 or 6')
            return
##################################################################################
def FE_concatenate_multiple_columns(df, cols, filler=" ", drop=True):
    """
    This handy function combines multiple string columns into a single NLP text column.
    You can do further pre-processing on such a combined column with TFIDF or BERT style embedding.

    Inputs
    ---------
        df: pandas dataframe
        cols: string columns that you want to concatenate into a single combined column
        filler: string (default: " "): you can input any string that you want to combine them with.
        drop: default True. If True, drop the columns input. If False, keep the columns.

    Outputs:
    ----------
        df: there will be a new column called ['combined'] that will be added to your dataframe.
    """
    df = df.copy(deep=True)
    df['combined'] = df[cols].apply(lambda row: filler.join(row.values.astype(str)), axis=1)
    if drop:
        df = df.drop(cols, axis=1)
    return df
##################################################################################
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture

def FE_discretize_numeric_variables(train, bin_dict, test='', strategy='kmeans',verbose=0):
    """
    This handy function discretizes numeric variables into binned variables using kmeans algorithm.
    You need to provide the names of the variables and the numbers of bins for each variable in a dictionary.
    It will return the same dataframe with new binned variables that it has created.

    Inputs:
    ----------
    df : pandas dataframe - please ensure it is a dataframe. No arrays please.
    bin_dict: dictionary of names of variables and the bins that you want for each variable.
    strategy: default is 'kmeans': but you can choose: {'gauusian','uniform', 'quantile', 'kmeans'}

    Outputs:
    ----------
    df: pandas dataframe with new variables with names such as:  variable+'_discrete'
    """
    df = copy.deepcopy(train)
    test = copy.deepcopy(test)
    num_cols = len(bin_dict)
    nrows = int((num_cols/2)+0.5)
    #print('nrows',nrows)
    if verbose:
        fig = plt.figure(figsize=(10,3*num_cols))
    for i, (col, binvalue) in enumerate(bin_dict.items()):
        new_col = col+'_discrete'
        if strategy == 'gaussian':
            kbd = GaussianMixture(n_components=binvalue, random_state=99)
            df[new_col] = kbd.fit_predict(df[[col]]).astype(int)
            if not isinstance(test, str):
                test[new_col] = kbd.predict(test[[col]]).astype(int)
        else:
            kbd = KBinsDiscretizer(n_bins=binvalue, encode='ordinal', strategy=strategy)
            df[new_col] = kbd.fit_transform(df[[col]]).astype(int)
            if not isinstance(test, str):
                test[new_col] = kbd.transform(test[[col]]).astype(int)
        if verbose:
            ax1 = plt.subplot(nrows,2,i+1)
            ax1.scatter(df[col],df[new_col])
            ax1.set_title(new_col)
    if not isinstance(test, str):
        return df, test
    else:
        return df
##################################################################################
def FE_transform_numeric_columns_to_bins(df, bin_dict, verbose=0):
    """
    This handy function discretizes numeric variables into binned variables using kmeans algorithm.
    You need to provide the names of the variables and the numbers of bins for each variable in a dictionary.
    It will return the same dataframe with new binned variables that it has created.

    Inputs:
    ----------
    df : pandas dataframe - please ensure it is a dataframe. No arrays please.
    bin_dict: dictionary of names of variables and the kind of transformation you want
        default is 'log': but you can choose: {'log','log10', 'sqrt', 'max-abs'}

    Outputs:
    ----------
    df: pandas dataframe with new variables with names such as:  variable+'_discrete'
    """
    df = copy.deepcopy(df)
    num_cols = len(bin_dict)
    nrows = int((num_cols/2)+0.5)
    if verbose:
        fig = plt.figure(figsize=(10,3*num_cols))
    for i, (col, binvalue) in enumerate(bin_dict.items()):
        new_col = col+'_'+binvalue
        if binvalue == 'log':
            print('Warning: Negative values in %s have been made positive before log transform!' %col)
            df.loc[df[col]==0,col] = 1e-15  ### make it a small number
            df[new_col] = np.abs(df[col].values)
            df[new_col] = np.log(df[new_col]).values
        elif binvalue == 'log10':
            print('Warning: Negative values in %s have been made positive before log10 transform!' %col)
            df.loc[df[col]==0,col] = 1e-15  ### make it a small number
            df[new_col] = np.abs(df[col].values)
            df[new_col] = np.log10(df[new_col]).values
        elif binvalue == 'sqrt':
            print('Warning: Negative values in %s have been made positive before sqrt transform!' %col)
            df[new_col] = np.abs(df[col].values)  ### make it a small number
            df[new_col] = np.sqrt(df[new_col]).values
        elif binvalue == 'max-abs':
            print('Warning: Negative values in %s have been made positive before max-abs transform!' %col)
            col_max = max(np.abs(df[col].values))
            if col_max == 0:
                col_max = 1
            df[new_col] = np.abs(df[col].values)/col_max
        else:
            print('Warning: Negative values in %s have been made positive before log transform!' %col)
            df.loc[df[col]==0,col] = 1e-15  ### make it a small number
            df[new_col] = np.abs(df[col].values)
            df[new_col] = np.log(df[new_col]).values
        if verbose:
            ax1 = plt.subplot(nrows,2,i+1)
            df[col].plot.kde(ax=ax1, label=col,alpha=0.5,color='r')
            ax2 = ax1.twiny()
            df[new_col].plot.kde(ax=ax2,label=new_col,alpha=0.5,color='b')
            plt.legend();
    return df
#################################################################################
from itertools import cycle, combinations
def FE_create_interaction_vars(df, intxn_vars):
    """
    This handy function creates interaction variables among pairs of numeric vars you send in.
    Your input must be a dataframe and a list of tuples. Each tuple must contain a pair of variables.
    All variables must be numeric. Double check your input before sending them in.
    """
    if type(df) == dask.dataframe.core.DataFrame:
        ## skip if it is a dask dataframe ####
        pass
    else:
        df = df.copy(deep=True)
    combos = combinations(intxn_vars, 2)
    ### I have tested this for both category and object dtypes so don't worry ###
    for (each_intxn1,each_intxn2)  in combos:
        new_col = each_intxn1 + '_x_' + each_intxn2
        try:
            df[new_col] = df[each_intxn1].astype(str) + ' ' + df[each_intxn2].astype(str)
        except:
            continue
    ### this will return extra features generated by interactions ####    
    return df
################################################################################
def FE_create_interaction_vars_train(df, intxn_vars):
    """
    This handy function creates interaction variables among pairs of categorical vars you send in.
    Your input must be a dataframe and a list of tuples. Each tuple must contain a pair of variables.
    All variables must be numeric. Double check your input before sending them in.
    """
    if type(df) == dask.dataframe.core.DataFrame:
        ## skip if it is a dask dataframe ####
        pass
    else:
        df = df.copy(deep=True)
    combos = combinations(intxn_vars, 2)
    ###  This is only for integer vars   ###
    for (each_intxn1,each_intxn2)  in combos:
        new_col = each_intxn1 + '_x_' + each_intxn2
        try:
            df[new_col] = df[each_intxn1].astype(str) + ' ' + df[each_intxn2].astype(str)
        except:
            continue
    ### this will return extra features generated by interactions ####    
    return df
###################################################################################
def FE_create_interaction_vars_test(df, intxn_vars, combo_vars):
    """
    This handy function creates interaction variables among pairs of numeric vars you send in.
    Your input must be a dataframe and a list of tuples. Each tuple must contain a pair of variables.
    All variables must be numeric. Double check your input before sending them in.
    """
    if type(df) == dask.dataframe.core.DataFrame:
        ## skip if it is a dask dataframe ####
        pass
    else:
        df = df.copy(deep=True)
    combos = combinations(intxn_vars, 2)
    ### I have tested this for both category and object dtypes so don't worry ###
    newcols = []
    copy_combos =  copy.deepcopy(combos)
    for (each_intxn1,each_intxn2)  in copy_combos:
        new_col = each_intxn1 + '_x_' + each_intxn2
        newcols.append(new_col)
    left_vars = left_subtract(newcols, combo_vars)
    
    if len(left_vars) > 0:
        for each_left  in left_vars:
            df[each_left] = 0
    for (each_intxn1,each_intxn2)  in combos:
        new_col = each_intxn1 + '_x_' + each_intxn2
        try:
            df[new_col] = df[each_intxn1].astype(str) + ' ' + df[each_intxn2].astype(str)
        except:
            continue
    ### this will return extra features generated by interactions ####    
    return df
##################################################################################
import matplotlib.pyplot as plt
def EDA_binning_numeric_column_displaying_bins(dft, target, bins=4, test=""):
    """
    This splits the data column into the number of bins specified and returns labels, bins, and dataframe.
    Outputs:
       labels = the names of the bins
       edges = the edges of the bins
       dft = the dataframe with an added column called "binned_"+name of the column you sent in
    """
    dft = copy.deepcopy(dft)
    _, edges = pd.qcut(dft[target].dropna(axis=0),q=bins, retbins=True, duplicates='drop')
    ### now we create artificial labels to match the bins edges ####
    ls = []
    for i, x in enumerate(edges):
        #print('i = %s, next i = %s' %(i,i+1))
        if i < len(edges)-1:
            ls.append('from_'+str(round(edges[i],3))+'_to_'+str(round(edges[i+1],3)))
    ##### Next we add a column to hold the bins created by above ###############
    dft['binned_'+target] = pd.cut(dft[target], bins=edges, retbins=False, labels=ls, include_lowest=True).values.tolist()
    if not isinstance(test, str):
        test['binned_'+target] = pd.cut(test[target], bins=edges, retbins=False, labels=ls, include_lowest=True).values.tolist()
    nrows = int(len(edges)/2 + 1)
    plt.figure(figsize=(15,nrows*3))
    plt.subplots_adjust(hspace=.5)
    collect_bins = []
    for i in range(len(edges)):
        if i == 0:
            continue
        else:
            dftc = dft[(dft[target]>edges[i-1]) & (dft[target]<=edges[i])]
            collect_bins.append(dftc)
            ax1 = plt.subplot(nrows, 2, i)
            dftc[target].hist(bins=30, ax=ax1)
            ax1.set_title('bin %d: size: %d, %s %0.2f to %0.2f' %(i, dftc.shape[0], target,
                                                                  edges[i-1], edges[i]))
    return ls, edges, dft, test
#########################################################################################
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import minmax_scale
#### This is where we add other libraries to form a pipeline ###
import copy
import time
import re
from scipy.ndimage import convolve
from sklearn import  datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedShuffleSplit

def add_text_paddings(train_data,nlp_column,glove_filename_with_path,tokenized,
                            fit_flag=True,
                            max_length=100):
    """
    ##################################################################################################
    This function uses a GloVe pre-trained model to add embeddings to your data set.
    ########  I N P U T ##############################:
    data: DataFrame
    nlp_column: name of the NLP column in the DataFrame
    target: name of the target variable in the DataFrame
    glovefile: location of where the glove.txt file is. You must give the full path to that file.
    max_length: specify the dimension of the glove vector  you can have upto the dimension of the glove txt file.
           Make sure you don't exceed the dimension specified in the glove.txt file. Otherwise, error result.
    ####### O U T P U T #############################
    The dataframe is split into train and test and are modified into the specified vector dimension of max_length
    X_train_padded: the train dataframe with dimension specified in max_length
    y_train: the target vector using data and target column
    X_test_padded:  the test dataframe with dimension specified in max_length
    tokenized: This is the tokenizer that was used to split the words in data set. This must be used later.
    ##################################################################################################
    """
    train_index = train_data.index
    ### Encode Train data text into sequences
    train_data_encoded = tokenized.texts_to_sequences(train_data[nlp_column])
    ### Pad_Sequences function is used to make lists of unequal length to stacked sets of padded and truncated arrays
    ### Pad Sequences for Train
    X_train_padded = pad_sequences(train_data_encoded,
                                maxlen=max_length,
                                padding='post',
                               truncating='post')
    print('    Data shape after padding = %s' %(X_train_padded.shape,))
    new_cols = ['glove_dim_' + str(x+1) for x in range(X_train_padded.shape[1])]
    X_train_padded = pd.DataFrame(X_train_padded, columns=new_cols, index=train_index)
    if fit_flag:
        return X_train_padded, tokenized, vocab_size
    else:
        return X_train_padded
#####################################################################################################
def load_embeddings(tokenized,glove_filename_with_path,vocab_size,glove_dimension):
    """
    ##################################################################################################
    # glove_filename_with_path: Make sure u have downloaded and unzipped the GloVe ".txt" file to the location here.
    # we now create a dictionary that maps GloVe tokens to 100, or 200- or 300-dimensional real-valued vectors
    # Then we load the whole embedding into memory. Make sure you have plenty of memory in your machine!
    ##################################################################################################
    """
    MAX_NUM_WORDS = 100000
    glove_path = Path(glove_filename_with_path)
    print('    Creating embeddings. This will take time...')
    embeddings_index = dict()
    for line in glove_path.open(encoding='latin1'):
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        embeddings_index[word] = coefs
    print('Loaded {:,d} Glove vectors.'.format(len(embeddings_index)))
    #There are around 340,000 word vectors that we use to create an embedding matrix
    # that matches the vocabulary so that the RNN model can access embeddings by the token index
    # prepare embedding matrix
    word_index = tokenized.word_index
    embedding_matrix = np.zeros((vocab_size, glove_dimension))
    print('Preparing embedding matrix.')
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('    Completed.')
    return embedding_matrix, glove_dimension
#####################################################################################################
import copy
def FE_convert_mixed_datatypes_to_string(df):
    df = copy.deepcopy(df)
    for col in df.columns:
        if len(df[col].apply(type).value_counts()) > 1:
            print('Mixed data type detected in %s column. Converting all rows to string type now...' %col)
            df[col] = df[col].map(lambda x: x if isinstance(x, str) else str(x)).values
            if len(df[col].apply(type).value_counts()) == 1:
                print('    completed.')
            else:
                print('    could not change column type. Fix it manually and then re-run EDA.')
    return df
##################################################################################
def remove_duplicate_cols_in_dataset(df):
    df = copy.deepcopy(df)
    number_duplicates = df.columns.duplicated().astype(int).sum()
    duplicates = df.columns[df.columns.duplicated()]
    if  number_duplicates > 0:
        print('Removing %d duplicate column(s) of %s' %(number_duplicates, duplicates))
        df = df.loc[:,~df.columns.duplicated()]
    return df
###########################################################################
def FE_split_list_into_columns(df, col, cols_in=[]):
    """
    This is a Feature Engineering function. It will automatically detect object variables that contain lists
    and convert them into new columns. You need to provide the dataframe, the name of the object column.
    Optionally, you can decide to send the names of the new columns you want to create as cols_in.
    It will return the dataframe with additional columns. It will drop the column which you sent in as input.

    Inputs:
    --------
    df: pandas dataframe
    col: name of the object column that contains a list. Remember it must be a list and not a string.
    cols_in: names of the columns you want to create. If the number of columns is less than list length,
             it will automatically choose only the fist few items of the list to match the length of cols_in.
    
    Outputs:
    ---------
    df: pandas dataframe with new columns and without the column you sent in as input.
    """
    df = copy.deepcopy(df)
    if cols_in:
        max_col_length = len(cols_in)
        df[cols_in] = df[col].apply(pd.Series).values[:,:max_col_length]
        df = df.drop(col,axis=1)
    else:
        if len(df[col].map(type).value_counts())==1 and df[col].map(type).value_counts().index[0]==list:
            ### Remember that fillna only works at dataframe level! ###
            max_col_length = df[[col]].fillna('missing').map(len).max()
            cols = [col+'_'+str(i) for i in range(max_col_length)]
            df[cols] = df[col].apply(pd.Series)
            df = df.drop(col,axis=1)
        else:
            print('Column %s does not contain lists or has mixed types other than lists. Fix it and rerun.' %col)
    return df
#############################################################################################
def EDA_randomly_select_rows_from_dataframe(train_dataframe, targets, nrows_limit, DS_LEN=''):
    maxrows = 10000
    train_dataframe = copy.deepcopy(train_dataframe)
    copy_targets = copy.deepcopy(targets)
    if not DS_LEN:
        DS_LEN = train_dataframe.shape[0]
    ####### we randomly sample a small dataset to classify features #####################
    test_size = min(0.9, (1 - (maxrows/DS_LEN))) ### make sure there is a small train size
    if test_size <= 0:
        test_size = 0.9
    ###   Float variables are considered Regression #####################################
    modeltype, _ = analyze_problem_type(train_dataframe[copy_targets], copy_targets, verbose=0)
    ####### If it is a classification problem, you need to stratify and select sample ###
    if modeltype != 'Regression':
        print('    loading a random sample of %d rows into pandas for EDA' %nrows_limit)
        for each_target in copy_targets:
            ### You need to remove rows that have very class samples - that is a problem while splitting train_small
            list_of_few_classes = train_dataframe[each_target].value_counts()[train_dataframe[each_target].value_counts()<=3].index.tolist()
            train_dataframe = train_dataframe.loc[~(train_dataframe[each_target].isin(list_of_few_classes))]
        try:
            train_small, _ = train_test_split(train_dataframe, test_size=test_size, stratify=train_dataframe[targets])
        except:
            ## This split sometimes errors. It is then better to split using a random sample ##
            train_small = train_dataframe.sample(n=nrows_limit, replace=True, random_state=99)
    else:
        ### For Regression problems: load a small sample of data into a pandas dataframe ##
        print('    loading a sequential sample of %d rows into pandas for EDA' %nrows_limit)
        train_small = train_dataframe[:nrows_limit]
    return train_small
################################################################################################
from lazytransform import LazyTransformer
import pdb
class FeatureWiz(BaseEstimator, TransformerMixin):
    """
    FeatureWiz is a feature selection and engineering tool compatible with scikit-learn. 
    It automates the process of selecting the most relevant features for a dataset and 
    supports various encoding, scaling, and data preprocessing techniques.

    Parameters
    ----------
    corr_limit : float, default=0.90
        The correlation limit to consider for feature selection. Features with correlations 
        above this limit may be excluded.

    verbose : int, default=0
        Level of verbosity in output messages.

    feature_engg : str or list, default=''. List = ['interactions', 'groupby', 'target', 
                    'dae', 'vae', 'dae_add', 'vae_add']

        It specifies the feature engineering methods to apply, such as 'interactions', 'groupby', 
        and 'target'. Two new feature engg types have been added:
        1. First is called "dae" (or dae_add) which will call a Denoising Auto Encoder to create
         a low-dimensional representation of the original data by reconstructing the 
         original data from noisy types for a multi-class problem. Use this selectively
          for multi-class or highly imbalanced datasets to improve Classifier
           performance. 'dae' will replace X while 'dae_add' will add features to X.
        2. The second is called "vae" (or vae_add) which stands for Variational Autoencoder (VAE)
          which can be very useful for multi-class problemns. VAE is a type of generative model 
          that not only learns a compressed representation of the data (like a traditional autoencoder)
          but also models the underlying probability distribution of the data. 
          This can be particularly useful in multi-class problems, especially when 
          dealing with complex datasets or when you need to generate new samples for
          data augmentation. 'vae' will replace X while 'vae_add' will add features to X.
        3. If Autoencoders are selected in feature engg, then Recursive XGBoost will be skipped.
          That's because Recursive XGBoost tends to remove too many features and may hurt performance.


    ae_options : must be a dict, default={}. Possible values for auto encoders can be sent
        via this dictionary such as the following examples. You can even use it to GridSearch.
        For 'dae', use this dict: {'encoding_dim': 50, 'noise_factor': 0.1, 'learning_rate': 0.001,
                                    'epochs': 100, 'batch_size': 16, 
                'callbacks':keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10),
                                    'use_simple_architecture': None}
        For 'vae', use this dict: {'intermediate_dim':64, 'latent_dim': 4, 'epochs': 50, 
                                    'batch_size': 16, 'learning_rate': 0.01}
        For 'gan', use this dict: {'input_dim':10, "embedding_dim': 100, 'epochs': 50,
                                    'num_synthetic_samples': }


    category_encoders : str or list, default=''
        Encoders for handling categorical variables. Supported encoders include 'onehot', 
        'ordinal', 'hashing', 'count', 'catboost', 'target', 'glm', 'sum', 'woe', 'bdc', 
        'loo', 'base', 'james', 'helmert', 'label', 'auto', etc.

    add_missing : bool, default=False
        If True, adds indicators for missing values in the dataset.

    dask_xgboost_flag : bool, default=False
        If set to True, enables the use of Dask for parallel computing with XGBoost.

    nrows : int or None, default=None
        Limits the number of rows to process.

    skip_sulov : bool, default=False
        If True, skips the application of the Super Learning Optimized (SULO) method in 
        feature selection.

    skip_xgboost : bool, default=False
        If True, bypasses the recursive XGBoost feature selection.

    transform_target : bool, default=False
        When True, transforms the target variable(s) into numeric format if they are not 
        already.

    scalers : str or None, default=None
        Specifies the scaler to use for feature scaling. Available options include 
        'std', 'standard', 'minmax', 'max', 'robust', 'maxabs'.

    Attributes
    ----------
    features : list
        List of selected features after feature selection process.

    Examples
    --------
    >>> wiz = FeatureWiz(feature_engg = '', nrows=None, transform_target=True, scalers="std",
                category_encoders="auto", add_missing=False, verbose=0)
    >>> X_train_selected, y_train = wiz.fit_transform(X_train, y_train)
    >>> X_test_selected = wiz.transform(X_test)
    >>> selected_features = wiz.features

    Notes
    -----
    - If Autoencoders are selected in feature engg, then Recursive XGBoost will be skipped.
      That's because Recursive XGBoost tends to remove too many features and may hurt performance.
    - The class is built to automatically determine the most suitable encoder and scaler 
      based on the dataset characteristics, unless explicitly specified by the user.
    - FeatureWiz is designed to work with both numerical and categorical variables, 
      applying various preprocessing steps such as missing value flagging, feature 
      engineering, and feature selection.
    - It's important to note that using extensive feature engineering can lead to a 
      significant increase in the number of features, which may affect processing time.

    Raises
    ------
    ValueError
        If inputs are not in the expected format or if invalid parameters are provided.
    """

    def __init__(self, corr_limit=0.90, verbose=0, feature_engg='', 
                    auto_encoders=[], category_encoders='', ae_options={},
                 add_missing=False, dask_xgboost_flag=False, nrows=None, 
                 skip_sulov=False, skip_xgboost=False, transform_target=False, 
                 scalers=None, imbalanced=False, **kwargs):
        """
        Initialize the FeatureWiz class with the given parameters. 
        """
        self.features = None
        self.corr_limit = corr_limit
        self.verbose = verbose
        self.add_missing = add_missing
        self.feature_engg = self._parse_feature_engg(feature_engg)
        self.category_encoders = self._parse_category_encoders(category_encoders)
        #print('    %s parsed as encoders...' %self.category_encoders)
        self.dask_xgboost_flag = dask_xgboost_flag
        self.nrows = nrows
        self.skip_sulov = skip_sulov
        self.skip_xgboost = skip_xgboost
        self.transform_target = transform_target
        if scalers is None:
            self.scalers = ''
        elif isinstance(scalers, str):
            self.scalers = scalers.lower()
        self.imbalanced=imbalanced
        self.auto_encoders = self._parse_auto_encoders(auto_encoders)
        self.ae_options = ae_options
        self._initialize_other_attributes()

    def _parse_feature_engg(self, feature_engg):
        #### This is complicated logic ### be careful changing it!
        if isinstance(feature_engg, str):
            if feature_engg == '':
                return []
            else:
                return [feature_engg]
        elif feature_engg is None:
            return []
        elif isinstance(feature_engg, list):
            return feature_engg
        else:
            print('feature engg must be a list of strings or a string')
            return []

    def _parse_category_encoders(self, encoders):
        approved_encoders = {
            'onehot', 'ordinal', 'hashing', 'count', 'catboost',
            'target', 'glm', 'sum', 'woe', 'bdc', 'loo', 'base',
            'james', 'helmert', 'label', 'auto'
        }
        #### This is complicated logic ### be careful changing it!
        if isinstance(encoders, str):
            if encoders == '':
                encoders = 'auto'
            else:
                ### first create a list and then check for validity of each one ###
                encoders = [encoders]
                encoders = [e for e in encoders if e in approved_encoders]
        #### Leave the next line as "if" - Don't change it to "elif"
        if isinstance(encoders, list):
            if encoders == []:
                encoders = 'auto'
        ###### let us find approved encoders, if not send 'auto' ###
        return encoders

    def _parse_auto_encoders(self, encoders):
        approved_encoders = {
            'dae', 'dae_add', 'vae', 'vae_add', 'cnn', 'cnn_add', 'gan',
        }
        #### This is complicated logic ### be careful changing it!
        if isinstance(encoders, str):
            if encoders == '':
                encoders = []
            else:
                ### first create a list and then check for validity of each one ###
                encoders = [encoders.lower()]
                encoders = [e for e in encoders if e in approved_encoders]
        if isinstance(encoders, list):
                encoders = [e.lower() for e in encoders if e in approved_encoders]
        #### just return them ###
        return encoders

    def _initialize_other_attributes(self):
        self.model_type = ''
        self.grouper = None
        self.targeter = None
        self.numvars = []
        self.catvars = []
        self.missing_flags = []
        self.cols_zero_variance = []
        self.target = None
        self.targets = None
        ### setting autoencoder to None ###
        self.ae = None
        self.interaction_flag = False
        self.poly_feature_adder = None
        encoders_dict = {
                        'OneHotEncoder': 'onehot',
                        'OrdinalEncoder': 'ordinal',
                        'HashingEncoder': 'hashing',
                        'CountEncoder': 'count',
                        'CatBoostEncoder': 'catboost',
                        'TargetEncoder': 'target',
                        'GLMMEncoder': 'glm',
                        'SumEncoder': 'sum',
                        'WOEEncoder': 'woe',
                        'BackwardDifferenceEncoder': 'bdc',
                        'LeaveOneOutEncoder': 'loo',
                        'BaseNEncoder': 'base',
                        'JamesSteinEncoder': 'james',
                        'HelmertEncoder': 'helmert',
                        'label': 'label',
                        'auto': 'auto',
                        }
        approved_encoders = ['onehot','ordinal', 'hashing','count','catboost',
                            'target','glm','sum','woe','bdc','loo','base',
                            'james','helmert', 'label','auto']
        encoders = []
        for each_encoder in self.category_encoders:
            if not each_encoder in approved_encoders:
                enc = encoders_dict.get(each_encoder, 'label')
                encoders.append(enc)
            else:
                ### if they are in approved list check if they chose auto!
                if each_encoder == 'auto':
                    encoders = ['onehot', 'label']
                else:
                    encoders.append(each_encoder)
        print('featurewiz is given %0.1f as correlation limit...' %self.corr_limit)
        if len(encoders) > 2:
            encoders = encoders[:2]
        #print('    %s given as encoders...' %encoders)
        #### This is complicated logic. Be careful before changing it! 
        self.category_encoders = encoders
        feature_generators = ['interactions', 'groupby', 'target', 'poly2', 'poly3']
        feature_gen = []
        if self.feature_engg:
            print('    Warning: Too many features will be generated since feature engg specified')
            if isinstance(self.feature_engg, str):
                self.feature_engg = [self.feature_engg]
            #### Once you have made it into a list, now do all this processing.
            ### Don't change the next line to elif. It needs to be if! I know!
            if isinstance(self.feature_engg, list):
                for each_fe in self.feature_engg:
                    if each_fe in feature_generators:
                        if each_fe == 'target':
                            if self.category_encoders == 'auto':
                                ### Convert it to two encoders from one since they added target encoding
                                self.category_encoders = ['label']
                                self.category_encoders.append(each_fe)
                            else:
                                ### otherwise just add 'target' to the existing list of cat encoders
                                self.category_encoders.append(each_fe)
                            print('    moving target encoder from feature_engg to category_encoders list')
                        else:
                            feature_gen.append(each_fe)
                    else:
                        print('feature engg types must be one or more of: %s. Continuing...' %feature_generators)
                        self.feature_engg = []
                        feature_gen = []
            self.feature_gen = copy.deepcopy(feature_gen)
            print('    final list of feature engineering given: %s' %self.feature_gen)
        else:
            print('    Skipping feature engineering since no feature_engg input...')
            self.feature_gen = []
        #### This is complicated logic. Be careful before changing it! 
        if len(self.category_encoders) == 1:
            self.category_encoders = ['label'] + self.category_encoders
        if self.category_encoders:
            print('    final list of category encoders given: %s' %self.category_encoders)
        
        ### all Auto Encoders need their features to be scaled - MinMax works best!
        try:
            if 'dae' in self.auto_encoders or 'dae_add' in self.auto_encoders:
                self.ae = DenoisingAutoEncoder(**self.ae_options)
                ### Even if user gives input on scalers, then use this as it is the best\
                print('Since Auto Encoders are selected for feature extraction,')
                self.scalers='minmax'
                self.skip_xgboost = True
                if 'dae' in self.auto_encoders:
                    print('    SULOV is skipped since auto-encoders is given...')
                    self.skip_sulov = True
            elif 'vae' in self.auto_encoders or 'vae_add' in self.auto_encoders:
                self.ae = VariationalAutoEncoder(**self.ae_options)
                ### Even if user gives input on scalers, then use this as it is the best
                print('Since Auto Encoders are selected for feature extraction,')
                self.scalers='minmax'
                self.skip_xgboost = True
                if 'vae' in self.auto_encoders:
                    print('    SULOV is skipped since auto-encoders is given...')
                    self.skip_sulov = True
            elif 'cnn' in self.auto_encoders or 'cnn_add' in self.auto_encoders:
                self.ae = CNNAutoEncoder(**self.ae_options)
                ### Even if user gives input on scalers, then use this as it is the best
                print('Since Auto Encoders are selected for feature extraction,')
                self.scalers='minmax'
                self.skip_xgboost = True
                if 'cnn' in self.auto_encoders:
                    print('    SULOV is skipped since auto-encoders is given...')
                    self.skip_sulov = True
            elif 'gan' in self.auto_encoders:
                print('Since Auto Encoders are selected for feature extraction,')
                self.ae = GANAugmenter(**self.ae_options)
                ### If user does not give input on scalers, then use one of your own
                self.scalers='minmax'
                self.skip_xgboost = True
                print('    SULOV is skipped since auto-encoders is given...')
                self.skip_sulov = True
            ### print the options for Auto Encoder if available ##
            if self.ae:
                print('    Recursive XGBoost is also skipped...')
                print('%s\n    AE dictionary given: %s' %(self.ae,
                                         self.ae_options.items()))
        except Exception as e:
            print('ae_options erroring due to %s. Please check documentation and try again.' %e)

        print('    final list of scalers given: [%s]' %self.scalers)
        #### Now you can set up the parameters for Lazy Transformer

        self.lazy = LazyTransformer(model=None, encoders=self.category_encoders, 
            scalers=self.scalers, date_to_string=False,
            transform_target=self.transform_target, imbalanced=self.imbalanced, 
            save=False, combine_rare=False, verbose=self.verbose)

    def fit(self, X, y):
        max_cats = 10
        if isinstance(X, np.ndarray):
            print('X input must be a dataframe since we use column names to build data pipelines. Returning')
            return X, y
        if isinstance(y, np.ndarray):
            print('   y input is an numpy array and hence convert into a series or dataframe and re-try.')
            return X, y
        ### Now you can process the X and y datasets ####
        if isinstance(y, pd.Series):
            self.target = y.name
            self.targets = [self.target]
            if self.target is None:
                print('   y input is a pandas series with no name. Convert it and re-try.')
                return X, y                
        elif isinstance(y, pd.DataFrame):
            self.target = y.columns.tolist()
            self.targets = y.columns.tolist()
        elif isinstance(X, np.ndarray):
            print('y must be a pd.Series or pd.DataFrame since we use column names to build data pipeline. Returning')
            return {}, {}
        ######################################################################################
        #####      MAKING FEATURE_TYPE AND FEATURE_GEN SELECTIONS HERE           #############
        ######################################################################################
        X_sel = copy.deepcopy(X)
        print('Loaded input data. Shape = %s' %(X_sel.shape,))
        ##### This where we find the features  to modify ######################
        preds = [x for x in list(X_sel) if x not in self.targets]
        ###   This is where we sort the columns to make sure that the order of columns doesn't matter in selection ###########
        numvars = X_sel[preds].select_dtypes(include = 'number').columns.tolist()
        self.numvars = numvars
        if self.verbose:
            print('    selecting %d numeric features for further processing...' %len(numvars))
        catvars = left_subtract(preds, numvars)
        self.catvars = catvars
        if len(self.catvars) > max_cats:
            print('    Warning: Too many features will be generated since categorical vars > %s. This may take time...' %max_cats)
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        X_sel, y_sel = self.transform(X, y)
        return X_sel, y_sel

    def transform(self, X, y=None):
        start_time = time.time()
        
        if y is None:
            ##########################################################
            ############# This is only for test data #################
            ##########################################################
            ### Now you can process the X dataset ####
            print('#### Starting featurewiz transform for test data ####')
            if isinstance(X, np.ndarray):
                print('X must be a pd.Series or pd.DataFrame since we use column names to build data pipeline. Returning')
                return {}
            print('Loaded input data. Shape = %s' %(X.shape,))
            if self.add_missing:
                print('    Caution: add_missing adds a missing flag column for every column in your dataset. Beware...')
                X = add_missing(X)
                print('    transformed dataset shape: %s' %(X.shape,))
            if self.feature_gen:
                print('    Beware! feature_engg will add 100s, if not 1000s of additional features to your dataset!')
                if np.where('groupby' in self.feature_gen,True, False).tolist():
                    if not self.grouper is None:
                        X = self.grouper.transform(X)

                if np.where('interactions' in self.feature_gen,True, False).tolist():
                    X = FE_create_interaction_vars_train(X, self.catvars)
            #### This is where we add polynomial features to data set #####
            if np.where('poly2' in self.feature_gen,True, False).tolist() or np.where('poly3' in self.feature_gen,True, False).tolist():
                if len(self.numvars) > 1:
                    # Transform the training and test data
                    X = self.poly_feature_adder.transform(X)
            else:
                ### If there is no poly but at least have interaction_only, then do this!
                if self.interaction_flag:
                    if len(self.numvars) > 1:
                        # Transform the training and test data
                        X = self.poly_feature_adder.transform(X)

            ### this is only for test data ######
            print('#### Starting lazytransform for test data ####')
            X_sel = self.lazy.transform(X)

            ### Sometimes the index becomes huge after imabalanced flag is set!
            X_index = X_sel.index
            #### This is where you transform using the Denoising Auto Encoder
            if not self.ae is None:
                #### It is okay if y is None ########
                if not 'gan' in self.auto_encoders:
                    ### This is for VAE, CNN and DAE #####
                    X_sel_ae = self.ae.transform(X_sel)
                else:
                    ### if it is GAN just return the dataframe as it is
                    return X_sel[self.features]

                if np.all(np.isnan(X_sel_ae)):
                    print('Auto encoder is erroring. Using existing features shape: %s' %(X_sel.shape,))
                else:
                    ### Since this results in a higher dimension you need to create new columns ##
                    new_vars = ['ae_feature_'+str(x+1) for x in range(X_sel_ae.shape[1])]
                    X_sel_ae = pd.DataFrame(X_sel_ae, columns=new_vars, index=X_index)
                    if 'dae_add' in self.auto_encoders or 'vae_add' in self.auto_encoders or 'cnn_add' in self.auto_encoders:
                        ### Only if add is specified do you add the features to X
                        ## Since this results in a higher dimension you need to create new columns ##
                        old_vars = list(X_sel)
                        X_sel = pd.concat([X_sel, X_sel_ae], axis=1)
                        X_sel.columns = old_vars+new_vars
                    else:
                        ### Just replace X_sel with X_sel_ae ###
                        X_sel = copy.deepcopy(X_sel_ae)
                    print('Shape of transformed data due to auto encoder = %s' %(X_sel.shape,))

            ### return either fitted features or all features depending on error ###
            if len(self.cols_zero_variance) > 0:
                if self.verbose:
                    print('    Dropping %d columns due to zero variance: %s' %(len(
                                    self.cols_zero_variance), self.cols_zero_variance))
                X_sel = X_sel.drop(self.cols_zero_variance, axis=1)
            print('Returning dataframe with %d features ' %len(self.features))
            
            try:
                return X_sel[self.features]
            except:
                print('Returning dataframe with all features since error in feature selection...')
                return X_sel
        else:
            ##########################################################
            ###################### this is only for train data #######
            ##########################################################
            print('#### Starting featurewiz transform for train data ####')
            X_sel = copy.deepcopy(X)
            X_index = X.index
            y_index = y.index
            #############    This adds a missing flag column for each column ############
            if self.add_missing:
                print('    Caution: add_missing adds a missing flag column for every column in your dataset. Beware...')
                orig_vars = X_sel.columns.tolist()
                X_sel = add_missing(X_sel)
                self.missing_flags = left_subtract(X_sel.columns.tolist(), orig_vars)
                print('    transformed dataset shape: %s' %(X_sel.shape,))
            ##################   This is where we do groupby features    #################
            if self.feature_gen:
                if np.where('groupby' in self.feature_gen,True, False).tolist():
                    if len(self.catvars) >= 1 and len(self.numvars) >= 1:
                        #### We make sure that only those numvars and catvars in X are used. Not the missing flags!
                        grp = Groupby_Aggregator(categoricals=self.catvars, aggregates=['mean'], numerics=self.numvars)
                        X_sel = grp.fit_transform(X_sel)
                        self.grouper = grp
                    else:
                        print('No groupby features created since no categorical or numeric vars in dataset.')
                else:
                    print('No groupby features created since no groupby feature engg specified')
                ##################  This is where we test for feature interactions ###########
                if np.where('interactions' in self.feature_gen,True, False).tolist():
                    if len(self.catvars) > 1:
                        num_combos = len(list(combinations(self.catvars, 2)))
                        print('Adding %s interactions between %s categorical_vars %s...' %(
                                            num_combos, len(self.catvars), self.catvars))
                        #### We make sure that only those numvars and catvars in X are used. Not the missing flags!
                        X_sel = FE_create_interaction_vars_train(X_sel, self.catvars)
                        #### Since missing flags are not included in numvars, we are adding them here to select the rest
                        combovars = left_subtract(X_sel.columns.tolist(), self.numvars+self.missing_flags)
                        self.combovars = combovars
                    else:
                        print('No interactions created for categorical vars since number less than 2')
                    ### Set the interaction flag anyway since X may have numeric vars!
                    self.interaction_flag = True
                else:
                    print('No interactions created for categorical vars since no interactions feature engg specified')

                ##################  This is where we add Polynomial interactions ###########
                if np.where('poly2' in self.feature_gen,True, False).tolist(): 
                    if self.interaction_flag:
                        self.interaction_flag = False ### Interaction only will be set to false
                    if len(self.numvars) > 1:
                        # Create the transformer
                        poly_feature_adder = PolyFeatureAdder(degree=2, interaction_only=self.interaction_flag)
                        # Transform the training and test data
                        X_sel = poly_feature_adder.fit_transform(X_sel)
                        self.poly_feature_adder = poly_feature_adder
                elif np.where('poly3' in self.feature_gen,True, False).tolist():
                    if self.interaction_flag:
                        self.interaction_flag = False ### Interaction only will be set to false
                    if len(self.numvars) > 1:
                        # Fit to the training data
                        poly_feature_adder = PolyFeatureAdder(degree=3, interaction_only=self.interaction_flag)
                        # Transform the training and test data
                        X_sel = poly_feature_adder.fit_transform(X_sel)
                        self.poly_feature_adder = poly_feature_adder
                else:
                    ### If there is no Poly, then you need to create interaction_only variables!
                    if self.interaction_flag:
                        poly_feature_adder = PolyFeatureAdder(degree=2, interaction_only=self.interaction_flag)
                        # Transform the training and test data
                        X_sel = poly_feature_adder.fit_transform(X_sel)
                        self.poly_feature_adder = poly_feature_adder
                    print('No polynomial vars created since no flag set or less than 2 numeric vars in dataset')

            ##### Now put a dataframe together of transformed X and y  #### 
            X_sel.index = X_index

            #### Use lazytransform to transform all variables to numeric ###
            X_sel, y_sel = self.lazy.fit_transform(X_sel, y)
            
            ### Sometimes after imbalanced flag, this index becomes different!
            X_index = X_sel.index
            y_index = y_sel.index

            #### Now check if y is transformed properly ###############
            if isinstance(y_sel, np.ndarray):
                if isinstance(y, pd.Series):
                    y_sel = pd.Series(y_sel, name=self.target, index=y_index)
                elif isinstance(y, pd.DataFrame):
                    y_sel = pd.Series(y_sel, columns=self.targets, index=y_index)
                else:
                    print('y is not formatted correctly: check your input y and try again.')
                    return self

            ## Fit DAE or VAE transformer to the data. This method trains the autoencoder model.
            new_epochs = 150  # Set the desired number of epochs
            if not self.ae is None:
                ##### Now we set up the hyper params for DAE if it is selected ########
                if 'dae' in self.auto_encoders or 'dae_add' in self.auto_encoders or 'vae' in self.auto_encoders or 'vae_add' in self.auto_encoders:
                    if not self.ae_options:
                        # Perform Hyperparam selection only when ae_options is empty ###
                        start_time = time.time()
                        print('Performing hyper param selection for DAE. Will take approx. %s seconds' %(27*4))
                        if 'dae' in self.auto_encoders or 'dae_add' in self.auto_encoders:
                            grid_search = dae_hyperparam_selection(self.ae, X_sel, y_sel)
                        elif 'vae' in self.auto_encoders or 'vae_add' in self.auto_encoders:
                            grid_search = vae_hyperparam_selection(self.ae, X_sel, y_sel)
                        best_model = grid_search.best_estimator_
                        best_model = best_model.named_steps['feature_extractor']
                        ## Print the best parameters
                        print('    time taken for hyper param selection = %0.0f seconds' %(time.time()-start_time))
                        print(grid_search.best_params_)
                        # Assuming best_model is the best estimator from GridSearchCV
                        # Update the epochs parameter of the feature_extractor
                        best_model.epochs = new_epochs
                        self.ae = best_model
                    else:
                        print('    No hyperparam selection since ae_options is given')
                else:
                    print('    No hyperparam selection since GAN or CNN is selected for auto_encoders...')
                ### since you cannot fit model before transforming data, leave it here ###
                if 'gan' in self.auto_encoders:
                    print('Fitting and transforming a GAN for each class...')
                    #### You must fit separately and then transform. Otherwise, you get errors in transform.
                    self.ae.fit(X_sel, y_sel)
                    X_sel_ae, y_sel = self.ae.transform(X_sel, y_sel)
                elif ('dae' in self.auto_encoders) | ('dae_add' in self.auto_encoders):
                    print('Fitting and transforming DenoisingAutoEncoder for dataset...')
                    self.ae.fit(X_sel)
                    X_sel_ae = self.ae.transform(X_sel)
                elif ('cnn' in self.auto_encoders) | ('cnn_add' in self.auto_encoders):
                    print('Fitting and transforming CNNAutoEncoder for dataset...')
                    X_sel_ae = self.ae.fit_transform(X_sel)
                elif ('vae' in self.auto_encoders) | ('vae_add' in self.auto_encoders):
                    #### You need X and y for Variational Auto Encoders training #####
                    print('Fitting and transforming an Auto Encoder for dataset...')
                    self.ae.fit(X_sel, y_sel)
                    X_sel_ae = self.ae.transform(X_sel, y_sel)
                #### After transforming X_sel, you need to figure out whether to add it or not ####
                if np.all(np.isnan(X_sel_ae)):
                    print('Auto encoder is erroring. Using existing features: %s' %(X_sel.shape,))
                else:
                    ### You need to check for both 'vae' and 'vae_add': this does both!!
                    if [y for y in self.auto_encoders if 'dae' in y] or [y for y in self.auto_encoders if 'vae' in y]:
                        new_vars = ['ae_feature_'+str(x+1) for x in range(X_sel_ae.shape[1])]
                        ## Since this results in a higher dimension you need to create new columns ##
                        X_sel_ae = pd.DataFrame(X_sel_ae, columns=new_vars, index=X_index)
                    elif [y for y in self.auto_encoders if 'cnn' in y] :
                        new_vars = ['ae_feature_'+str(x+1) for x in range(X_sel_ae.shape[1])]
                        ## Since this results in a higher dimension you need to create new columns ##
                        X_sel_ae = pd.DataFrame(X_sel_ae, columns=new_vars, index=X_index)
                    else:
                        #### This is for GAN only since it doesn't add columns but adds rows! ###
                        old_rows = X_index.max()
                        add_rows = len(X_sel_ae) - len(X_index) 
                        new_vars = X_sel.columns
                        X_index = np.concatenate((X_index, np.arange(old_rows+1, old_rows+add_rows+1)))
                        X_sel_ae = pd.DataFrame(X_sel_ae, columns=new_vars, index=X_index)
                        y_index = X_sel_ae.index
                        y_sel = pd.DataFrame(y_sel, columns=self.targets, index=y_index)
                    #### Don't change this next line since it applies new rules to above! ###
                    if 'dae_add' in self.auto_encoders or 'vae_add' in self.auto_encoders or 'cnn_add' in self.auto_encoders:
                        ## Since this results in a higher dimension you need to create new columns ##
                        old_vars = list(X_sel)
                        X_sel = pd.concat([X_sel, X_sel_ae], axis=1)
                        X_sel.columns = old_vars+new_vars
                    else:
                        ### Just replace X_sel with X_sel_ae for all other values ###
                        X_sel = copy.deepcopy(X_sel_ae)
                    print('Shape of transformed data due to auto encoder = %s' %(X_sel.shape,))
            #####  Put the dataframe together #######################
            if (X_index == y_index).all():
                if isinstance(X_sel, pd.DataFrame) and (isinstance(y_sel, pd.DataFrame) or isinstance(y_sel, pd.Series)):
                    df = pd.concat([X_sel, y_sel], axis=1)
                else:
                    print('X and y are not pandas dataframes or series. Check your input and try again')
                    return X, y
            else:
                df = pd.concat([X_sel.reset_index(drop=True), y_sel], axis=1)
                df.index = X_index
            # Select features using featurewiz
            self.model_type, self.multi_label_type = analyze_problem_type(df[self.targets], self.targets)
            #### This is where you need to drop columns that have zero variance ######
            self.cols_zero_variance = X_sel.columns[(X_sel.var()==0)]
            if len(self.cols_zero_variance) > 0:
                if self.verbose:
                    print('    Dropping %d columns due to zero variance: %s' %(
                        len(self.cols_zero_variance), self.cols_zero_variance))
                X_sel = X_sel.drop(self.cols_zero_variance, axis=1)
                df = df.drop(self.cols_zero_variance, axis=1)
            self.numvars = X_sel.columns.tolist()
            if not self.skip_sulov:
                self.numvars = FE_remove_variables_using_SULOV_method(df, self.numvars, self.model_type, self.targets,
                                     self.corr_limit, self.verbose, self.dask_xgboost_flag)
            #### Now you need to send the selected numvars to next stage  ###
            if not self.skip_xgboost:
                print('Performing recursive XGBoost feature selection from %d features...' %len(self.numvars))
                features = FE_perform_recursive_xgboost(df[self.numvars+self.targets], self.targets, 
                                self.model_type, self.multi_label_type, 
                                self.dask_xgboost_flag, self.verbose)
            else:
                features = copy.deepcopy(self.numvars)
            # find the time taken to run feature selection ####
            difftime = max(1, np.int16(time.time()-start_time))
            print('    time taken to run entire featurewiz = %s second(s)' %difftime)
            # column of labels
            self.features = features
            print('Recursive XGBoost selected %d features...' %len(self.features))
            return X_sel[self.features], y_sel
###################################################################################################
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import copy
from sklearn import __version__ as sklearn_version
class PolyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, interaction_only=False, include_bias=False):
        self.poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, 
                                       include_bias=include_bias)
        self.feature_names = None

    def fit(self, X, y=None):
        # We only want numeric columns for polynomial features
        self.numeric_cols = X.select_dtypes(include='number').columns
        self.poly.fit(X[self.numeric_cols])
        # Check sklearn version
        if sklearn_version >= '1.0':  # Since get_feature_names_out was added in version 1.0
            self.feature_names = self.poly.get_feature_names_out(self.numeric_cols)
        else:
            self.feature_names = self.poly.get_feature_names(self.numeric_cols)
        return self

    def transform(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X for Polynomial features must be a pandas DataFrame.")
        X = copy.deepcopy(X)
        X_index = X.index

        # Apply polynomial transformation
        poly_features = self.poly.transform(X[self.numeric_cols])
        df_poly = pd.DataFrame(poly_features, columns=self.feature_names, index=X_index)

        # Drop original columns and join the new features
        X_transformed = X.drop(self.numeric_cols, axis=1).join(df_poly, rsuffix='_extra')

        return X_transformed
##############################################################################################
def EDA_remove_special_chars(df):
    """
    This function removes special chars from column names and returns a df with new column names.
    Inputs and outputs are both the same dataframe except column names are changed.
    """
    import copy
    import re
    cols = df.columns.tolist()
    copy_cols = copy.deepcopy(cols)
    ser = pd.Series(cols)
    ### This function removes all special chars from a list ###
    remove_special_chars =  lambda x:re.sub('[^A-Za-z0-9_]+', '', x)
    newls = ser.map(remove_special_chars).values.tolist()
    df.columns = newls
    return df
###################################################################################################
def dask_xgboost_training(X_trainx, y_trainx, params):
    
    cluster = dask.distributed.LocalCluster()
    dask_client = dask.distributed.Client(cluster)
    X_trainx = dd.from_pandas(X_train, npartitions=1)
    y_trainx = dd.from_pandas(y_train, npartitions=1)
    print("DASK XGBoost training...")
    dtrain = xgb.dask.DaskDMatrix(dask_client, X_trainx, y_trainx)
    bst = xgb.dask.train(dask_client, params, dtrain, num_boost_round=10)
    dask_client.close()
    print("    training completed...")
    return bst
####################################################################################
def FE_remove_commas_in_numerics(train, nums=[]):
    """ 
    This function removes commas in numeric columns and returns the columns transformed.
    You can send in a dataframe with one column name as a string or a list of columns.
    Returns a single array if only one column is sent.
    Returns the entire dataframe if a list of columns is sent. This includes all columns.
    """
    train = copy.deepcopy(train)
    if isinstance(nums, str):
        return train[each_num].map(lambda x: float("".join( x.split(",")))).values
    else:
        for each_num in nums:
            train[each_num] = train[each_num].map(lambda x: float("".join( x.split(",")))).values
    return train
####################################################################################
### this works only on pandas dataframes but it is extremely fast
import copy
def FE_calculate_duration_from_timestamp(df, id_column, timestamp_column):
    """
    ###################################################################################
    Calculate the total time and average time spent online per day by user. 
    Also it calculates the number of logins per user per day.
    ###   This is very useful for logs data, IOT data, and pings from visits data #####
    This function takes a DataFrame with user ids and timestamps (of logins, etc.) 
    and returns a DataFrame with duration or the time spent between two timestamps.
    This is calculated by taking pairs of rows and assuming the first row is login
    and the second row is logout. Then we subtract the timestamp of the login from 
    the timestamp of the logout for each pair of rows. This function uses alternate rows
    of a dataframe and splits them into separate columns. It then subtracts the two 
    columns to find time delta in seconds during those two times. It also eliminates
    any data entry errors by removing negative durations. This is the best and 
    speediest way to calculate online time spent per user per day.
    ###################################################################################
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame with user ids, timestamps and values.
    id_column : str
        The name of the column that contains the user ids.
    timestamp_column : str
        Name of the timestamp column

    Returns
    -------
    result : pandas.DataFrame
        The output DataFrame with user ids, dates, average time spent and number of logins.
    """
    df = copy.deepcopy(df)
    # Create an empty DataFrame to store the results
    columns = [id_column, timestamp_column]
    df = df[columns]
    leng = len(df)
    # Reshape the DataFrame into two columns by stacking every other row
    df1 = df.iloc[::2] # select every even row
    df2 = df.drop(0, axis=0) # drop the first row
    df2 = df2.iloc[::2] # select every even row from the remaining rows
    # If length of dataframe is not an even number, process until the last row
    if leng%2 != 0:
        lastrow = dict(df.iloc[-1]) # get the last row as a dictionary
        df2 = df2.append(lastrow, ignore_index=True) # append it to df2
    df1 = df1.rename(columns={timestamp_column:timestamp_column+'_begin'}) # rename the timestamp column in df1
    df2 = df2.rename(columns={timestamp_column:timestamp_column+'_end'}) # rename the timestamp column in df2
    df1x = df1.reset_index(drop=True) # reset the index of df1
    df2x = df2.reset_index(drop=True) # reset the index of df2
    df3 = pd.concat([df1x, df2x], axis=1) # concatenate df1 and df2 horizontally
    result = df3.iloc[:,[0,1,3]] # select only the relevant columns from df3
    # calculate the time difference between each pair of rows
    result["time_diff"] = result[timestamp_column+"_end"] - result[timestamp_column+"_begin"] 
    #convert the value_diff column to seconds using np.timedelta64(1, 's') function
    result['time_diff'] = result['time_diff'] / np.timedelta64(1, 's')
    result.loc[(result['time_diff']<0),"time_diff"] = 0
    
    #return the result
    return result
###############################################################################################
from pandas.api.types import is_object_dtype
from sklearn.impute import MissingIndicator
def add_missing(df):
    """
        ####   Missing values indicator - it adds missing flag to all columns in a dataframe #########
        ### It does not make sense to add an indicator when train has no missing values and test does.
        ### In such cases, you will have an extra column in test while there won't be in train
        ### So it is better to create this extra column for all columns in a dataframe so that 
        ### train and test data sets have same features when creating feature transformer pipelines.
        ##############################################################################################

    """
    df = copy.deepcopy(df)
    df_index = df.index
    col_names = df.columns
    col_names = [x+'_missing' for x in col_names]
    if is_object_dtype(df.columns):
        miss = MissingIndicator(features="all")
        df_add = pd.DataFrame(miss.fit_transform(df).astype(np.int8), index=df_index, columns=col_names)
        df = df.join(df_add)
        return df
    else:
        print('Column names must be strings in dataframe. Returning as is...')
        return df
###############################################################################################
import copy
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold
from lazytransform import print_classification_metrics, print_regression_metrics
def cross_val_model_predictions(model, train, test, targets, modeltype, 
                    feature_engg=[], cv=None, splits=5, feature_selection=False):
    """
    Conducts cross-validation and generates predictions using a specified machine learning model.

    This function performs cross-validation on the provided training data and generates predictions for both training
    and test datasets. It supports both regression and classification models, including special handling for
    XGBoost models. The function allows for different cross-validation strategies and handles feature engineering
    and target transformation if required.

    Parameters:
    - model: A scikit-learn compatible model object. This can be any model that follows scikit-learn's API.
    - train (pd.DataFrame): Training data as a Pandas DataFrame. It should include both features and the target variable.
    - test (pd.DataFrame): Test data as a Pandas DataFrame. It should include features but not the target variable.
    - targets (list): A list containing the names of the target variable(s) in the train DataFrame.
    - modeltype (string): A string defining the type of model whether "Regression", "Binary_Classification" or "Multi-Classification"
    - feature_engg (list): default is []. You can add "interactions", "groupby", "target", or all three features to your model.
    - cv (Optional): A cross-validation strategy object. If None, KFold with 5 splits is used by default.
    - splits: the number of splits to be used in CV strategy. Default is 5
    - feature_selection: To use or not use feature_selection using featurewiz. Default is False.

    The function performs the following steps:
    1. Initializes various variables and sets up the cross-validation strategy.
    2. Performs feature engineering using the FeatureWiz library if necessary.
    3. Trains the model on each fold of the cross-validation and makes predictions.
    4. Evaluates the model performance using appropriate metrics.
    5. Generates predictions for the test set.
    
    Returns:
    - test_preds (np.array): Array of predictions for the test set.
    - test_probabs (np.array): Array of prediction probabilities for the test set (if applicable).

    Note:
    - The function assumes that the input data frames (train and test) are pre-processed and ready for model training.
    - For XGBoost models, the number of estimators is handled automatically. For other models, a default of 200 estimators is used.
    - The function uses FeatureWiz for feature selection and transformation, which needs to be installed separately.

    Raises:
    - ValueError: If the model type is not recognized or if there are issues with the data input.

    Example usage:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> val_scores, test_preds, test_probabs = cross_val_model_predictions(model, train_df, test_df, 
                                                targets=['target'], modeltype='Regression', 
                                                feature_engg=['groupby'], feature_selection=False,
                                                cv=None, splits=5)
    """
    seed = 42
    enco = 'catboost'
    np.random.seed(seed)    
    X = train.copy(deep=True)
    y = X.pop(targets[0])
    test_copy = test.copy(deep=True)
    ### define test set ###
    X_test = test_copy.drop(targets[0], axis=1)
    ### Do this only if the model is XGBoost ###
    if str(model).split("(")[0] == 'XGBRegressor' or str(model).split("(")[0] == 'XGBClassifier' :
        n_ests = model.get_params()['n_estimators']
    else:
        n_ests = 200
    ### if cv is None, just use KFold ###
    if cv is None:
        if modeltype == 'Regression':
            cv = KFold(n_splits = splits, random_state = 99, shuffle = True)
        else:
            cv = StratifiedKFold(n_splits = splits, random_state = 99, shuffle = True)

    #initiate prediction arrays and score lists
    train_predictions = None
    val_predictions = None
    test_predictions = None
    test_probas = None

    train_scores, val_scores = [], []

    #training model, predicting prognosis probability, and evaluating log loss
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, train[targets])):
        print('##################   Fold %s processing   ###############################' %(fold+1))
        #define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        #define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        if feature_selection:
            fwiz = FeatureWiz(
                corr_limit=0.9,
                feature_engg=feature_engg,
                category_encoders=enco,
                add_missing=False,
                nrows=None,
                verbose=0,
                transform_target=True,
                scalers="std",
            )
            X_train, y_train = fwiz.fit_transform(
                X=X_train,
                y=y_train,)

            X_val  = fwiz.transform(X_val)
            ### This transforms y_test alone without touching X_test. Nice trick!
            if modeltype != 'Regression':
                y_val = fwiz.lazy.yformer.transform(y_val)

        else:
            ### use lazy transform with default settings ###########
            lazy = LazyTransformer(model=None, encoders='label', scalers='', 
                                transform_target=True, imbalanced=False, verbose=1)
            X_train, y_train = lazy.fit_transform(X_train, y_train)
            X_val = lazy.transform(X_val)
            ### This transforms y_test alone without touching X_test. Nice trick!
            if modeltype != 'Regression':
                y_val = lazy.yformer.transform(y_val)

        #train model
        if str(model).split("(")[0] == 'XGBRegressor' or str(model).split("(")[0] == 'XGBClassifier' :
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=int(0.2*n_ests), verbose=0, )
        else:
            model.fit(X_train, y_train)

        #make predictions
        if modeltype == 'Regression':
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
        else:
            train_preds = model.predict_proba(X_train)
            val_preds = model.predict_proba(X_val)

        if fold == 0:
            if modeltype == 'Regression':
                train_predictions = copy.deepcopy(train_preds)
                val_predictions = copy.deepcopy(val_preds)
            elif modeltype == 'Binary_Classification':
                train_predictions = copy.deepcopy(train_preds[:,1])
                val_predictions = copy.deepcopy(val_preds[:,1])
            else:
                train_predictions = copy.deepcopy(train_preds)
                val_predictions = copy.deepcopy(val_preds)
        else:
            if modeltype == 'Regression':
                train_predictions =  np.hstack([train_predictions, train_preds])
                val_predictions =  np.hstack([val_predictions, val_preds])
            elif modeltype == 'Binary_Classification':
                train_predictions = np.hstack([train_predictions, train_preds[:,1]])
                val_predictions = np.hstack([val_predictions, val_preds[:,1]])
            else:
                train_predictions = np.vstack([train_predictions, train_preds])
                val_predictions = np.vstack([val_predictions, val_preds])

        #evaluate model for a fold
        if modeltype == 'Regression':
            print('Model results on Train data:')
            train_score = print_regression_metrics(y_train, train_preds)
            print('Model results on Validation data:')
            val_score = print_regression_metrics(y_val, val_preds)
        else:
            print('Model results on Train data:')
            train_score = print_classification_metrics(y_train, model.predict(X_train), train_preds)
            print('Model results on Validation data:')
            val_score = print_classification_metrics(y_val, model.predict(X_val), val_preds)

        #append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)
        
        ### make your predictions now
        if feature_selection:
            X_test_trans  = fwiz.transform(X_test)
        else:
            X_test_trans  = lazy.transform(X_test)
            
        if fold == 0:
            if modeltype == 'Regression':
                test_predictions = model.predict(X_test_trans) / splits
                test_probas = copy.deepcopy(test_predictions)
            elif modeltype == 'Binary_Classification':
                test_predictions = model.predict(X_test_trans) / splits
                test_probas = model.predict_proba(X_test_trans) / splits
            else:
                test_predictions = model.predict(X_test_trans) / splits
                test_probas = model.predict_proba(X_test_trans) / splits
        else:
            if modeltype == 'Regression':
                test_predictions = np.vstack([test_predictions, model.predict(X_test_trans) / splits])
                test_probas = copy.deepcopy(test_predictions)
            elif modeltype == 'Binary_Classification':
                test_predictions = np.dstack([test_predictions, model.predict(X_test_trans) / splits])
                test_probas = np.dstack([test_probas, model.predict_proba(X_test_trans) / splits])
            else:
                test_predictions = np.dstack([test_predictions, model.predict(X_test_trans) / splits])
                test_probas = np.dstack([test_probas, model.predict_proba(X_test_trans) / splits])

    print(f'Val Scores average: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Scores average: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {targets}')
    if modeltype == 'Regression':
        test_preds = np.sum(test_predictions, axis=0)
        test_probabs = copy.deepcopy(test_predictions)
    elif modeltype == 'Binary_Classification':
        test_preds = np.sum(test_predictions, axis=2).astype(int).squeeze()
        test_probabs = np.sum(test_probas,axis=2)
    else:
        test_preds = np.sum(test_predictions, axis=2).astype(int).squeeze()
        test_probabs = np.sum(test_probas,axis=2)
    return test_preds, test_probabs
#########################################################################################################

