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
import pandas as pd
import numpy as np
np.random.seed(99)
import random
random.seed(42)
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
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
from .my_encoders import Groupby_Aggregator, My_LabelEncoder_Pipe
from .my_encoders import Rare_Class_Combiner, Rare_Class_Combiner_Pipe
from . import settings
settings.init()
################################################################################
#### The warnings from Sklearn are so annoying that I have to shut it off #######
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
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
def load_file_dataframe(dataname, sep=",", header=0, verbose=0, nrows=None,parse_dates=False):
    start_time = time.time()
    ###########################  This is where we load file or data frame ###############
    if isinstance(dataname,str):
        if dataname == '':
            print('    No file given. Continuing...')
            return None
        #### this means they have given file name as a string to load the file #####
        codex_flag = False
        codex = ['ascii', 'utf-8', 'iso-8859-1', 'cp1252', 'latin1']
        ## this is the total number of rows in df  ###
        try:
            max_rows = lenopenreadlines(dataname)
        except:
            ### if for some reason, it errors, just set a default size to read file
            max_rows = 10000
        if nrows is None: ### this means get all rows -> don't skip any one
            skip_function = None  ## will not skip any rows
        else:
            get_rows = max_rows  ## this is the number of rows we want to sample from the df
            rem_rows = max_rows-get_rows  ## this will be the number of rows we need to skip
            skip_function = random.sample(range(max_rows), rem_rows)  ## will generate a random list of rows to skip
        if dataname != '' and dataname.endswith(('csv')):
            try:
                ### You can read the entire data into pandas first and then stratify split it ##
                ###   If you don't stratify it, then 
                dfte = pd.read_csv(dataname, sep=sep, header=header, encoding=None, 
                                parse_dates=parse_dates)
                if not nrows is None:
                    if nrows < dfte.shape[0]:
                        print('    max_rows_analyzed is smaller than dataset shape %d...' %dfte.shape[0])
                        dfte = dfte.sample(nrows, replace=False, random_state=99)
                        print('        randomly sampled %d rows from read CSV file' %nrows)
                if len(np.array(list(dfte))[dfte.columns.duplicated()]) > 0:
                    print('You have duplicate column names in your data set. Removing duplicate columns now...')
                    dfte = dfte[list(dfte.columns[~dfte.columns.duplicated(keep='first')])]
                return dfte
            except:
                codex_flag = True
            if codex_flag:
                for code in codex:
                    try:
                        dfte = pd.read_csv(dataname, sep=sep, header=header, encoding=None, nrows=nrows,
                                        skiprows=skip_function, parse_dates=parse_dates) 
                    except:
                        print('    pandas %s encoder does not work for this file. Continuing...' %code)
                        continue
        elif dataname.endswith(('xlsx','xls','txt')):
            #### It's very important to get header rows in Excel since people put headers anywhere in Excel#
            if nrows is None:
                dfte = pd.read_excel(dataname,header=header, parse_dates=parse_dates)
            else:
                dfte = pd.read_excel(dataname,header=header, nrows=nrows, parse_dates=parse_dates)
            print('    Shape of your Data Set loaded: %s' %(dfte.shape,))
            return dfte
        elif dataname.endswith(('gzip', 'bz2', 'zip', 'xz')):
            print('    Reading compressed file...')
            try:
                #### Dont use skip_function in zip files #####
                compression = 'infer'
                dfte = pd.read_csv(dataname, sep=sep, header=header, encoding=None, nrows=nrows,
                                compression=compression, parse_dates=parse_dates) 
                return dfte
            except:
                print('    Could not read compressed file. Please unzip and try again...')
                return None
    elif isinstance(dataname,pd.DataFrame):
        
        #### this means they have given a dataframe name to use directly in processing #####
        if nrows is None:
            dfte = copy.deepcopy(dataname)
        else:
            if nrows < dataname.shape[0]:
                print('    Since nrows is smaller than dataset, loading random sample of %d rows into pandas...' %nrows)
                dfte = dataname.sample(n=nrows, replace=False, random_state=99)
            else:
                dfte = copy.deepcopy(dataname)
        print('    Shape of your Data Set loaded: %s' %(dfte.shape,))
        if len(np.array(list(dfte))[dfte.columns.duplicated()]) > 0:
            print('You have duplicate column names in your data set. Removing duplicate columns now...')
            
            dfte = dfte[list(dfte.columns[~dfte.columns.duplicated(keep='first')])]
        return dfte
    else:
        print('Dataname input must be a filename with path to that file or a Dataframe')
        return None
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
        print('    Converted pandas dataframe into a Dask dataframe ...' )
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
###########################################################################################
############## CONVERSION OF STRING COLUMNS TO NUMERIC using MY_LABELENCODER #########
#######################################################################################
def FE_convert_all_object_columns_to_numeric(train, test="", features=[]):
    """
    FE stands for Feature Engineering - it means this function performs feature engineering
    ######################################################################################
    This is a utility that converts string columns to numeric using MY_LABEL ENCODER.
    Make sure test and train have the same number of columns. If you have target in train,
    remove it before sending it through this utility. Otherwise, might blow up during test transform.
    The beauty of My_LabelEncoder is it handles NA's and future values in test that are not in train.
    #######################################################################################
    Inputs:
    train : pandas dataframe
    test: (optional) pandas dataframe

    Outputs:
    train: this is the transformed DataFrame
    test: (optional) this is the transformed test dataframe if given.
    ######################################################################################
    """
    
    train = copy.deepcopy(train)
    test = copy.deepcopy(test)
    #### This is to fill all numeric columns with a missing number ##########
    nums = train.select_dtypes('number').columns.tolist()
    nums = [x for x in nums if x in features]
    #### We don't want to look for ID columns and deleted columns ########
    if len(nums) == 0:
        pass
    else:

        if train[nums].isnull().sum().sum() > 0:
            null_cols = np.array(nums)[train[nums].isnull().sum()>0].tolist()
            for each_col in null_cols:
                new_missing_col = each_col + '_Missing_Flag'
                train[new_missing_col] = 0
                train.loc[train[each_col].isnull(),new_missing_col]=1
                ### Remember that fillna only works at dataframe level! ###
                train[[each_col]] = train[[each_col]].fillna(-9999)
                if not train[each_col].dtype in [np.float64,np.float32,np.float16]:
                    train[each_col] = train[each_col].astype(int)
                if not isinstance(test, str):
                    if test is None:
                        pass
                    else:
                        new_missing_col = each_col + '_Missing_Flag'
                        test[new_missing_col] = 0
                        test.loc[test[each_col].isnull(),new_missing_col]=1
                        test[each_col] = test[each_col].fillna(-9999)
                        if not test[each_col].dtype in [np.float64,np.float32,np.float16]:
                            test[each_col] = test[each_col].astype(int)
    ###### Now we convert all object columns to numeric ##########
    lis = []
    error_columns = []
    
    lis = train.select_dtypes('object').columns.tolist() + train.select_dtypes('category').columns.tolist()
    if not isinstance(test, str):
        if test is None:
            pass
        else:
            lis_test = test.select_dtypes('object').columns.tolist() + test.select_dtypes('category').columns.tolist()
            if len(left_subtract(lis, lis_test)) > 0:
                ### if there is an extra column in train that is not in test, then remove it from consideration
                lis = copy.deepcopy(lis_test)
    if not (len(lis)==0):
        for everycol in lis:
            MLB = My_LabelEncoder()
            try:
                train[everycol] = MLB.fit_transform(train[everycol])
                if not isinstance(test, str):
                    if test is None:
                        pass
                    else:
                        test[everycol] = MLB.transform(test[everycol])
            except:
                print('    error converting %s column from string to numeric. Continuing...' %everycol)
                error_columns.append(everycol)
                continue
    
    return train, test, error_columns
###################################################################################
from sklearn.base import TransformerMixin, BaseEstimator
from collections import defaultdict
class My_LabelEncoder(TransformerMixin):
    """
    ################################################################################################
    ######  This Label Encoder class works just like sklearn's Label Encoder!  #####################
    #####  You can label encode any column in a data frame using this new class. But unlike sklearn,
    the beauty of this function is that it can take care of NaN's and unknown (future) values.
    It uses the same fit() and fit_transform() methods of sklearn's LabelEncoder class.
    ################################################################################################
    ##################### This is the best working version - don't mess with it! ###################
    Usage:
          MLB = My_LabelEncoder()
          train[column] = MLB.fit_transform(train[column])
          test[column] = MLB.transform(test[column])
    """
    def __init__(self):
        self.transformer = defaultdict(str)
        self.inverse_transformer = defaultdict(str)

    def fit(self,testx):
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            return testx
        outs = np.unique(testx.factorize()[0])
        ins = testx.value_counts(dropna=False).index
        #if -1 in outs:
        #   it already has nan if -1 is in outs. No need to add it.
        #    ins.insert(0,np.nan)
        self.transformer = dict(zip(ins,outs.tolist()))
        self.inverse_transformer = dict(zip(outs.tolist(),ins))
        return self

    def transform(self, testx):
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            return testx
        ### now convert the input to transformer dictionary values
        ins = np.unique(testx.factorize()[1]).tolist()
        missing = [x for x in ins if x not in self.transformer.keys()]
        if len(missing) > 0:
            for each_missing in missing:
                max_val = np.max(list(self.transformer.values())) + 1
                self.transformer[each_missing] = max_val
                self.inverse_transformer[max_val] = each_missing
        outs = testx.map(self.transformer).values
        testk = testx.map(self.transformer)
        if testx.dtype not in [np.int16, np.int32, np.int64, float, bool, object]:
            if testx.isnull().sum().sum() > 0:
                fillval = self.transformer[np.nan]
                testk = testk.cat.add_categories([fillval])
                testk = testk.fillna(fillval)
                testk = testk.astype(int)
                return testk
            else:
                testk = testk.astype(int)
                return testk
        else:
            return outs

    def inverse_transform(self, testx):
        ### now convert the input to transformer dictionary values
        if isinstance(testx, pd.Series):
            outs = testx.map(self.inverse_transformer).values
        elif isinstance(testx, np.ndarray):
            outs = pd.Series(testx).map(self.inverse_transformer).values
        else:
            outs = testx[:]
        return outs
##################################################################################
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from .databunch import DataBunch
from .encoders import FrequencyEncoder

from sklearn.model_selection import train_test_split
def featurewiz(dataname, target, corr_limit=0.7, verbose=0, sep=",", header=0,
            test_data='', feature_engg='', category_encoders='', dask_xgboost_flag=False,
            nrows=None,  **kwargs):
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
    print('############################################################################################')
    print('############       F A S T   F E A T U R E  E N G G    A N D    S E L E C T I O N ! ########')
    print("# Be judicious with featurewiz. Don't use it to create too many un-interpretable features! #")
    print('############################################################################################')
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
        print('Skipping feature engineering since no feature_engg input...')
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
    print('**INFO: featurewiz can now read feather formatted files. Loading train data...')
    if isinstance(dataname, str):
        #### This is where we get a filename as a string as an input #################
        if re.search(r'(.ftr)', dataname) or re.search(r'(.feather)', dataname):
            print("""**INFO: Feather format allowed. Loading feather formatted file...**""")
            import feather
            dataname = pd.read_feather(dataname, use_threads=True)
            train = load_dask_data(dataname, sep)
        else:
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
                train = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, nrows=nrows)
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
            train = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, nrows=nrows)
            if (train.memory_usage().sum()/1000000) > mem_limit:
                dataname = reduce_mem_usage(train)
            else:
                dataname = copy.deepcopy(train)
    print('    Loaded train data. Shape = %s' %(dataname.shape,))
    if dataname.shape[0] >= 1e6:
        ### we are going to set a hard limit of 900K for nows since some datasets are huge and break
        nrows = 900000
        dataname = dataname.sample(n=nrows, replace=False)
        print(' setting a hard limit of 900K samples for train since some it is huge and breaks pandas...')

    ##################    L O A D    T E S T   D A T A      ######################
    dataname = remove_duplicate_cols_in_dataset(dataname)

    #### Convert mixed data types to string data type  ############################
    #dataname = FE_convert_mixed_datatypes_to_string(dataname)
    
    ######   XGBoost cannot handle special chars in column names ###########
    old_col_names = dataname.columns.tolist()
    new_col_names, special_char_flag = make_column_names_unique(dataname)
    if special_char_flag:
        dataname.columns = new_col_names
        if dask_xgboost_flag:
            train = load_dask_data(dataname, sep)

    ###### Now save the old and new columns in a dictionary to use them later ###
    col_name_mapper = dict(zip(new_col_names, old_col_names))
    col_name_replacer = dict([(y,x) for (x,y) in col_name_mapper.items()])
    item_replacer = col_name_replacer.get  # For faster gets.

    #### You need to change the target name if you have changed the column names ### 
    if special_char_flag:
        if isinstance(target, str):
            targets = [target]
            targets = [item_replacer(n, n) for n in targets]
            target = targets[0]
        else:
            target = [item_replacer(n, n) for n in target]

    train_index = dataname.index
    
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
                print("""**INFO: to increase file loading performance, convert huge `csv` files to `feather` format using `df.to_feather("path/to/save/file.feather")`**""")
                print('**INFO: featurewiz can now read feather formatted files...***')
                ### only if test_data is a filename load this #####
                print('Loading test data filename = %s...' %test_data)
                if dask_xgboost_flag:
                    print('    Since dask_xgboost_flag is True, reducing memory size and loading into dask')
                    ### nrows does not apply to test data in the case of featurewiz ###############
                    test_data = load_file_dataframe(test_data, sep=sep, header=header, verbose=verbose, nrows=None)
                    ### sometimes, test_data returns None if there is an error. ##########
                    if test_data is not None:
                        test_data = reduce_mem_usage(test_data)
                        ### test_data is the pandas dataframe object and test is dask dataframe object ##
                        test = load_dask_data(test_data, sep)
                else:
                    #### load the entire test dataframe - there is no limit applicable there #########
                    test_data = load_file_dataframe(test_data, sep=sep, header=header, verbose=verbose)
                    test = copy.deepcopy(test_data)
        else:
            print('No test data filename given...')
            test_data = None
            test = None
    else:
        print('loading the entire test dataframe - there is no nrows limit applicable #########')
        test_data = load_file_dataframe(test_data, sep=sep, header=header, verbose=verbose)
        test = copy.deepcopy(test_data)
    ### sometimes, test_data returns None if there is an error. ##########
    if test_data is not None:
        test_data = remove_duplicate_cols_in_dataset(test_data)
        test_index = test_data.index
        print('    Loaded test data. Shape = %s' %(test_data.shape,))
        #######  Once again remove the same in test data as well ###
        new_col_test, special_char_flag_test = make_column_names_unique(test_data)
        if special_char_flag_test:
            test_data.columns = new_col_test
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
        train_small = select_rows_from_dataframe(dataname, targets, nrows_limit, DS_LEN=dataname.shape[0])
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
        print('Caution: Since there are date-time variables in datatset, it is best to load them using pandas')
        dask_xgboost_flag = False ### Set the dask flag to be False since it is now becoming Pandas dataframe 
        date_time_vars = features_dict['date_vars']
        dataname = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, nrows=nrows,
                             parse_dates=date_time_vars)
        if (dataname.memory_usage().sum()/1000000) > mem_limit:
            dataname = reduce_mem_usage(dataname)
        train = load_dask_data(dataname, sep)
        if not test_data is None:
            ### You must load the entire test data - there is no limit there ##################
            ### test_data is the pandas dataframe object and test is dask dataframe object ##
            test_data = load_file_dataframe(test_data, sep=sep, header=header, verbose=verbose, 
                                 parse_dates=date_time_vars)
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
    GPU_exists = check_if_GPU_exists()
    n_workers = get_cpu_worker_count()
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    cpu_tree_method = 'hist'
    tree_method = 'hist'
    cpu_params['nthread'] = -1
    cpu_params['tree_method'] = tree_method
    cpu_params['eta'] = 0.01
    cpu_params['subsample'] = 0.5
    cpu_params['grow_policy'] = 'depthwise' #'lossguide'
    cpu_params['max_depth'] = max_depth
    cpu_params['max_leaves'] = 0
    cpu_params['verbosity'] = 0
    cpu_params['gpu_id'] = 0
    cpu_params['updater'] = 'grow_colmaker'
    cpu_params['predictor'] = 'cpu_predictor'
    cpu_params['num_parallel_tree'] = 1
    if GPU_exists:
        tree_method = 'gpu_hist'
        param['nthread'] = -1
        param['tree_method'] = tree_method
        param['eta'] = 0.01
        param['subsample'] = 0.5
        param['grow_policy'] = 'depthwise' # 'lossguide' # 
        param['max_depth'] = max_depth
        param['max_leaves'] = 0
        param['verbosity'] = 0
        param['gpu_id'] = 0
        param['updater'] = 'grow_gpu_hist' #'prune'
        param['predictor'] = 'gpu_predictor'
        param['num_parallel_tree'] = 1
        print('    Tuning XGBoost using GPU hyper-parameters. This will take time...')
    else:
        param = copy.deepcopy(cpu_params)
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
    preds = [x for x in list(dataname) if x not in target+cols_to_remove]
    print('    After removing redundant variables from further processing, features left = %d' %len(preds))
    numvars = dataname[preds].select_dtypes(include = 'number').columns.tolist()
    ######   F I N D I N G    C A T  V A R S   H E R E  ##################################
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
            print('Adding %s interactions between categorical_vars %s...' %(
                                num_combos, catvars))
            dataname = FE_create_interaction_vars(dataname, catvars)
            train = FE_create_interaction_vars(train, catvars)
            catvars = left_subtract(dataname.columns.tolist(), numvars)
            catvars = left_subtract(catvars, target)
            preds =  left_subtract(dataname.columns.tolist(), target)
            if not test_data is None:
                test_data = FE_create_interaction_vars(test_data, catvars)
                test = FE_create_interaction_vars(test, catvars)
        else:
            print('No interactions created for categorical vars since number less than 2')
    else:
        print('No interactions created for categorical vars since feature engg does not specify it')
    ##### Now we need to re-set the catvars again since we have created new features #####
    rem_vars = copy.deepcopy(catvars)
    ########## Now we need to select the right model to run repeatedly ####
    
    if target is None or len(target) == 0:
        cols_list = list(dataname)
        settings.modeltype, _ = 'Clustering'
    else:
        settings.modeltype, _ = analyze_problem_type(dataname[target], target)
        cols_list = left_subtract(list(dataname),target)
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
        if settings.modeltype != 'Regression':
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
            dataname = data1.append(data2)
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
    ###### This top_num decides how many top_n features XGB selects in each iteration.
    ####  There a total of 5 iterations. Hence 5x10 means maximum 50 features will be selected.
    #####  If there are more than 50 variables, then maximum 25% of its variables will be selected
    if len(preds) <= 50:
        top_num = 10
    else:
        ### the maximum number of variables will 25% of preds which means we divide by 5 and get 5% here
        ### The five iterations result in 10% being chosen in each iteration. Hence max 50% of variables!
        top_num = int(len(preds)*0.15)
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
    if len(numvars) > 1:
        if data_dim < 50:
            try:
                final_list = FE_remove_variables_using_SULOV_method(dataname,numvars,settings.modeltype,target,
                             corr_limit,verbose, dask_xgboost_flag)
            except:
                print('    SULOV method is erroring. Continuing ...')
                final_list = copy.deepcopy(numvars)
        else:
                print('    Skipping SULOV method since data dimension %s m > 50 m. Continuing ...' %int(data_dim))
                final_list = copy.deepcopy(numvars)            
    else:
        print('    Skipping SULOV method since there are no continuous vars. Continuing ...')
        final_list = copy.deepcopy(numvars)
    ####### This is where you draw how featurewiz works when the verbose = 2 ###########
    print('Time taken for SULOV method = %0.0f seconds' %(time.time()-start_time1))
    #### Now we create interaction variables between categorical features ########
    print('    Adding %s categorical variables to reduced numeric variables  of %d' %(
                                len(important_cats),len(final_list)))
    if  isinstance(final_list,np.ndarray):
        final_list = final_list.tolist()
    preds = final_list+important_cats
    print('Final list of selected vars after SULOV = %s' %len(preds))
    #######You must convert category variables into integers ###############    
    print('Readying dataset for Recursive XGBoost by converting all features to numeric...')
    if len(important_cats) > 0:
        dataname, test_data, error_columns = FE_convert_all_object_columns_to_numeric(dataname,  test_data, preds)
        important_cats = left_subtract(important_cats, error_columns)
        if len(error_columns) > 0:
            print('    removing %s object columns that could not be converted to numeric' %len(error_columns))
        else:
            print('    there were no mixed data types or object columns that errored. Data is all numeric...')
        dataname.drop(error_columns, axis=1, inplace=True)
        print('Shape of train data after pruning = %s' %(dataname.shape,) )
        if not test_data is None:
            test_data.drop(error_columns, axis=1, inplace=True)
            print('    Shape of test data after pruning = %s' %(test_data.shape,) )
    print('#######################################################################################')
    print('#####    R E C U R S I V E   X G B O O S T : F E A T U R E   S E L E C T I O N  #######')
    print('#######################################################################################')
    important_features = []
    #######################################################################
    #####   This is for DASK XGB Regressor and XGB Classifier problems ####
    #######################################################################
    if settings.multi_label:
        ### only regular xgboost with multi-output can work well in multi-label settings #
        dask_xgboost_flag = False 
    bst_models = []
    start_time2 = time.time()
    #########   This is for DASK Dataframes XGBoost training ####################
    try:
        xgb.set_config(verbosity=0)
    except:
        ## Some cases, this errors, so pass ###
        pass
    #################################################################################################
    ########   Now if dask_xgboost_flag is True, convert pandas dfs back to Dask Dataframes     #####
    #################################################################################################
    if dask_xgboost_flag:
        ### we reload the dataframes into dask since columns may have been dropped ##
        print('    using DASK XGBoost')  
        train = load_dask_data(dataname, sep)
        if not test_data is None:
            test = load_dask_data(test_data, sep)
    else:
        ### we reload the dataframes into dask since columns may have been dropped ##
        print('    using regular XGBoost') 
        train = copy.deepcopy(dataname)
        test = copy.deepcopy(test_data)
    ########  Conversion completed for train and test data ##########
    print('Train and Test loaded into Dask dataframes successfully after feature_engg completed')
   #### If Category Encoding took place, these cat variables are no longer needed in Train. So remove them!
    if feature_gen or feature_type:
        print('Since %s category encoding is done, dropping original categorical vars from predictors...' %feature_gen)
        preds = left_subtract(preds, catvars)
    train_p = train[preds]
    if train_p.shape[1] < 10:
        iter_limit = 2
    else:
        iter_limit = int(train_p.shape[1]/5+0.5)
    print('Current number of predictors = %d ' %(train_p.shape[1],))
    if dask_xgboost_flag:
        print('    Dask version = %s' %dask.__version__)
        ### You can remove dask_ml here since you don't do any split of train test here ##########
        #from dask_ml.model_selection import train_test_split
        ### check available memory and allocate at least 1GB of it in the Client in DASK #############################
        n_workers = get_cpu_worker_count()
        ### Avoid reducing the free memory - leave it as big as it wants to be ###
        memory_free = str(max(1, int(psutil.virtual_memory()[0]/(1*1e9))))+'GB'
        print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(get_cpu_worker_count(), memory_free))
        client = Client(n_workers= n_workers, threads_per_worker=1, processes=True, silence_logs=50,
                        memory_limit=memory_free)
        print('Dask client configuration: %s' %client)
        import gc
        client.run(gc.collect) 
        #train_p = client.persist(train_p)
    print('    XGBoost version: %s' %xgb.__version__)
    ### This is to convert the target labels to proper numeric columns ######
    ### check if they are not starting from zero ##################
    if dask_xgboost_flag:
        ### train is already a dask dataframe -> you can leave it as it is
        y_train = train[target]
    else:
        y_train = dataname[target]
    #### Now we process the numeric  values through DASK XGBoost repeatedly ###################
    
    try:
        for i in range(0,train_p.shape[1],iter_limit):
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
                print('Number of booster rounds = %s' %num_rounds)

            if train_p.shape[1]-i < 2:
                ### If there is just one variable left, then just skip it #####
                continue
            else:
                print('        using %d variables...' %(train_p.shape[1]-i))
            
            #########   This is where we check target type ##########
            if not y_train.dtypes[0] in [np.float16, np.float32, np.float64, np.int8,np.int16,np.int32,np.int64]:
                y_train = y_train.astype(int) 
            if settings.modeltype == 'Regression':
                objective = 'reg:squarederror'
                params = {'objective': objective, 
                                #'tree_method': tree_method,
                                   "silent":True, "verbosity": 0, 'min_child_weight': 0.5}
            else:
                #### This is for Classifiers only ##########                    
                if settings.modeltype == 'Binary_Classification':
                    objective = 'binary:logistic'
                    num_class = 1
                    params = {'objective': objective, 'num_class': num_class,
                                #'tree_method': tree_method,
                                    "silent":True,  "verbosity": 0,  'min_child_weight': 0.5}
                else:
                    objective = 'multi:softmax'
                    num_class  =  dataname[target].nunique()[0]
                    # Use GPU training algorithm if needed
                    params = {'objective': objective, 
                                #'tree_method': tree_method,
                                "silent":True, "verbosity": 0,   'min_child_weight': 0.5, 'num_class': num_class}
            ############################################################################################################
            ######### This is where we find out whether to use single or multi-label for xgboost #######################
            ############################################################################################################
            
            if settings.multi_label:
                if settings.modeltype == 'Regression':
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
                    #### now this training via bst works well for both xgboost 0.0.90 as well as 1.5.1 ##
                    
                    try:
                        if settings.modeltype == 'Multi_Classification':
                            wt_array = get_sample_weight_array(y_train)
                            dtrain = xgb.DMatrix(X_train, label=y_train, weight=wt_array, feature_names=cols_sel)
                        else:
                            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=cols_sel)
                        bst = xgb.train(params, dtrain, num_boost_round=num_rounds)                
                    except Exception as error_msg:
                        print('Regular XGBoost is crashing due to: %s' %error_msg)
                        if settings.modeltype == 'Regression':
                            params = {'tree_method': cpu_tree_method}
                        else:
                            params = {'tree_method': cpu_tree_method,'num_class': num_class}
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
                        if settings.modeltype == 'Regression':
                            params = {'tree_method': cpu_tree_method}
                        else:
                            params = {'tree_method': cpu_tree_method,'num_class': num_class}
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
            if settings.multi_label:
                imp_feats = dict(zip(X_train.columns, bst.estimators_[0].feature_importances_))
            else:
                if not dask_xgboost_flag:
                    imp_feats = bst.get_score(fmap='', importance_type='gain')
                else:
                    imp_feats = bst['booster'].get_score(fmap='', importance_type='gain')
            ### skip the next statement since it is duplicating the work of sort_values ##
            #imp_feats = dict(sorted(imp_feats.items(),reverse=True, key=lambda item: item[1]))
            ### doing this for single-label is a little different from settings.multi_label #########
            
            #imp_feats = model_xgb.get_booster().get_score(importance_type='gain')
            #print('%d iteration: imp_feats = %s' %(i+1,imp_feats))
            important_features += pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
            #######  order this in the same order in which they were collected ######
            important_features = list(OrderedDict.fromkeys(important_features))
            if dask_xgboost_flag:
                print('            Time taken for DASK XGBoost feature selection = %0.0f seconds' %(time.time()-start_time2))
            else:
                print('            Time taken for regular XGBoost feature selection = %0.0f seconds' %(time.time()-start_time2))
        #### plot all the feature importances in a grid ###########
        if verbose >= 2:
            if settings.multi_label:
                draw_feature_importances_multi_label(bst_models, dask_xgboost_flag)
            else:
                draw_feature_importances_single_label(bst_models, dask_xgboost_flag)
        important_features = list(OrderedDict.fromkeys(important_features))
    except Exception as e:
        if dask_xgboost_flag:
            print('Dask XGBoost is crashing due to %s. Returning with currently selected features...' %e)
        else:
            print('Regular XGBoost is crashing due to %s. Returning with currently selected features...' %e)
        important_features = copy.deepcopy(preds)
    ######    E    N     D      O  F      X  G  B  O  O  S  T    S E L E C T I O N ####################
    print('            Total time taken for XGBoost feature selection = %0.0f seconds' %(time.time()-start_time2))
    if len(idcols) > 0:
        print('    Alert: No ID variables %s are included in selected features' %idcols)
    print("#######################################################################################")
    print("#####          F E A T U R E   S E L E C T I O N   C O M P L E T E D            #######")
    print("#######################################################################################")
    if len(important_features) <= 30:
        print('Selected %d important features:\n%s' %(len(important_features), important_features))
    else:
        print('Selected %d important features. Too many to print...' %len(important_features))
    numvars = [x for x in numvars if x in important_features]
    important_cats = [x for x in important_cats if x in important_features]
    print('\n    Time taken for feature selection = %0.0f seconds' %(time.time()-start_time))
    #### Now change the feature names back to original feature names ################
    item_replacer = col_name_mapper.get  # For faster gets.
    if isinstance(test_data, str) or test_data is None:
        ## In one case, column names get changed in train but not in test since it test is not available.
        try:
            important_features2 = [item_replacer(n, n) for n in important_features]
            print('    Reverted column names to original names given in train dataset')
        except:
            important_features2 = copy.deepcopy(important_features)
            print('    Could not revert column names to original. Try replacing them manually.')
        #print(f'Returning list of {len(important_features)} important features and a dataframe.')
        if len(date_cols) > 0:
            date_replacer = date_col_mappers.get  # For faster gets.
            important_features1 = [date_replacer(n, n) for n in important_features2]
            important_features2 = find_remove_duplicates(important_features1)
        if len(np.intersect1d(train_ids.columns.tolist(),dataname.columns.tolist())) > 0:
            return important_features2, dataname[important_features+target]
        else:
            dataname = pd.concat([train_ids, dataname], axis=1)
            return important_features2, dataname[important_features+target]
    else:
        print('Returning 2 dataframes: dataname and test_data with %d important features.' %len(important_features))
        
        if len(date_cols) > 0:
            date_replacer = date_col_mappers.get  # For faster gets.
            important_features1 = [date_replacer(n, n) for n in important_features]
            important_features = find_remove_duplicates(important_features1)
        if feature_gen or feature_type:
            ### if feature engg is performed, id columns are dropped. Hence they must rejoin here.
            dataname = pd.concat([train_ids, dataname], axis=1)
            test_data = pd.concat([test_ids, test_data], axis=1)
        if target in list(test_data): ### see if target exists in this test data
            return dataname[important_features+target], test_data[important_features+target]
        else:
            return dataname[important_features+target], test_data[important_features]
################################################################################
def remove_highly_correlated_vars_fast(df, corr_limit=0.70):
    """
    This is a simple method to remove highly correlated features fast using Pearson's Correlation.
    Use this only for float and integer variables. It will automatically select those only.
    It can be used for very large data sets where featurewiz has trouble with memory
    """
    # Creating correlation matrix
    correlation_dataframe = df.corr().abs().astype(np.float16)
    # Selecting upper triangle of correlation matrix
    upper_tri = correlation_dataframe.where(np.triu(np.ones(correlation_dataframe.shape),
                                  k=1).astype(np.bool))
    # Finding index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_limit)]
    print();
    print('Highly correlated columns to remove: %s' %to_drop)
    return to_drop
#####################################################################################
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
    print('        memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('        decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
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
import copy
def _create_ts_features(df, tscol):
    """
    This takes in input a dataframe and a date variable.
    It then creates time series features using the pandas .dt.weekday kind of syntax.
    It also returns the data frame of added features with each variable as an integer variable.
    """
    df = copy.deepcopy(df)
    dt_adds = []
    try:
        df[tscol+'_hour'] = df[tscol].dt.hour.fillna(0).astype(int)
        df[tscol+'_minute'] = df[tscol].dt.minute.fillna(0).astype(int)
        dt_adds.append(tscol+'_hour')
        dt_adds.append(tscol+'_minute')
    except:
        print('    Error in creating hour-second derived features. Continuing...')
    try:
        df[tscol+'_dayofweek'] = df[tscol].dt.dayofweek.fillna(0).astype(int)
        dt_adds.append(tscol+'_dayofweek')
        if tscol+'_hour' in dt_adds:
            DAYS = dict(zip(range(7),['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']))
            df[tscol+'_dayofweek'] = df[tscol+'_dayofweek'].map(DAYS)
            df.loc[:,tscol+'_dayofweek_hour_cross'] = df[tscol+'_dayofweek'] +" "+ df[tscol+'_hour'].astype(str)
            dt_adds.append(tscol+'_dayofweek_hour_cross')
        df[tscol+'_quarter'] = df[tscol].dt.quarter.fillna(0).astype(int)
        dt_adds.append(tscol+'_quarter')
        df[tscol+'_month'] = df[tscol].dt.month.fillna(0).astype(int)
        MONTHS = dict(zip(range(1,13),['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                    'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
        df[tscol+'_month'] = df[tscol+'_month'].map(MONTHS)
        dt_adds.append(tscol+'_month')
        #### Add some features for months ########################################
        festives = ['Oct','Nov','Dec']
        name_col = tscol+"_is_festive"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in festives else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[[name_col]].fillna(0)
        dt_adds.append(name_col)
        summer = ['Jun','Jul','Aug']
        name_col = tscol+"_is_summer"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in summer else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[name_col].fillna(0)
        dt_adds.append(name_col)
        winter = ['Dec','Jan','Feb']
        name_col = tscol+"_is_winter"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in winter else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[[name_col]].fillna(0)
        dt_adds.append(name_col)
        cold = ['Oct','Nov','Dec','Jan','Feb','Mar']
        name_col = tscol+"_is_cold"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in cold else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[[name_col]].fillna(0)
        dt_adds.append(name_col)
        warm = ['Apr','May','Jun','Jul','Aug','Sep']
        name_col = tscol+"_is_warm"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in warm else 0).values
        ### Remember that fillna only works at dataframe level! ###
        df[[name_col]] = df[[name_col]].fillna(0)
        dt_adds.append(name_col)
        #########################################################################
        if tscol+'_dayofweek' in dt_adds:
            df.loc[:,tscol+'_month_dayofweek_cross'] = df[tscol+'_month'] +" "+ df[tscol+'_dayofweek']
            dt_adds.append(tscol+'_month_dayofweek_cross')
        df[tscol+'_year'] = df[tscol].dt.year.fillna(0).astype(int)
        dt_adds.append(tscol+'_year')
        today = date.today()
        df[tscol+'_age_in_years'] = today.year - df[tscol].dt.year.fillna(0).astype(int)
        dt_adds.append(tscol+'_age_in_years')
        df[tscol+'_dayofyear'] = df[tscol].dt.dayofyear.fillna(0).astype(int)
        dt_adds.append(tscol+'_dayofyear')
        df[tscol+'_dayofmonth'] = df[tscol].dt.day.fillna(0).astype(int)
        dt_adds.append(tscol+'_dayofmonth')
        df[tscol+'_weekofyear'] = df[tscol].dt.weekofyear.fillna(0).astype(int)
        dt_adds.append(tscol+'_weekofyear')
        weekends = (df[tscol+'_dayofweek'] == 'Sat') | (df[tscol+'_dayofweek'] == 'Sun')
        df[tscol+'_typeofday'] = 'weekday'
        df.loc[weekends, tscol+'_typeofday'] = 'weekend'
        dt_adds.append(tscol+'_typeofday')
        if tscol+'_typeofday' in dt_adds:
            df.loc[:,tscol+'_month_typeofday_cross'] = df[tscol+'_month'] +" "+ df[tscol+'_typeofday']
            dt_adds.append(tscol+'_month_typeofday_cross')
    except:
        print('    Error in creating date time derived features. Continuing...')
    print('    created %d columns from time series %s column' %(len(dt_adds),tscol))
    return df, dt_adds
################################################################
from dateutil.relativedelta import relativedelta
from datetime import date
##### This is a little utility that computes age from year ####
def compute_age(year_string):
    today = date.today()
    age = relativedelta(today, year_string)
    return age.years
#################################################################
def FE_create_time_series_features(dft, ts_column, ts_adds_in=[]):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    #######        B E W A R E  : H U G E   N U M B E R   O F  F E A T U R E S  ###########
    This creates between 100 and 110 date time features for each date variable. The number
    of features depends on whether it is just a year variable or a year+month+day and
    whether it has hours and minutes or seconds. So this can create all these features
    using just the date time column that you send in. Optinally, you can send in a list
    of columns that you want returned. It will preserved those same columns and return them.
    ######################################################################################
    Inputs:
    dtf: pandas DataFrame
    ts_column: name of the time series column
    ts_adds_in: list of time series columns you want in the returned dataframe.

    Outputs:
    dtf: pandas dataframe
    ######################################################################################
    It returns the same dataframe with the added variables as output. It will return ts_adds_in
    if you give it a list of columns to return. This is useful for matching test with train.
    """
    dtf = copy.deepcopy(dft)
    reset_index = False
    if not ts_adds_in:
        # ts_column = None assumes that that index is the time series index
        reset_index = False
        if ts_column is None:
            reset_index = True
            ts_column = dtf.index.name
            dtf = dtf.reset_index()

        ### In some extreme cases, date time vars are not processed yet and hence we must fill missing values here!
        null_nums = dtf[ts_column].isnull().sum()
        if  null_nums > 0:
            # missing_flag = True
            new_missing_col = ts_column + '_Missing_Flag'
            dtf[new_missing_col] = 0
            dtf.loc[dtf[ts_column].isnull(),new_missing_col]=1
            ### Remember that fillna only works at dataframe level! ###
            dtf[[ts_column]] = dtf[[ts_column]].fillna(method='ffill')
            print('        adding %s column due to missing values in data' %new_missing_col)
            if dtf[dtf[ts_column].isnull()].shape[0] > 0:
                ### Remember that fillna only works at dataframe level! ###
                dtf[[ts_column]] = dtf[[ts_column]].fillna(method='bfill')

        if dtf[ts_column].dtype == float:
            dtf[ts_column] = dtf[ts_column].astype(int)

        ### if we have already found that it was a date time var, then leave it as it is. Thats good enough!
        items = dtf[ts_column].apply(str).apply(len).values
        #### In some extreme cases,
        if all(items[0] == item for item in items):
            if items[0] == 4:
                ### If it is just a year variable alone, you should leave it as just a year!
                dtf[ts_column] = pd.to_datetime(dtf[ts_column],format='%Y')
                ts_adds = []
            else:
                ### if it is not a year alone, then convert it into a date time variable
                dtf[ts_column] = pd.to_datetime(dtf[ts_column], infer_datetime_format=True)
                ### this is where you create the time series features #####
                dtf, ts_adds = _create_ts_features(df=dtf, tscol=ts_column)
        else:
            dtf[ts_column] = pd.to_datetime(dtf[ts_column], infer_datetime_format=True)
            ### this is where you create the time series features #####
            dtf, ts_adds = _create_ts_features(df=dtf, tscol=ts_column)
    else:
        dtf[ts_column] = pd.to_datetime(dtf[ts_column], infer_datetime_format=True)
        ### this is where you create the time series features #####
        dtf, ts_adds = _create_ts_features(df=dtf, tscol=ts_column)
    ####### This is where we make sure train and test have the same number of columns ####
    try:
        if not ts_adds_in:
            ts_adds_copy = copy.deepcopy(ts_adds)
            rem_cols = left_subtract(dtf.columns.tolist(), ts_adds_copy)
            ts_adds_num = dtf[ts_adds].select_dtypes(include='number').columns.tolist()
            ### drop those columns where all rows are same i.e. zero variance  ####
            for col in ts_adds_num:
                if dtf[col].std() == 0:
                    dtf = dtf.drop(col, axis=1)
                    ts_adds.remove(col)
            removed_ts_cols = left_subtract(ts_adds_copy, ts_adds)
            print('        dropped %d time series added columns due to zero variance' %len(removed_ts_cols))
            rem_ts_cols = ts_adds
            dtf = dtf[rem_cols+rem_ts_cols]
        else:
            #rem_cols = left_subtract(dtf.columns.tolist(), ts_adds_in)
            rem_cols = left_subtract(ts_adds, ts_adds_in)
            dtf.drop(rem_cols, axis=1, inplace=True)
            #dtf = dtf[rem_cols+ts_adds_in]
            rem_ts_cols = ts_adds_in
        # If you had reset the index earlier, set it back before returning
        # to  make it consistent with the dataframe that was sent as input
        if reset_index:
            dtf = dtf.set_index(ts_column)
        elif ts_column in dtf.columns:
            print('        dropping %s column after time series done' %ts_column)
            dtf = dtf.drop(ts_column, axis=1)
        else:
            pass
        print('    After dropping zero variance cols, shape of data: %s' %(dtf.shape,))
    except Exception as e:
        print(e)
        print('Error in Processing %s column for date time features. Continuing...' %ts_column)
    return dtf, rem_ts_cols
######################################################################################
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

from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.over_sampling import ADASYN, SMOTENC

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
from sklearn.utils.class_weight import compute_class_weight
def get_class_distribution(y_input):
    y_input = copy.deepcopy(y_input)
    classes = np.unique(y_input)
    xp = Counter(y_input)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_input), y=y_input)
    if len(class_weights[(class_weights> 10)]) > 0:
        class_weights = (class_weights/10).astype(int)
    else:
        class_weights = (class_weights).astype(int)
    print('class_weights = %s' %class_weights)
    class_weights[(class_weights<1)]=1
    class_rows = class_weights*[xp[x] for x in classes]
    class_weighted_rows = dict(zip(classes,class_rows))
    print('class_weighted_rows = %s' %class_weighted_rows)
    return class_weighted_rows

def oversample_SMOTE(X,y):
    #input DataFrame
    #X →Independent Variable in DataFrame\
    #y →dependent Variable in Pandas DataFrame format
    # Get the class distriubtion for perfoming relative sampling in the next line
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
def FE_transform_numeric_columns(df, bin_dict, verbose=0):
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
from tqdm import tqdm
def FE_add_lagged_targets_by_date_category(train2, target_col, date_col, category_col, test2):
    """
    ######    F E A T U R E    E N G G    F O R    T I M E    S E R I E S     D A T A  ############
    This handy function adds the mean of lagged values for target column based on date and a category.
    For each date in date column, it collects all target values prior to that date and calculates mean.
    The collected target values are in same category as current category in each row.
    This function enables a model to learn average of prior target (values) grouped by a category.
    It also calculates same values for test using values in train data 9(by using it as look up tables)
    1. In Train data, we select rows prior to the current row's date and match category. We calculate mean.
    2. In Test, we select rows matching the category and any row prior to the current row's date. Again mean.
    3. In both cases, if there is no match, we just select all rows prior to the current row's date and find their mean.
    ############    I N P U T S    A N D      O U T P U T S     #####################################
    Inputs:
        train2: a training dataframe
        target_col: name of target variable you want the mean of the lagged values before a certain date
        date_col: name of the
        category_col: name of category column prior target values are grouped by (and mean calculated)

    Outputs:
        new_col: new column in train and test
        - representing average of prior target values by category before each date in a row
    #################################################################################################
    """
    train2 = copy.deepcopy(train2)
    test2 = copy.deepcopy(test2)
    new_col = target_col+'_array'
    train2[new_col] = 0
    i = 0
    train2 = train2.reset_index(drop=False)  # index column will be called 'index'
    train2 = train2.set_index(pd.to_datetime(train2.pop(date_col)))
    train2 = train2.sort_index()
    for index, each_train in tqdm(train2.iterrows(), total=train2.shape[0]):
        before_grp = train2.loc[train2.index < index]
        prim = each_train[category_col]
        mask = (before_grp[category_col] == prim)
        if len(before_grp.loc[mask]) > 0:
            price_val = before_grp.loc[mask,target_col].values.tolist()
        else:
            price_val = before_grp[target_col].values.tolist()
        #price_val = f"{price_val}"
        price_val = np.mean(price_val)
        #print(' i = ', i+1, price_val)
        train2.iloc[i,-1] = price_val
        i += 1
    if not isinstance(test2, str):
        test2[new_col] = 0
        i = 0
        test2 = test2.reset_index(drop=False)  # index column will be called 'index'
        test2 = test2.set_index(pd.to_datetime(test2.pop(date_col)))
        test2 = test2.sort_index()
        for index1, each_test in tqdm(test2.iterrows(), total=test2.shape[0]):
            before_grp = train2.loc[train2.index < index1]
            prim = each_test[category_col]
            mask = (before_grp[category_col] == prim)
            if len(before_grp.loc[mask]) > 0:
                price_val = before_grp.loc[mask,target_col].values.tolist()
            else:
                price_val = before_grp[target_col].values.tolist()
            price_val = np.mean(price_val)
            test2.iloc[i,-1] = price_val
            i += 1
    return train2.set_index('index'), test2.set_index('index')
#############################################################################################
import numpy as np
import pandas as pd
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
    cols = df.columns.tolist()
    number_duplicates = df.columns.duplicated().astype(int).sum()
    if  number_duplicates > 0:
        print('Detected %d duplicate columns in dataset. Removing duplicates...' %number_duplicates)
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
def select_rows_from_dataframe(train_dataframe, targets, nrows_limit, DS_LEN=''):
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
    print('    loading a random sample of %d rows into pandas for EDA' %nrows_limit)
    ####### If it is a classification problem, you need to stratify and select sample ###
    if modeltype != 'Regression':
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
        train_small = train_dataframe.sample(n=nrows_limit, replace=True, random_state=99)
    return train_small
################################################################################################
class FeatureWiz(BaseEstimator, TransformerMixin):
    def __init__(self, corr_limit=0.70, verbose=2, sep=',', 
        header=0, feature_engg='', category_encoders='',
        dask_xgboost_flag=False, nrows=None):
        self.features = None
        self.corr_limit= corr_limit
        self.verbose=verbose
        self.sep=sep
        self.header=header
        self.test_data = "" ## leave testdata permanently as empty for now ##
        self.feature_engg=feature_engg
        self.category_encoders=category_encoders
        self.dask_xgboost_flag=dask_xgboost_flag
        self.nrows=nrows
        #self.X_sel = None

    def fit(self, X, y):
        start_time = time.time()
        if isinstance(X, np.ndarray):
            print('X input must be a dataframe since we use column names to build data pipelines. Returning')
            return {}, {}
        X_index = X.index
        y_index = y.index
        if (X_index == y_index).all():
            df = pd.concat([X, y], axis=1)
        else:
            df = pd.concat([X.reset_index(drop=True), y], axis=1)
            df.index = X_index
        ### Now you can process the X and y datasets ####
        if isinstance(y, pd.Series):
            target = y.name
        elif isinstance(y, pd.DataFrame):
            target = y.columns.tolist()
        elif isinstance(X, np.ndarray):
            print('y must be a pd.Series or pd.DataFrame since we use column names to build data pipeline. Returning')
            return {}, {}
        #### Send target variable as it is so that y_train is analyzed properly ###
        # Select features using featurewiz
        features, X_sel = featurewiz(df, target, self.corr_limit, self.verbose, self.sep, 
                self.header, self.test_data, self.feature_engg, self.category_encoders,
                self.dask_xgboost_flag, self.nrows)
        # Convert the remaining column names back to integers and drop the
        difftime = max(1, int(time.time()-start_time))
        print('    Time taken to create entire pipeline = %s second(s)' %difftime)
        # column of labels
        self.features = features
        return self

    def transform(self, X):
        return X[self.features]
###################################################################################################
import random
import collections
def make_column_names_unique(data_input):
    special_char_flag = False
    cols = data_input.columns.tolist()
    copy_cols = copy.deepcopy(cols)
    ser = pd.Series(cols)
    ### This function removes all special chars from a list ###
    remove_special_chars =  lambda x:re.sub('[^A-Za-z0-9_]+', '', x)
    newls = ser.map(remove_special_chars).values.tolist()
    ### there may be duplicates in this list - we need to make them unique by randomly adding strings to name ##
    seen = [item for item, count in collections.Counter(newls).items() if count > 1]
    new_cols = [x+str(random.randint(1,1000)) if x in seen else x for x in newls]
    copy_new_cols = copy.deepcopy(new_cols)
    copy_cols.sort()
    copy_new_cols.sort()
    if copy_cols != copy_new_cols:
        print('    Some column names had special characters which were removed...')
        special_char_flag = True
    return new_cols, special_char_flag
##########################################################################################
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
