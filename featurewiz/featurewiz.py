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
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
###########   This is from category_encoders Library ################################################
from category_encoders import HashingEncoder, SumEncoder, PolynomialEncoder, BackwardDifferenceEncoder
from category_encoders import OneHotEncoder, HelmertEncoder, OrdinalEncoder, CountEncoder, BaseNEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder, JamesSteinEncoder
from category_encoders.glmm import GLMMEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.wrapper import PolynomialWrapper
from .encoders import FrequencyEncoder
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
import random
import xlrd
import statsmodels
from io import BytesIO
import base64
from functools import reduce
import copy
import dask
import dask.dataframe as dd
import dask_xgboost
import xgboost
from dask.distributed import Client, progress
import psutil
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
        print('        No variables removed since no ID or low-information variables found in data set')
    else:
        print('        %d variable(s) will be ignored since they are ID or low-information variables'
                                %len(IDcols+cols_delete+discrete_string_vars))
        if verbose >= 1:
            print('    List of variables ignored: %s' %(IDcols+cols_delete+discrete_string_vars))
    #############  Check if there are too many columns to visualize  ################
    ppt = pprint.PrettyPrinter(indent=4)
    if verbose==1 and len(cols_list) <= max_cols_analyzed:
        marthas_columns(dft,verbose)
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
            print('Data Set columns info:')
            for col in data.columns:
                print('* %s: %d nulls, %d unique vals, most common: %s' % (
                        col,
                        data[col].isnull().sum(),
                        data[col].nunique(),
                        data[col].value_counts().head(2).to_dict()
                    ))
            print('--------------------------------------------------------------------')
################################################################################
######### NEW And FAST WAY to CLASSIFY COLUMNS IN A DATA SET #######
################################################################################
def classify_columns(df_preds, verbose=0):
    """
    This actually does Exploratory data analysis - it means this function performs EDA
    ######################################################################################
    Takes a dataframe containing only predictors to be classified into various types.
    DO NOT SEND IN A TARGET COLUMN since it will try to include that into various columns.
    Returns a data frame containing columns and the class it belongs to such as numeric,
    categorical, date or id column, boolean, nlp, discrete_string and cols to delete...
    ####### Returns a dictionary with 10 kinds of vars like the following: # continuous_vars,int_vars
    # cat_vars,factor_vars, bool_vars,discrete_string_vars,nlp_vars,date_vars,id_vars,cols_delete
    """
    train = copy.deepcopy(df_preds)
    #### If there are 30 chars are more in a discrete_string_var, it is then considered an NLP variable
    max_nlp_char_size = 30
    max_cols_to_print = 30
    print('############## C L A S S I F Y I N G  V A R I A B L E S  ####################')
    print('Classifying variables in data set...')
    #### Cat_Limit defines the max number of categories a column can have to be called a categorical colum
    cat_limit = 35
    float_limit = 15 #### Make this limit low so that float variables below this limit become cat vars ###
    def add(a,b):
        return a+b
    sum_all_cols = dict()
    orig_cols_total = train.shape[1]
    #Types of columns
    cols_delete = [col for col in list(train) if (len(train[col].value_counts()) == 1
                                       ) | (train[col].isnull().sum()/len(train) >= 0.90)]
    train = train[left_subtract(list(train),cols_delete)]
    var_df = pd.Series(dict(train.dtypes)).reset_index(drop=False).rename(
                        columns={0:'type_of_column'})
    sum_all_cols['cols_delete'] = cols_delete
    var_df['bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['bool','object']
                        and len(train[x['index']].value_counts()) == 2 else 0, axis=1)
    string_bool_vars = list(var_df[(var_df['bool'] ==1)]['index'])
    sum_all_cols['string_bool_vars'] = string_bool_vars
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                            np.uint16, np.uint32, np.uint64,
                            'int8','int16','int32','int64',
                            'float16','float32','float64'] and len(
                        train[x['index']].value_counts()) == 2 else 0, axis=1)
    num_bool_vars = list(var_df[(var_df['num_bool'] ==1)]['index'])
    sum_all_cols['num_bool_vars'] = num_bool_vars
    ######   This is where we take all Object vars and split them into diff kinds ###
    discrete_or_nlp = var_df.apply(lambda x: 1 if x['type_of_column'] in ['object']  and x[
        'index'] not in string_bool_vars+cols_delete else 0,axis=1)
    ######### This is where we figure out whether a string var is nlp or discrete_string var ###
    var_df['nlp_strings'] = 0
    var_df['discrete_strings'] = 0
    var_df['cat'] = 0
    var_df['id_col'] = 0
    discrete_or_nlp_vars = var_df.loc[discrete_or_nlp==1]['index'].values.tolist()
    if len(var_df.loc[discrete_or_nlp==1]) != 0:
        for col in discrete_or_nlp_vars:
            #### first fill empty or missing vals since it will blowup ###
            train[col] = train[col].fillna('  ')
            if train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= max_nlp_char_size and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) == len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                var_df.loc[var_df['index']==col,'cat'] = 1
    nlp_vars = list(var_df[(var_df['nlp_strings'] ==1)]['index'])
    sum_all_cols['nlp_vars'] = nlp_vars
    discrete_string_vars = list(var_df[(var_df['discrete_strings'] ==1) ]['index'])
    sum_all_cols['discrete_string_vars'] = discrete_string_vars
    ###### This happens only if a string column happens to be an ID column #######
    #### DO NOT Add this to ID_VARS yet. It will be done later.. Dont change it easily...
    #### Category DTYPE vars are very special = they can be left as is and not disturbed in Python. ###
    var_df['dcat'] = var_df.apply(lambda x: 1 if str(x['type_of_column'])=='category' else 0,
                            axis=1)
    factor_vars = list(var_df[(var_df['dcat'] ==1)]['index'])
    sum_all_cols['factor_vars'] = factor_vars
    ########################################################################
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                         np.uint16, np.uint32, np.uint64,
                         'int8','int16',
                        'int32','int64']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ######### This is where we figure out whether a numeric col is date or id variable ###
    var_df['int'] = 0
    var_df['date_time'] = 0
    ### if a particular column is date-time type, now set it as a date time variable ##
    var_df['date_time'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['<M8[ns]','datetime64[ns]']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ### this is where we save them as date time variables ###
    if len(var_df.loc[date_or_id==1]) != 0:
        for col in var_df.loc[date_or_id==1]['index'].values.tolist():
            if len(train[col].value_counts()) == len(train):
                if train[col].min() < 1900 or train[col].max() > 2050:
                    var_df.loc[var_df['index']==col,'id_col'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                if train[col].min() < 1900 or train[col].max() > 2050:
                    if col not in num_bool_vars:
                        var_df.loc[var_df['index']==col,'int'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        if col not in num_bool_vars:
                            var_df.loc[var_df['index']==col,'int'] = 1
    else:
        pass
    int_vars = list(var_df[(var_df['int'] ==1)]['index'])
    date_vars = list(var_df[(var_df['date_time'] == 1)]['index'])
    id_vars = list(var_df[(var_df['id_col'] == 1)]['index'])
    sum_all_cols['int_vars'] = int_vars
    copy_date_vars = copy.deepcopy(date_vars)
    for date_var in copy_date_vars:
        #### This test is to make sure sure date vars are actually date vars
        try:
            pd.to_datetime(train[date_var],infer_datetime_format=True)
        except:
            ##### if not a date var, then just add it to delete it from processing
            cols_delete.append(date_var)
            date_vars.remove(date_var)
    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    sum_all_cols['cols_delete'] = cols_delete
    ## This is an EXTREMELY complicated logic for cat vars. Don't change it unless you test it many times!
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= float_limit and len(train[col].value_counts()) <= len(train):
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if col not in num_bool_vars:
                    var_df.loc[var_df['index']==col,'numeric'] = 1
    cat_vars = list(var_df[(var_df['cat'] ==1)]['index'])
    continuous_vars = list(var_df[(var_df['numeric'] ==1)]['index'])
    ########  V E R Y    I M P O R T A N T   ###################################################
    ##### There are a couple of extra tests you need to do to remove abberations in cat_vars ###
    cat_vars_copy = copy.deepcopy(cat_vars)
    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['continuous_vars'] = continuous_vars
    sum_all_cols['id_vars'] = id_vars
    ###### This is where you consoldate the numbers ###########
    var_dict_sum = dict(zip(var_df.values[:,0], var_df.values[:,2:].sum(1)))
    for col, sumval in var_dict_sum.items():
        if sumval == 0:
            print('%s of type=%s is not classified' %(col,train[col].dtype))
        elif sumval > 1:
            print('%s of type=%s is classified into more then one type' %(col,train[col].dtype))
        else:
            pass
    ###############  This is where you print all the types of variables ##############
    ####### Returns 8 vars in the following order: continuous_vars,int_vars,cat_vars,
    ###  string_bool_vars,discrete_string_vars,nlp_vars,date_or_id_vars,cols_delete
    if verbose == 1:
        print("    Number of Numeric Columns = ", len(continuous_vars))
        print("    Number of Integer-Categorical Columns = ", len(int_vars))
        print("    Number of String-Categorical Columns = ", len(cat_vars))
        print("    Number of Factor-Categorical Columns = ", len(factor_vars))
        print("    Number of String-Boolean Columns = ", len(string_bool_vars))
        print("    Number of Numeric-Boolean Columns = ", len(num_bool_vars))
        print("    Number of Discrete String Columns = ", len(discrete_string_vars))
        print("    Number of NLP String Columns = ", len(nlp_vars))
        print("    Number of Date Time Columns = ", len(date_vars))
        print("    Number of ID Columns = ", len(id_vars))
        print("    Number of Columns to Delete = ", len(cols_delete))
    if verbose == 2:
        marthas_columns(df_preds,verbose=1)
        print("    Numeric Columns: %s" %continuous_vars[:max_cols_to_print])
        print("    Integer-Categorical Columns: %s" %int_vars[:max_cols_to_print])
        print("    String-Categorical Columns: %s" %cat_vars[:max_cols_to_print])
        print("    Factor-Categorical Columns: %s" %factor_vars[:max_cols_to_print])
        print("    String-Boolean Columns: %s" %string_bool_vars[:max_cols_to_print])
        print("    Numeric-Boolean Columns: %s" %num_bool_vars[:max_cols_to_print])
        print("    Discrete String Columns: %s" %discrete_string_vars[:max_cols_to_print])
        print("    NLP text Columns: %s" %nlp_vars[:max_cols_to_print])
        print("    Date Time Columns: %s" %date_vars[:max_cols_to_print])
        print("    ID Columns: %s" %id_vars[:max_cols_to_print])
        print("    Columns that will not be considered in modeling: %s" %cols_delete[:max_cols_to_print])
    ##### now collect all the column types and column names into a single dictionary to return!
    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()])
    if len_sum_all_cols == orig_cols_total:
        print('    %d Predictors classified...' %orig_cols_total)
        #print('        This does not include the Target column(s)')
    else:
        print('No of columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]
        if len(left_subtract(list(train),flat_list)) == 0:
            print(' Missing columns = None')
        else:
            print(' Missing columns = %s' %left_subtract(list(train),flat_list))
    return sum_all_cols
#################################################################################
def lenopenreadlines(filename):
    with open(filename) as f:
        return len(f.readlines())
#########################################################################################
from collections import Counter
import time
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
import random
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
                print('    Shape of your Data Set loaded: %s' %(dfte.shape,))
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
    This function loads a given filename into a pandas dataframe and sets the
    ts_column as a Time Series index. Note that filename should contain the full
    path to the file.
    """
    if isinstance(filename, str):
        dft = dd.read_csv(filename, blocksize='default')
        print('    Too big to fit into pandas. Hence loaded file %s into a Dask dataframe ...' % filename)
    else:
        ### If filename is not a string, it must be a dataframe and can be loaded
        dft =   dd.from_pandas(filename, npartitions=1)
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
#### Regression or Classification type problem
def analyze_problem_type(train, target, verbose=0) :
    target = copy.deepcopy(target)
    cat_limit = 30 ### this determines the number of categories to name integers as classification ##
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    if isinstance(target, str):
        target = [target]
    if len(target) == 1:
        targ = target[0]
        model_label = 'Single_Label'
    else:
        targ = target[0]
        model_label = 'Multi_Label'
    ####  This is where you detect what kind of problem it is #################
    if  train[targ].dtype in ['int64', 'int32','int16']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= cat_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    elif  train[targ].dtype in ['float16','float32','float64']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= float_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    else:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        else:
            model_class = 'Multi_Classification'
    ########### print this for the start of next step ###########
    if verbose <= 1:
        print('''#### %s %s Feature Selection Started ####''' %(
                                model_label,model_class))
    return model_class
#####################################################################################
from collections import defaultdict
from collections import OrderedDict
import time
def return_dictionary_list(lst_of_tuples):
    """ Returns a dictionary of lists if you send in a list of Tuples"""
    orDict = defaultdict(list)
    # iterating over list of tuples
    for key, val in lst_of_tuples:
        orDict[key].append(val)
    return orDict
##################################################################################
def EDA_find_columns_with_infinity(df):
    """
    This function finds all columns in a dataframe that have inifinite values (np.inf or -np.inf)
    It returns a list of column names. If the list is empty, it means no columns were found.
    """
    add_cols = []
    sum_cols = 0
    for col in df.columns:
        inf_sum1 = 0 
        inf_sum2 = 0
        inf_sum1 = len(df[df[col]==np.inf])
        inf_sum2 = len(df[df[col]==-np.inf])
        if (inf_sum1 > 0) or (inf_sum2 > 0):
            add_cols.append(col)
            sum_cols += inf_sum1
            sum_cols += inf_sum2
    return add_cols
####################################################################################
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
    cols_with_infinity = EDA_find_columns_with_infinity(df)
    if cols_with_infinity:
        print('    There are still %d columns with infinite values. Returning...' %len(cols_with_infinity))
    else:
        print('    There are no more columns with infinite values')
    return df
##################################################################################
def FE_remove_variables_using_SULOV_method(df, numvars, modeltype, target,
                                corr_limit = 0.70,verbose=0, dask_xgboost_flag=True):
    """
    FE stands for Feature Engineering - it means this function performs feature engineering
    ###########################################################################################
    #####              SULOV stands for Searching Uncorrelated List Of Variables  #############
    This highly efficient method removes variables that are highly correlated using a series of
    pair-wise correlation knockout rounds. It is extremely fast and hence can work on thousands
    of variables in less than a minute, even on a laptop. You need to send in a list of numeric
    variables and that's all! The method defines high Correlation as anything over 0.70 (absolute)
    but this can be changed. If two variables have absolute correlation higher than this, they
    will be marked, and using a process of elimination, one of them will get knocked out:
    To decide order of variables to keep, we use mutuail information score to select. MIS returns
    a ranked list of these correlated variables: when we select one, we knock out others
    that it is correlated to. Then we select next var. This way we knock out correlated variables.
    Finally we are left with uncorrelated variables that are also highly important in mutual score.
    ########  YOU MUST INCLUDE THE ABOVE MESSAGE IF YOU COPY THIS CODE IN YOUR LIBRARY ##########
    """
    df = copy.deepcopy(df)
    ### for some reason, doing a mass fillna of vars doesn't work! Hence doing it individually!
    null_vars = np.array(numvars)[df[numvars].isnull().sum()>0]
    for each_num in null_vars:
        df[each_num] = df[each_num].fillna(0)
    target = copy.deepcopy(target)
    print('Searching for highly correlated variables from %d variables using SULOV method' %len(numvars))
    print('#####  SULOV : Searching for Uncorrelated List Of Variables (takes time...) ############')
    correlation_dataframe = df[numvars].corr().abs().astype(np.float16)
    ######### This is how you create a dictionary of which var is highly correlated to a list of vars ####
    corr_values = correlation_dataframe.values
    col_index = correlation_dataframe.columns.tolist()
    index_triupper = list(zip(np.triu_indices_from(corr_values,k=1)[0],np.triu_indices_from(
                                corr_values,k=1)[1]))
    high_corr_index_list = [x for x in np.argwhere(abs(corr_values[np.triu_indices(len(corr_values), k = 1)])>=corr_limit)]
    low_corr_index_list =  [x for x in np.argwhere(abs(corr_values[np.triu_indices(len(corr_values), k = 1)])<corr_limit)]
    tuple_list = [y for y in [index_triupper[x[0]] for x in high_corr_index_list]]
    correlated_pair = [(col_index[tuple[0]],col_index[tuple[1]]) for tuple in tuple_list]
    corr_pair_dict = dict(return_dictionary_list(correlated_pair))
    keys_in_dict = list(corr_pair_dict.keys())
    reverse_correlated_pair = [(y,x) for (x,y) in correlated_pair]
    reverse_corr_pair_dict = dict(return_dictionary_list(reverse_correlated_pair))
    for key, val in reverse_corr_pair_dict.items():
        if key in keys_in_dict:
            if len(key) > 1:
                corr_pair_dict[key] += val
        else:
            corr_pair_dict[key] = val
    #### corr_pair_dict is used later to make the network diagram to see which vars are correlated to which
    # Selecting upper triangle of correlation matrix ## this is a fast way to find highly correlated vars
    upper_tri = correlation_dataframe.where(np.triu(np.ones(correlation_dataframe.shape),
                                  k=1).astype(np.bool))
    empty_df = upper_tri[abs(upper_tri)>corr_limit]
    ### if none of the variables are highly correlated, you can skip this whole drawing
    if empty_df.isnull().all().all():
        print('    No highly correlated variables in data set to remove. All selected...')
        return numvars
    #### It's important to find the highly correlated features first #############
    lower_tri = correlation_dataframe.where(np.tril(np.ones(correlation_dataframe.shape),
                                  k=-1).astype(np.bool))
    lower_df = lower_tri[abs(lower_tri)>corr_limit]
    corr_list =  empty_df.columns[[not(empty_df[x].isnull().all()) for x in list(empty_df)]].tolist(
                    )+lower_df.columns[[not(lower_df[x].isnull().all()) for x in list(lower_df)]].tolist()
    corr_list = find_remove_duplicates(corr_list)
    ###### This is for ordering the variables in the highest to lowest importance to target ###
    if len(corr_list) == 0:
        final_list = list(correlation_dataframe)
        print('Selecting all (%d) variables since none of them are highly correlated...' %len(numvars))
        return numvars
    else:
        if isinstance(target, list):
            target = target[0]
        max_feats = len(corr_list)
        if modeltype == 'Regression':
            sel_function = mutual_info_regression
            fs = SelectKBest(score_func=sel_function, k=max_feats)
        else:
            sel_function = mutual_info_classif
            fs = SelectKBest(score_func=sel_function, k=max_feats)
        ##### you must ensure there are no infinite nor null values in corr_list df ##
        df_fit = df[corr_list]
        ### Now check if there are any NaN values in the dataset #####
        if df_fit.isnull().sum().sum() > 0:
            df_fit = df_fit.dropna()
        else:
            print('There are no null values in dataset.')
        ##### Ready to perform fit and find mutual information score ####       
        try:
            fs.fit(df_fit.astype(np.float32), df[target])
        except:
            print('Select K Best erroring. Returning with all variables...')
            return corr_list
        ##########################################################################
        try:
            mutual_info = dict(zip(corr_list,fs.scores_))
            #### The first variable in list has the highest correlation to the target variable ###
            sorted_by_mutual_info =[key for (key,val) in sorted(mutual_info.items(), key=lambda kv: kv[1],reverse=True)]
            #####   Now we select the final list of correlated variables ###########
            selected_corr_list = []
            #### You have to make multiple copies of this sorted list since it is iterated many times ####
            orig_sorted = copy.deepcopy(sorted_by_mutual_info)
            copy_sorted = copy.deepcopy(sorted_by_mutual_info)
            copy_pair = copy.deepcopy(corr_pair_dict)
            #### select each variable by the highest mutual info and see what vars are correlated to it
            for each_corr_name in copy_sorted:
                ### add the selected var to the selected_corr_list
                selected_corr_list.append(each_corr_name)
                for each_remove in copy_pair[each_corr_name]:
                    #### Now remove each variable that is highly correlated to the selected variable
                    if each_remove in copy_sorted:
                        copy_sorted.remove(each_remove)
            ##### Now we combine the uncorrelated list to the selected correlated list above
            rem_col_list = left_subtract(list(correlation_dataframe),corr_list)
            final_list = rem_col_list + selected_corr_list
            removed_cols = left_subtract(numvars, final_list)
        except:
            print('    SULOV Method crashing due to memory error, trying alternative simpler method...')
            #### Dropping highly correlated Features fast using simple linear correlation ###
            removed_cols = remove_highly_correlated_vars_fast(train[numvars],corr_limit)
            final_list = left_subtract(numvars, removed_cols)
        if len(removed_cols) > 0:
            print('    Removing (%d) highly correlated variables:' %(len(removed_cols)))
            if len(removed_cols) <= 30:
                print('    %s' %removed_cols)
            if len(final_list) <= 30:
                print('    Following (%d) vars selected: %s' %(len(final_list),final_list))
        ##############    D R A W   C O R R E L A T I O N   N E T W O R K ##################
        selected = copy.deepcopy(final_list)
        try:
            import networkx as nx
        except:
            print('    Python networkx library not installed. Install it for feature selection visualization.')
            return
        #### Now start building the graph ###################
        gf = nx.Graph()
        ### the mutual info score gives the size of the bubble ###
        multiplier = 2100
        for each in orig_sorted:
            gf.add_node(each, size=int(max(1,mutual_info[each]*multiplier)))
        ######### This is where you calculate the size of each node to draw
        sizes = [mutual_info[x]*multiplier for x in list(gf.nodes())]
        ####  The sizes of the bubbles for each node is determined by its mutual information score value
        corr = df_fit.corr()
        high_corr = corr[abs(corr)>corr_limit]
        ## high_corr is the dataframe of a few variables that are highly correlated to each other
        combos = combinations(corr_list,2)
        ### this gives the strength of correlation between 2 nodes ##
        multiplier = 20
        for (var1, var2) in combos:
            if np.isnan(high_corr.loc[var1,var2]):
                pass
            else:
                gf.add_edge(var1, var2,weight=multiplier*high_corr.loc[var1,var2])
        ######## Now start building the networkx graph ##########################
        widths = nx.get_edge_attributes(gf, 'weight')
        nodelist = gf.nodes()
        cols = 5
        height_size = 5
        width_size = 15
        rows = int(len(corr_list)/cols)
        if rows < 1:
            rows = 1
        plt.figure(figsize=(width_size,min(20,height_size*rows)))
        pos = nx.shell_layout(gf)
        nx.draw_networkx_nodes(gf,pos,
                               nodelist=nodelist,
                               node_size=sizes,
                               node_color='blue',
                               alpha=0.5)
        nx.draw_networkx_edges(gf,pos,
                               edgelist = widths.keys(),
                               width=list(widths.values()),
                               edge_color='lightblue',
                               alpha=0.6)
        pos_higher = {}
        x_off = 0.04  # offset on the x axis
        y_off = 0.04  # offset on the y axis
        for k, v in pos.items():
            pos_higher[k] = (v[0]+x_off, v[1]+y_off)
        if len(selected) == 0:
            nx.draw_networkx_labels(gf, pos=pos_higher,
                                labels=dict(zip(nodelist,nodelist)),
                                font_color='black')
        else:
            nx.draw_networkx_labels(gf, pos=pos_higher,
                                labels = dict(zip(nodelist,[x+' (selected)' if x in selected else x+' (removed)' for x in nodelist])),
                                font_color='black')
        plt.box(True)
        plt.title("""In SULOV, we repeatedly remove features with lower mutual info scores among highly correlated pairs (see figure),
                    SULOV selects the feature with higher mutual info score related to target when choosing between a pair. """, fontsize=10)
        plt.suptitle('How SULOV Method Works by Removing Highly Correlated Features', fontsize=20,y=1.03)
        red_patch = mpatches.Patch(color='blue', label='Bigger circle denotes higher mutual info score with target')
        blue_patch = mpatches.Patch(color='lightblue', label='Thicker line denotes higher correlation between two variables')
        plt.legend(handles=[red_patch, blue_patch],loc='best')
        plt.show();
        #####    N E T W O R K     D I A G R A M    C O M P L E T E   #################
        return final_list
###############################################################################################
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
def FE_convert_all_object_columns_to_numeric(train, test=""):
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
    if len(nums) == 0:
        pass
    else:

        if train[nums].isnull().sum().sum() > 0:
            null_cols = np.array(nums)[train[nums].isnull().sum()>0].tolist()
            for each_col in null_cols:
                new_missing_col = each_col + '_Missing_Flag'
                train[new_missing_col] = 0
                train.loc[train[each_col].isnull(),new_missing_col]=1
                train[each_col] = train[each_col].fillna(-9999)
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
            #print('    Converting %s to numeric' %everycol)
            MLB = My_LabelEncoder()
            try:
                train[everycol] = MLB.fit_transform(train[everycol])
                if not isinstance(test, str):
                    if test is None:
                        pass
                    else:
                        test[everycol] = MLB.transform(test[everycol])
            except:
                print('Error converting %s column from string to numeric. Continuing...' %everycol)
                continue
    return train, test
###################################################################################
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from .databunch import DataBunch
from .encoders import FrequencyEncoder

from sklearn.model_selection import train_test_split
def featurewiz(dataname, target, corr_limit=0.7, verbose=0, sep=",", header=0,
            test_data='', feature_engg='', category_encoders='', dask_xgboost_flag=True,
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
        dataname: dataname is the name of the training data you want to transform or select.
            dataname could be a datapath+filename or a dataframe. featurewiz will detect whether
            your input is a filename or a dataframe and load it automatically.
        target: name of the target variable in the data set.
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
        feature_engg: You can let featurewiz select its best encoders for your data set by settning this flag
            for adding feature engineering. There are three choices. You can choose one, two or all three.
            'interactions': This will add interaction features to your data such as x1*x2, x2*x3, x1**2, x2**2, etc.
            'groupby': This will generate Group By features to your numeric vars by grouping all categorical vars.
            'target':  This will encode & transform all your categorical features using certain target encoders.
            Default is empty string (which means no additional features)
        category_encoders: Instead of above method, you can choose your own kind of category encoders from below.
            Recommend you do not use more than two of these.
                            Featurewiz will automatically select only two from your list.
            Default is empty string (which means no encoding of your categorical features)
                ['HashingEncoder', 'SumEncoder', 'PolynomialEncoder', 'BackwardDifferenceEncoder',
                'OneHotEncoder', 'HelmertEncoder', 'OrdinalEncoder', 'FrequencyEncoder', 'BaseNEncoder',
                'TargetEncoder', 'CatBoostEncoder', 'WOEEncoder', 'JamesSteinEncoder']
        dask_xgboost_flag: default = True. This flag enables DASK by default so that you can process large
            data sets faster using parallel processing. It detects the number of CPUs and GPU's in your machine
            automatically and sets the num of workers for DASK. It also uses DASK XGBoost to run it.
        nrows: default = None: None means all rows will be used to load into dataframe. If nrows is set,
            then only those rows will be loaded into dataframe.
    ########           Featurewiz Output           #############################
    Output: Tuple
    Featurewiz can output either a list of features or one dataframe or two depending on what you send in.
        1. features: featurewiz will return just a list of important features
                     in your data if you send in just a dataset.
        2. trainm: modified train dataframe is the dataframe that is modified
                        with engineered and selected features from dataname.
        3. testm: modified test dataframe is the dataframe that is modified with
                    engineered and selected features from test_data
    ######################################################################################
    ############       C A V E A T !  C A U T I O N !   W A R N I N G ! ###################
    In general: Be very careful with featurewiz. Don't use it mindlessly
                        to generate un-interpretable features!
    ######################################################################################
    """
    ### set all the defaults here ##############################################
    dataname = copy.deepcopy(dataname)
    max_nums = 30
    max_cats = 15
    maxrows = 10000
    RANDOM_SEED = 42
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
            feature_gen = [feature_engg]
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
    ##################   train will be the name of the dask dataframe of train data ######
    ##################   dataname will be the name of pandas dataframe of train data #####
    ######################################################################################
    print('Loading train data...')
    if isinstance(dataname, str):
        if dask_xgboost_flag:
            try:
                print('    Since dask_xgboost_flag is True, reducing memory size and loading into dask')
                dataname = pd.read_csv(dataname, sep=sep, header=header, nrows=nrows)
                dataname = reduce_mem_usage(dataname)
                train = load_dask_data(dataname, sep)
            except:
                print('File could not be loaded into dask. Check the path or filename and try again')
                return None
        else:
            train = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, nrows=nrows)
            dataname = pd.read_csv(dataname, sep=sep, header=header)
    else:
        if dask_xgboost_flag:
            print('    Since dask_xgboost_flag is True, reducing memory size and loading into dask')
            dataname = reduce_mem_usage(dataname)
            train = load_dask_data(dataname, sep)
        else:
            train = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, nrows=nrows)
    ##################    L O A D    T E S T   D A T A      ######################
    dataname = remove_duplicate_cols_in_dataset(dataname)
    train_index = dataname.index
    ######################################################################################
    ##################    L O A D      T E S T     D A T A   ##############################
    ##################   test will be the name of the dask dataframe of test data ######
    ##################   test_data will be the name of pandas dataframe of test data #####
    ######################################################################################
    print('Loading test data...')
    if dask_xgboost_flag:
        print('    Since dask_xgboost_flag is True, reducing memory size and loading into dask')
        test_data = load_file_dataframe(test_data, sep=sep, header=header, verbose=verbose, nrows=nrows)
        if test_data is not None:
            test_data = reduce_mem_usage(test_data)
            test = load_dask_data(test_data, sep)
    else:
        #### load the entire test dataframe - there is no limit applicable there #########
        test_data = load_file_dataframe(test_data, sep=sep, header=header, verbose=verbose)
    if test_data is not None:
        test_data = remove_duplicate_cols_in_dataset(test_data)
        test_index = test_data.index
    #############    C L A S S I F Y    F E A T U R E S      ####################
    if nrows is None:
        nrows_limit = maxrows
    else:
        nrows_limit = nrows
    if dataname.shape[0] >= nrows_limit:
        print('    since dataframe is larger than %s, loading a small sample to classify features' %nrows_limit)
        if isinstance(target, str):
            targets = [target]
        else:
            targets = copy.deepcopy(target)
        ##### you can use 
        train_small = select_rows_from_dataframe(dataname, targets, nrows_limit, DS_LEN=dataname.shape[0])
        features_dict = classify_features(train_small, target)
    else:
        features_dict = classify_features(dataname, target)
    ################    Load data frame with date var features correctly this time ################
    if len(features_dict['date_vars']) > 0:
        print('Caution: Since there are date-time variables in datatset, it is best to load them using pandas')
        dask_xgboost_flag = False ### Set the dask flag to be False since it is now becoming Pandas dataframe 
        date_time_vars = features_dict['date_vars']
        dataname = load_file_dataframe(dataname, sep=sep, header=header, verbose=verbose, nrows=nrows,
                             parse_dates=date_time_vars)
        if not test_data is None:
            ### You must load the entire test data - there is no limit there ##################
            test_data = load_file_dataframe(test_data, sep=sep, header=header, verbose=verbose, 
                                 parse_dates=date_time_vars)
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
    seed = 1
    early_stopping = 5
    ####### All the default parameters are set up now #########
    kf = KFold(n_splits=n_splits)
    #########     G P U     P R O C E S S I N G      B E G I N S    ############
    ###### This is where we set the CPU and GPU parameters for XGBoost
    GPU_exists = check_if_GPU_exists()
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    cpu_params['nthread'] = -1
    cpu_params['tree_method'] = 'hist'
    cpu_params['grow_policy'] = 'depthwise'
    cpu_params['max_depth'] = max_depth
    cpu_params['max_leaves'] = 0
    cpu_params['verbosity'] = 0
    cpu_params['gpu_id'] = 0
    cpu_params['updater'] = 'grow_colmaker'
    cpu_params['predictor'] = 'cpu_predictor'
    cpu_params['num_parallel_tree'] = 1
    if GPU_exists:
        param['nthread'] = -1
        param['tree_method'] = 'gpu_hist'
        param['grow_policy'] = 'depthwise'
        param['max_depth'] = max_depth
        param['max_leaves'] = 0
        param['verbosity'] = 0
        param['gpu_id'] = 0
        param['updater'] = 'grow_gpu_hist' #'prune'
        param['predictor'] = 'gpu_predictor'
        param['num_parallel_tree'] = 1
        print('    Running XGBoost using GPU parameters')
    else:
        param = copy.deepcopy(cpu_params)
        print('    Running XGBoost using CPU parameters')
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
        #### Since Feature Engineering module cannot handle Multi Label Targets,
        ####   we will turnoff creating target_enc_cat_features to False
        target_enc_cat_features = False
    else:
        ## If target is specified in feature_gen then use it to Generate target encoded features
        target_enc_cat_features = 'target' in feature_gen
    ######################################################################################
    ########     C L A S S I F Y    V A R I A B L E S           ##########################
    ###### Now we detect the various types of variables to see how to convert them to numeric
    ######################################################################################
    if len(features_dict['date_vars']) > 0:
        date_time_vars = features_dict['date_vars']
        date_cols = copy.deepcopy(date_time_vars)
        #### Do this only if date time columns exist in your data set!
        for date_col in date_cols:
            print('Processing %s column for date time features....' %date_col)
            dataname, ts_adds = FE_create_time_series_features(dataname, date_col)
            #date_col_adds_train = left_subtract(date_df_train.columns.tolist(),date_col)
            #print('    Adding %d column(s) from date-time column %s in train' %(len(date_col_adds_train),date_col))
            #train = train.join(date_df_train, rsuffix='2')
            if isinstance(test_data,str) or test_data is None:
                ### do nothing ####
                pass
            else:
                print('        Adding same time series features to test data...')
                test, _ = FE_create_time_series_features(test_data, date_col, ts_adds)
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
    if verbose:
        print('    columns removed: %s' %cols_to_remove)
    preds = [x for x in list(dataname) if x not in target+cols_to_remove]
    print('    After removing redundant variables from further processing, features left = %d' %len(preds))
    numvars = dataname[preds].select_dtypes(include = 'number').columns.tolist()
    if len(numvars) > max_nums:
        if feature_gen:
            print('Warning: Too many extra features will be generated by featurewiz. This may take time...')
    if cat_vars:
        ### if input is given for cat_vars, use it!
        catvars = copy.deepcopy(cat_vars)
        numvars = left_subtract(preds, catvars)
    else:
        catvars = left_subtract(preds, numvars)
    if len(catvars) > max_cats:
        if feature_type:
            print('Warning: Too many extra features will be generated by category encoding. This may take time...')
    rem_vars = copy.deepcopy(catvars)
    ########## Now we need to select the right model to run repeatedly ####
    if target is None or len(target) == 0:
        cols_list = list(dataname)
        settings.modeltype = 'Clustering'
    else:
        settings.modeltype = analyze_problem_type(dataname, target)
        cols_list = left_subtract(list(dataname),target)
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
        print('    Starting feature engineering...this will take time...')
        if isinstance(test_data, str) or test_data is None:
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
        data1 = data_tuple.X_train.join(y_train) ### data_tuple does not have a y_train, remember!
        if isinstance(test_data, str) or test_data is None:
            ### Since you have done a train_test_split using randomized split, you need to put it back again.
            if dask_xgboost_flag:
                ### since y_train is dask df and data_tuple.X_train is a pandas df, you can't merge them.
                y_test = y_test.compute()  ### remember you first have to convert them to a pandas df
            data2 = data_tuple.X_test.join(y_test)
            dataname = data1.append(data2)
            dataname = dataname.reindex(train_index)
        else:
            try:
                if dask_xgboost_flag:
                    ### since y_train is dask df and data_tuple.X_train is a pandas df, you can't merge them.
                    y_test = y_test.compute()  ### remember you first have to convert them to a pandas df
                test_data = data_tuple.X_test.join(y_test)
            except:
                test_data = copy.deepcopy(data_tuple.X_test)
            test_data = test_data.reindex(test_index)
            dataname = copy.deepcopy(data1)
        print('    Completed feature engineering. Shape of Train (with target) = %s' %(dataname.shape,))
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
        top_num = int(len(preds)*0.10)
    ######################   I M P O R T A N T ##############################################
    important_cats = copy.deepcopy(catvars)
    data_dim = int((len(dataname)*dataname.shape[1])/1e6)
    ################################################################################################
    ############     S   U  L  O   V       M   E   T   H   O  D      ###############################
    #### If the data dimension is less than 5o Million then do SULOV - otherwise skip it! #########
    ################################################################################################
    cols_with_infinity = EDA_find_columns_with_infinity(dataname)
    # first you must drop rows that have inf in them ####
    if cols_with_infinity:
        print('Found %d columns which contain infinity in dataset' %len(cols_with_infinity))
        dataname = FE_drop_rows_with_infinity(dataname, cols_with_infinity, fill_value=True)
    #######  This is where you start the process ##################################    
    if len(numvars) > 1:
        if data_dim < 50:
            try:
                final_list = FE_remove_variables_using_SULOV_method(dataname,numvars,settings.modeltype,target,
                             corr_limit,verbose)
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
    print('    Adding %s categorical variables to reduced numeric variables  of %d' %(
                            len(important_cats),len(final_list)))
    if  isinstance(final_list,np.ndarray):
        final_list = final_list.tolist()
    preds = final_list+important_cats
    #######You must convert category variables into integers ###############
    if len(important_cats) > 0:
        dataname, test_data = FE_convert_all_object_columns_to_numeric(dataname,  test_data)
    print('############## F E A T U R E   S E L E C T I O N  ####################')
    important_features = []
    #####   This is for DASK XGB Regressor and XGB Classifier problems ####
    bst_models = []
    if dask_xgboost_flag:
        #########   This is for DASK Dataframes XGBoost training ####################
        #################################################################################################
        ########   Now if dask_xgboost_flag is True, convert pandas dfs back to Dask Dataframes     #####
        #################################################################################################
        train = load_dask_data(dataname, sep)
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
        print('Dask version = %s' %dask.__version__)
        from dask_ml.model_selection import train_test_split
        ### check available memory and allocate at least 1GB of it in the Client in DASK #############################
        n_workers = get_cpu_worker_count()
        memory_free = str(max(1, int(psutil.virtual_memory()[0]/(n_workers*1e9))))+'GB'
        print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(get_cpu_worker_count(), memory_free))
        client = Client(n_workers= n_workers, threads_per_worker=1, processes=True, silence_logs=50,
                        memory_limit=memory_free)
        print('Dask client configuration: %s' %client)
        print('XGBoost version: %s' %xgb.__version__)
        import gc
        client.run(gc.collect) 
        train_p = client.persist(train_p)
        try:
            for i in range(0,train_p.shape[1],iter_limit):
                start_time2 = time.time()
                print('        using %d variables...' %(train_p.shape[1]-i))
                cat_targets = dataname[target].select_dtypes(include='object').columns.tolist() + dataname[target].select_dtypes(include='category').columns.tolist()
                if cat_targets:
                    for each_target in cat_targets:
                        MLB = My_LabelEncoder()
                        dataname[each_target] = MLB.fit_transform(dataname[each_target])
                    ### since dataname is still a pandas df, you need to convert it into dask df ##
                    y = load_dask_data(dataname[target], sep)
                else:
                    ### train is already a dask dataframe -> you can leave it as it is
                    y = train[target]
                imp_feats = []
                if train_p.shape[1]-i < iter_limit:
                    X = train_p.iloc[:,i:]
                    cols_sel = X.columns.tolist()
                else:
                    X = train_p[list(train_p.columns.values)[i:train_p.shape[1]]]
                    cols_sel = X.columns.tolist()
                ##### This is where you repeat the training and finding feature importances
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
                try:
                    cols_infinity = EDA_find_columns_with_infinity(X_train)
                    if cols_infinity:
                        ### drop columns with infinite values ###
                        X_train = X_train.drop(cols_infinity,axis=1)
                    cols_infinity = EDA_find_columns_with_infinity(X_test)
                    if cols_infinity:
                        X_test = X_test.drop(cols_infinity,axis=1)
                except:
                    print('Finding columns with infinite values erroring. Continuing...')
                if not y_train.dtypes[0] == float:
                    y_train = y_train.astype(int) 
                if settings.modeltype == 'Regression':
                    objective = 'reg:squarederror'
                    params = {'objective': objective, 'max_depth': 4, 'eta': 0.1, 'subsample': 0.5, 
                                        'min_child_weight': 0.5}
                else:
                    #### This is for Classifiers only ##########                    
                    if settings.modeltype == 'Binary_Classification':
                        objective = 'binary:logistic'
                        num_class = 1
                        params = {'objective': objective, 'max_depth': 4, 'eta': 0.1, 'subsample': 0.5, 
                                            'min_child_weight': 0.5}
                    else:
                        objective = 'multi:softmax'
                        num_class  =  dataname[target].nunique()[0]
                        params = {'objective': objective, 'max_depth': 4, 'eta': 0.1, 'subsample': 0.5, 
                                            'min_child_weight': 0.5, 'num_class': num_class}
                ##########   Training XGBoost model using dask_xgboost #########################
                ### the dtrain syntax can only be used xgboost 1.50 or greater. Dont use it until then.
                #dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train, enable_categorical=True)
                #### now this training via bst works well for both xgboost 0.0.90 as well as 1.5.1 ##
                if X_train.compute().shape[0] >= 100000:
                    num_rounds = 50
                else:
                    num_rounds = 100
                
                bst = dask_xgboost.train(client, params, X_train, y_train, num_boost_round=num_rounds)
                #### SYNTAX BELOW DOES NOT WORK YET. YOU CANNOT DO EVALS WITH DASK XGBOOST AS OF NOW ####
                #bst = dask_xgboost.train(client=client, params=params, data=X_train, labels=y_train, 
                #                num_boost_round=10, evals=[(dtrain, 'train')])
                ################################################################################
                bst_models.append(bst)
                imp_feats = bst.get_score(fmap='', importance_type='gain')
                imp_feats = dict(sorted(imp_feats.items(),reverse=True, key=lambda item: item[1]))
                ### doing this for single-label is a little different from settings.multi_label #########
                #imp_feats = model_xgb.get_booster().get_score(importance_type='gain')
                #print('%d iteration: imp_feats = %s' %(i+1,imp_feats))
                important_features += pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
                #######  order this in the same order in which they were collected ######
                important_features = list(OrderedDict.fromkeys(important_features))
                print('            Time taken for training: %0.0f seconds' %(time.time()-start_time2))
            #### plot all the feature importances in a grid ###########
            if verbose >= 2:
                draw_feature_importances_all(bst_models)
        except:
            print('Dask XGBoost is crashing. Continuing...')
            important_features = copy.deepcopy(preds)
            return important_features, train[important_features+target]
    else:
        #########  This is for pandas dataframes only ##################
        ########    Fill Missing values since XGB for some reason  #########
        ########    can't handle missing values in early stopping rounds #######
        null_vars = dataname.select_dtypes(include='number').columns
        null_vars = null_vars[(dataname[null_vars].isnull().sum()>0).values].tolist()
        for each_num in null_vars:
            dataname[each_num] = dataname[each_num].fillna(0)
        train = copy.deepcopy(dataname)
        y = train[target]
        if not test_data is None:
            test = copy.deepcopy(test_data)
        ########## This is for Single_Label problems ###################
        bst_models = []
        from sklearn.model_selection import train_test_split
        #### Since Category Encoding took place, these cat variables are no longer in Train. So remove them!
        if feature_gen or feature_type:
            print('Since %s category encoding is done, dropping original categorical vars from predictors...' %feature_gen)
            preds = left_subtract(preds, catvars)
        train_p = train[preds]
        if train_p.shape[1] < 10:
            iter_limit = 2
        else:
            iter_limit = int(train_p.shape[1]/5+0.5)
        print('Current number of predictors = %d ' %(train_p.shape[1],))
        print('    Finding Important Features using Boosted Trees algorithm...')
        ######## This is where we start training the XGBoost model to find top features ####
        try:
            for i in range(0,train_p.shape[1],iter_limit):
                if settings.modeltype == 'Regression':
                    objective = 'reg:squarederror'
                    model_xgb = XGBRegressor( n_estimators=100,booster='gbtree',subsample=subsample,objective=objective,
                                            colsample_bytree=col_sub_sample,reg_alpha=0.5, reg_lambda=0.5,
                                             seed=1,n_jobs=-1,random_state=1)
                    eval_metric = 'rmse'
                else:
                    #### This is for Classifiers only   ####
                    if settings.modeltype == 'Binary_Classification':
                        model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                            colsample_bytree=col_sub_sample,gamma=1, learning_rate=0.1, max_delta_step=0,
                            max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=100,
                            n_jobs=-1, nthread=None, objective='binary:logistic',
                            random_state=1, reg_alpha=0.5, reg_lambda=0.5,
                            seed=1)
                        eval_metric = 'logloss'
                    else:
                        num_class  =  train[target].nunique()[0]
                        model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                                    colsample_bytree=col_sub_sample, gamma=1, learning_rate=0.1, max_delta_step=0,
                            max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=100,
                            n_jobs=-1, nthread=None, objective='multi:softmax',
                            random_state=1, reg_alpha=0.5, reg_lambda=0.5, num_class= num_class,
                            seed=1)
                        eval_metric = 'mlogloss'
                #### Now set the parameters for XGBoost ###################
                model_xgb.set_params(**param)
                #print('Model parameters: %s' %model_xgb)
                if settings.multi_label:
                    ########## This is for settings.multi_label problems ###############################
                    if settings.modeltype == 'Regression':
                        model_xgb = MultiOutputRegressor(model_xgb)
                        #model_xgb = RegressorChain(model_xgb)
                    else:
                        ## just do randomized search CV - no need to do one vs rest unless multi-class
                        model_xgb = MultiOutputClassifier(model_xgb)
                        #model_xgb = ClassifierChain(model_xgb)
                ####   This is where you start to Iterate on Finding Important Features ################
                save_xgb = copy.deepcopy(model_xgb)
                new_xgb = copy.deepcopy(save_xgb)
                print('        using %d variables...' %(train_p.shape[1]-i))
                imp_feats = []
                if train_p.shape[1]-i < iter_limit:
                    X = train_p.iloc[:,i:]
                    cols_sel = X.columns.tolist()
                    if settings.modeltype == 'Regression':
                        train_part = int((1-test_size)*X.shape[0])
                        X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
                    else:
                        X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                    test_size=test_size, random_state=seed)
                    try:
                        if settings.multi_label:
                            eval_set = [(X_train.values,y_train.values),(X_cv.values,y_cv.values)]
                        else:
                            eval_set = [(X_train,y_train),(X_cv,y_cv)]
                        if settings.multi_label:
                            model_xgb.fit(X_train,y_train)
                        else:
                            model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,eval_set=eval_set,
                                                eval_metric=eval_metric,verbose=False)
                    except:
                        #### On Colab, even though GPU exists, many people don't turn it on.
                        ####  In that case, XGBoost blows up when gpu_predictor is used.
                        ####  This is to turn it back to cpu_predictor in case GPU errors!
                        if GPU_exists:
                            print('Warning: GPU exists but it is not turned on. Using CPU for predictions...')
                            if settings.multi_label:
                                model_xgb.estimator.set_params(**cpu_params)
                                model_xgb.fit(X_train,y_train)
                            else:
                                model_xgb.set_params(**cpu_params)
                                model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,eval_set=eval_set,
                                            eval_metric=eval_metric,verbose=False)
                    #### This is where you collect the feature importances from each run ############
                    if settings.multi_label:
                        ### doing this for multi-label is a little different for single label #########
                        imp_feats = [model_xgb.estimators_[i].feature_importances_ for i in range(len(target))]
                        imp_feats_df = pd.DataFrame(imp_feats).T
                        imp_feats_df.columns = target
                        imp_feats_df.index = cols_sel
                        imp_feats_df['sum'] = imp_feats_df.sum(axis=1).values
                        important_features += imp_feats_df.sort_values(by='sum',ascending=False)[:top_num].index.tolist()
                    else:
                        ### doing this for single-label is a little different from settings.multi_label #########
                        imp_feats = model_xgb.get_booster().get_score(importance_type='gain')
                        #print('%d iteration: imp_feats = %s' %(i+1,imp_feats))
                        important_features += pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
                    #######  order this in the same order in which they were collected ######
                    important_features = list(OrderedDict.fromkeys(important_features))
                    bst_models.append(model_xgb)
                else:
                    X = train_p[list(train_p.columns.values)[i:train_p.shape[1]]]
                    cols_sel = X.columns.tolist()
                    #### Split here into train and test #####
                    if settings.modeltype == 'Regression':
                        train_part = int((1-test_size)*X.shape[0])
                        X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
                    else:
                        X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                        test_size=test_size, random_state=seed)
                    ### set the validation data as arrays in multi-label case #####
                    if settings.multi_label:
                        eval_set = [(X_train.values,y_train.values),(X_cv.values,y_cv.values)]
                    else:
                        eval_set = [(X_train,y_train),(X_cv,y_cv)]
                    ########## Try training the model now #####################
                    try:
                        if settings.multi_label:
                            model_xgb.fit(X_train,y_train)
                        else:
                            model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,
                                      eval_set=eval_set,eval_metric=eval_metric,verbose=False)
                    except:
                        #### On Colab, even though GPU exists, many people don't turn it on.
                        ####  In that case, XGBoost blows up when gpu_predictor is used.
                        ####  This is to turn it back to cpu_predictor in case GPU errors!
                        if GPU_exists:
                            print('Warning: GPU exists but it is not turned on. Using CPU for predictions...')
                            if settings.multi_label:
                                model_xgb.estimator.set_params(**cpu_params)
                                model_xgb.fit(X_train,y_train)
                            else:
                                model_xgb.set_params(**cpu_params)
                                model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,
                                  eval_set=eval_set,eval_metric=eval_metric,verbose=False)
                    ### doing this for multi-label is a little different for single label #########
                    if settings.multi_label:
                        imp_feats = [model_xgb.estimators_[i].feature_importances_ for i in range(len(target))]
                        imp_feats_df = pd.DataFrame(imp_feats).T
                        imp_feats_df.columns = target
                        imp_feats_df.index = cols_sel
                        imp_feats_df['sum'] = imp_feats_df.sum(axis=1).values
                        important_features += imp_feats_df.sort_values(by='sum',ascending=False)[:top_num].index.tolist()
                    else:
                        imp_feats = model_xgb.get_booster().get_score(importance_type='gain')
                        imp_feats = dict(sorted(imp_feats.items(),reverse=True, key=lambda item: item[1]))
                        important_features += pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
                    important_features = list(OrderedDict.fromkeys(important_features))
                    bst_models.append(model_xgb)
        except:
            print('Finding top features using XGB is crashing. Continuing with all predictors...')
            important_features = copy.deepcopy(preds)
            return important_features, train[important_features+target]
        if verbose >= 2:
            if settings.multi_label:
                print('No feature importance plots available for multi-label datasets')
            else:
                draw_feature_importances_all(bst_models)
    important_features = list(OrderedDict.fromkeys(important_features))
    if len(important_features) <= 30:
        print('Selected %d important features:\n%s' %(len(important_features), important_features))
    else:
        print('Selected %d important features. Too many to print...' %len(important_features))
    numvars = [x for x in numvars if x in important_features]
    important_cats = [x for x in important_cats if x in important_features]
    print('    Time taken (in seconds) = %0.0f' %(time.time()-start_time))
    if isinstance(test_data, str) or test_data is None:
        print(f'Returning list of {len(important_features)} important features and dataframe.')
        if len(np.intersect1d(train_ids.columns.tolist(),dataname.columns.tolist())) > 0:
            return important_features, dataname[important_features+target]
        else:
            dataname = train_ids.join(dataname)
            return important_features, dataname[idcols+important_features+target]
    else:
        print('Returning 2 dataframes: dataname and test_data with %d important features.' %len(important_features))
        if feature_gen or feature_type:
            ### if feature engg is performed, id columns are dropped. Hence they must rejoin here.
            dataname = train_ids.join(dataname)
            test_data = test_ids.join(test_data)
        if target in list(test_data): ### see if target exists in this test data
            return dataname[idcols+important_features+target], test_data[idcols+important_features+target]
        else:
            return dataname[idcols+important_features+target], test_data[idcols+important_features]
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
import os
def check_if_GPU_exists():
    GPU_exists = False
    if not GPU_exists:
        try:
            os.environ['NVIDIA_VISIBLE_DEVICES']
            print('GPU active on this device')
            return True
        except:
            print('No GPU active on this device')
            return False
    else:
        return True
#############################################################################################
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
def draw_feature_importances_all(bst_models):
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
                    ax1 = xgboost.plot_importance(bst_models[counter], height=0.8, show_values=False,
                            importance_type='gain', max_num_features=10, ax=ax[k][l])
                except:
                    pass
                ax1.set_title('Top 10 features with XGB model %s' %(counter+1))
            counter += 1
    plt.show();
######################################################################################
def find_remove_duplicates(list_of_values):
    """
    # Removes duplicates from a list to return unique values - USED ONLY ONCE
    """
    output = []
    seen = set()
    for value in list_of_values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
################################################################################
def reduce_mem_usage(df):
    """
    #####################################################################
    Greatly indebted to :
    https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
        for this function to reduce memory usage.
    #####################################################################
    It is a bit slow as it iterate through all the columns of a dataframe and modifies data types
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
            c_min = df[col].min()
            c_max = df[col].max()
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    #######  Results after memory usage function ###################
    end_mem = df.memory_usage().sum() / 1024**2
    if type(df) == dask.dataframe.core.DataFrame:
        end_mem = end_mem.compute()
    print('    Memory usage after optimization is: {:.2f} MB'.format(end_mem))
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
    df[field] = df[field].fillna(filler)
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
import copy
from sklearn.base import TransformerMixin
from collections import defaultdict
class My_Groupby_Encoder(TransformerMixin):
    """
    #################################################################################################
    ######  This Groupby_Encoder Class works just like any Transformer in sklearn  ##################
    #####  You can add any groupby features based on categorical columns in a data frame  ###########
    #####  The beauty of this function is that it can take care of NaN's and unknown values in Test.#
    #####  It uses the same fit() and fit_transform() methods of sklearn's LabelEncoder class.  #####
    #################################################################################################
    ###   This function is a very fast function that will iteratively compute aggregates for all numeric columns
    ###   It returns original dataframe with added features using numeric variables grouped and aggregated
    ###   What do you mean aggregate? aggregates can be "count, "mean", "median", "mode", "min", "max", etc.
    ###   What do you aggregrate? all numeric columns in your data
    ###   What do you groupby? a groupby column
    ###      except those numeric variables you designate in the ignore_variables list. Can be empty.
    #################################################################################################
    ### Inputs:
    ###   dft: Just sent in the data frame df that you want features added to
    ###   agg_types: list of computational types: 'mean','median','count', 'max', 'min', 'sum', etc.
    ###         One caveat: these agg_types must be found in the following agg_func of numpy or pandas groupby statement.
    ###         List of aggregates available: {'count','sum','mean','mad','median','min','max','mode','abs',
    ###               'prod','std','var','sem','skew','kurt',
    ###                'quantile','cumsum','cumprod','cummax','cummin'}
    ###   groupby_column: this is to groupby all the numeric features and compute aggregates by.
    ###   ignore_variables: list of variables to ignore among numeric variables in data since they may be ID variables.
    ### Outputs:
    ###     dft: original dataframe with tons of additional features created by this function.
    #################################################################################################
    ###     Make sure you reduce correlated variables by using FE_remove_variables_using_SULOV_method()
    Usage:
        MGB = My_Groupby_Encoder(groupby_column, agg_types, ignore_variables=[])
        MGB.fit(train)
        train = MGB.transform(train)
        test = MGB.transform(test)
    """
    def __init__(self, groupby_column, agg_types, ignore_variables=[]):
        if isinstance(groupby_column, str):
            self.groupby_column = [groupby_column]
        else:
            self.groupby_column = groupby_column
        if isinstance(agg_types, str):
            self.agg_types = [agg_types]
        else:
            self.agg_types = agg_types

        if isinstance(ignore_variables, str):
            self.ignore_variables = [ignore_variables]
        else:
            self.ignore_variables = ignore_variables
        self.func_set = {'count','sum','mean','mad','median','min','max','mode',
                        'abs','prod','std','var','sem','skew','kurt',
                        'quantile','cumsum','cumprod','cummax','cummin'}
        self.train_cols = []  ## this keeps track of which cols were created ###
        self.MLB_dict = {}

    def fit(self, dft):
        dft = copy.deepcopy(dft)
        if isinstance(dft, pd.Series):
            print('data to transform must be a dataframe')
            return self
        elif isinstance(dft, np.ndarray):
            print('data to transform must be a dataframe')
            return self
        ### Make sure the list of functions they send in are acceptable functions. If not, the aggregate will blow up!
        ### Only select those that match the func set ############
        self.agg_types = list(set(self.agg_types).intersection(self.func_set))
        copy_cols = copy.deepcopy(self.groupby_column)
        for each_col in copy_cols:
            MLB = My_LabelEncoder()
            dft[each_col] = MLB.fit(dft[each_col])
            self.MLB_dict[each_col] = MLB

        return self

    def transform(self, dft ):
        ##### First make a copy of dataframe ###
        dft = copy.deepcopy(dft)
        if isinstance(dft, pd.Series):
            print('data to transform must be a dataframe')
            return self
        elif isinstance(dft, np.ndarray):
            print('data to transform must be a dataframe')
            return self
        try:
            ###
            ### first if groupby cols had NaN's you need to fill them before aggregating
            ### If you don't do that, then your groupby aggregating will miss those NaNs
            copy_cols = copy.deepcopy(self.groupby_column)
            for each_col in copy_cols:
                MLB = self.MLB_dict[each_col]
                dft[each_col] = MLB.transform(dft[each_col])

            ## Since you want to ignore some variables, you can drop them here
            ls = dft.select_dtypes('number').columns.tolist()
            ignore_in_list = [x for x in self.ignore_variables if x in ls]
            if len(ignore_in_list) == len(self.ignore_variables) and left_subtract(ignore_in_list,self.ignore_variables)==[]:
                dft_cont = copy.deepcopy(dft.select_dtypes('number').drop(self.ignore_variables,axis=1))
            else:
                dft_cont = copy.deepcopy(dft.select_dtypes('number'))

            #### This is the main part where we create aggregated columns ######
            dft_full = dft_cont.groupby(self.groupby_column).agg(self.agg_types)
            if len(self.groupby_column) == 1:
                str_col = self.groupby_column[0]
            else:
                str_col = "_".join(self.groupby_column)
            cols =  [x+'_by_'+str_col+'_'+y for (x,y) in dft_full.columns]
            dft_full.columns = cols
            dft_full = dft_full.reset_index()

            # make sure there are no zero-variance cols. If so, drop them #
            if len(self.train_cols) == 0:
                #### drop zero variance cols the first time
                copy_cols = copy.deepcopy(cols)
                for each_col in cols:
                    if len(dft_full[each_col].value_counts()) == 1:
                        dft_full.drop = dft_full.drop(each_col, axis=1)
                num_cols_created = dft_full.shape[1] - len(self.groupby_column)
                print('%d new columns created for numeric data grouped by %s for aggregates %s' %(num_cols_created,
                                    self.groupby_column, self.agg_types))
                self.train_cols = dft_full.columns.tolist()
            else:
                #### if it is the second time, just use column names created during train
                if len(left_subtract(self.train_cols, list(dft_full))) == 0:
                    #### make sure that they are the exact same columns, if not, leave dft_full as is
                    dft_full = dft_full[self.train_cols]
                else:
                    print('Alert! train and test have different number of columns')


            dft = dft.merge(dft_full, on=self.groupby_column, how='left')

            #### Now change the label encoded columns back to original status ##
            copy_cols = copy.deepcopy(self.groupby_column)
            for each_col in copy_cols:
                MLB = self.MLB_dict[each_col]
                dft[each_col] = MLB.inverse_transform(dft[each_col])
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly,
            ### if for some reason, the groupby blows up, then just return the dataframe as is - no changes!
            print('Error in groupby function: returning dataframe as is')
            return dft
        return dft
###################################################################################
def FE_add_groupby_features_aggregated_to_dataframe(train,
                    agg_types,groupby_columns,ignore_variables, test=""):
    """
    FE stands for Feature Engineering. That means this function performs feature engineering on data.
    ######################################################################################
    ###   This function is a very fast function that will iteratively compute aggregates for all numeric columns
    ###   It returns original dataframe with added features using numeric variables grouped and aggregated
    ###   What do you mean aggregate? aggregates can be "count, "mean", "median", "mode", "min", "max", etc.
    ###   What do you aggregrate? all numeric columns in your data
    ###   What do you groupby? one groupby column at a time or multiple columns one by one
    ###     -- if you give it a list of columns, it will execute the grouping one by one
    ###   What is the ignore_variables for? it will ignore these variables from grouping.
    ######################################################################################
    ### Inputs:
    ###   train: Just sent in the data frame df that you want features added to
    ###   agg_types: list of computational types: 'mean','median','count', 'max', 'min', 'sum', etc.
    ###         One caveat: these agg_types must be found in the following agg_func of numpy or pandas groupby statement.
    ###         List of aggregates available: {'count','sum','mean','mad','median','min','max','mode','abs',
    ###               'prod','std','var','sem','skew','kurt',
    ###                'quantile','cumsum','cumprod','cummax','cummin'}
    ###   groupby_columns: can be a string representing a single column or a list of multiple columns
    ###               - it will groupby all the numeric features using one groupby column at a time in a loop.
    ###   ignore_variables: list of variables to ignore among numeric variables in data since they may be ID variables.
    ###   test: (optional) a data frame  that you want features added to based on train
    ### Outputs:
    ###     trainm: original dataframe with tons of additional features created by this function.
    ###     testm: (optional) dataframe with tons of additional features based on train
    ######################################################################################
    ###     Make sure you reduce correlated variables by using FE_remove_variables_using_SULOV_method()
    """
    train_copy = copy.deepcopy(train)
    test_copy = copy.deepcopy(test)
    if isinstance(groupby_columns, str):
        groupby_columns = [groupby_columns]

    for groupby_column in groupby_columns:
        train_copy_index = train_copy.index
        MGB = My_Groupby_Encoder(groupby_column, agg_types, ignore_variables)
        train1 = MGB.fit_transform(train)
        addl_cols = left_subtract(train1.columns,train.columns)
        train1.index = train_copy_index
        train_copy = pd.concat([train_copy,train1[addl_cols]], axis=1)
        if isinstance(test, str) or test is None:
            pass
        else:
            test_copy_index = test_copy.index
            test1 = MGB.transform(test)
            addl_cols = left_subtract(test1.columns,test.columns)
            test1.index = test_copy_index
            test_copy = pd.concat([test_copy,test1[addl_cols]],axis=1)
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
        df[name_col] = df[name_col].fillna(0)
        dt_adds.append(name_col)
        summer = ['Jun','Jul','Aug']
        name_col = tscol+"_is_summer"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in summer else 0).values
        df[name_col] = df[name_col].fillna(0)
        dt_adds.append(name_col)
        winter = ['Dec','Jan','Feb']
        name_col = tscol+"_is_winter"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in winter else 0).values
        df[name_col] = df[name_col].fillna(0)
        dt_adds.append(name_col)
        cold = ['Oct','Nov','Dec','Jan','Feb','Mar']
        name_col = tscol+"_is_cold"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in cold else 0).values
        df[name_col] = df[name_col].fillna(0)
        dt_adds.append(name_col)
        warm = ['Apr','May','Jun','Jul','Aug','Sep']
        name_col = tscol+"_is_warm"
        df[name_col] = 0
        df[name_col] = df[tscol+'_month'].map(lambda x: 1 if x in warm else 0).values
        df[name_col] = df[name_col].fillna(0)
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
            dtf[ts_column] = dtf[ts_column].fillna(method='ffill')
            print('        adding %s column due to missing values in data' %new_missing_col)
            if dtf[dtf[ts_column].isnull()].shape[0] > 0:
                dtf[ts_column] = dtf[ts_column].fillna(method='bfill')

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
            rem_cols = left_subtract(dtf.columns.tolist(), ts_adds_in)
            dtf = dtf[rem_cols+ts_adds_in]
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
from sklearn.base import TransformerMixin
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
            if df1[each_cat].map(len).mean() >=40:
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
from tqdm.notebook import tqdm
from pathlib import Path

#sklearn data_preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#sklearn categorical encoding
import category_encoders as ce

#sklearn modelling
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

# boosting library
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

#################################################################################
from sklearn.impute import SimpleImputer
def data_transform(X_train, y_train, X_test="", enc_method='label', scaler = StandardScaler()):
    X_train_encoded = X_train.copy()
    X_test_encoded= copy.deepcopy(X_test)
    # Set up feature to encode
    feature_to_encode = X_train.columns[X_train.dtypes == 'O'].tolist()
    #print('features to label encode: %s' %feature_to_encode)
    #### Do label encoding now #################
    if enc_method == 'label':
        for feat in feature_to_encode:
            # Initia the encoder model
            lbEncoder = My_LabelEncoder()
            # fit the train data
            lbEncoder.fit(X_train[feat])
            # transform training set
            X_train_encoded[feat] = lbEncoder.transform(X_train[feat])
            # transform test set
            if not isinstance(X_test_encoded, str):
                X_test_encoded[feat] = lbEncoder.transform(X_test[feat])
    elif enc_method == 'glmm':
        # Initialize the encoder model
        GLMMEncoder = ce.glmm.GLMMEncoder(verbose=0 ,binomial_target=False)
        # fit the train data
        GLMMEncoder.fit(X_train[feature_to_encode],y_train)
        # transform training set
        X_train_encoded[feature_to_encode] = GLMMEncoder.transform(X_train[feature_to_encode])
        # transform test set
        if not isinstance(X_test_encoded, str):
            X_test_encoded[feature_to_encode] = GLMMEncoder.transform(X_test[feature_to_encode])
    else:
    	raise 'No encoding method stated'

    ### make sure there are no missing values ###
    try:
        imputer = SimpleImputer(strategy='constant', fill_value=0, verbose=0, add_indicator=True)
        imputer.fit_transform(X_train_encoded)
        if not isinstance(X_test_encoded, str):
            imputer.transform(X_test_encoded)
    except:
        X_train_encoded = X_train_encoded.fillna(0)
        if not isinstance(X_test_encoded, str):
            X_test_encoded = X_test_encoded.fillna(0)

    # fit the scaler to the entire train and transform the test set
    scaler.fit(X_train_encoded)
    # transform training set
    X_train_scaled = pd.DataFrame(scaler.transform(X_train_encoded), columns=X_train_encoded.columns, index=X_train_encoded.index)
    # transform test set
    if not isinstance(X_test_encoded, str):
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)
    return X_train_scaled, X_test_scaled
##################################################################################
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
import csv
import re
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_log_error, mean_squared_error,balanced_accuracy_score
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
import scipy as sp
import time
##################################################################################
def xgb_model_fit(model, x_train, y_train, x_test, y_test, modeltype, log_y, params, cpu_params):
    start_time = time.time()
    if str(model).split("(")[0] == 'RandomizedSearchCV':
        model.fit(x_train, y_train)
        print('Time take for Hyper Param tuning of XGB (in minutes) = %0.1f' %(
                                        (time.time()-start_time)/60))
    else:
        try:
            if modeltype == 'Regression':
                if log_y:
                    model.fit(x_train, y_train, early_stopping_rounds=6, eval_metric=['rmse'],
                            eval_set=[(x_test, np.log(y_test))], verbose=0)
                else:
                    model.fit(x_train, y_train, early_stopping_rounds=6, eval_metric=['rmse'],
                            eval_set=[(x_test, y_test)], verbose=0)
            else:
                if y_train.nunique() <= 2:
                    objective='binary:logistic'
                    eval_metric = 'logloss'
                else:
                    objective='multi:softmax'
                    eval_metric = 'mlogloss'
                model.fit(x_train, y_train, early_stopping_rounds=6, eval_metric = eval_metric,
                                eval_set=[(x_test, y_test)], verbose=0)
        except:
            print('GPU is present but not turned on. Please restart after that. Currently using CPU...')
            if str(model).split("(")[0] == 'RandomizedSearchCV':
                xgb = model.estimator_
                xgb.set_params(**cpu_params)
                model = RandomizedSearchCV(xgb,
                                                   param_distributions = params,
                                                   n_iter = 10,
                                                   n_jobs=-1,
                                                   cv=5)
            else:
                model = model.set_params(**cpu_params)
            if modeltype == 'Regression':
                if log_y:
                    model.fit(x_train, y_train, early_stopping_rounds=6, eval_metric=['rmse'],
                            eval_set=[(x_test, np.log(y_test))], verbose=0)
                else:
                    model.fit(x_train, y_train, early_stopping_rounds=6, eval_metric=['rmse'],
                            eval_set=[(x_test, y_test)], verbose=0)
            else:
                model.fit(x_train, y_train, early_stopping_rounds=6,eval_metric=eval_metric,
                                eval_set=[(x_test, y_test)], verbose=0)
    return model
#################################################################################
def simple_XGBoost_model(X_XGB, Y_XGB, X_XGB_test, modeltype, log_y=False, GPU_flag=False,
                                scaler = '', enc_method='label',verbose=0):
    """
    Easy to use XGBoost model. Just send in X_train, y_train and what you want to predict, X_test
    It will automatically split X_train into multiple folds (10) and train and predict each time on X_test.
    It will then use average (or use mode) to combine the results and give you a y_test.
    You just need to give the modeltype as "Regression" or 'Classification'

    Inputs:
    ------------
    X_XGB: pandas dataframe only: do not send in numpy arrays. This is the X_train of your dataset.
    Y_XGB: pandas Series or DataFrame only: do not send in numpy arrays. This is the y_train of your dataset.
    X_XGB_test: pandas dataframe only: do not send in numpy arrays. This is the X_test of your dataset.
    modeltype: can only be 'Regression' or 'Classification'
    log_y: default = False: If True, it means use the log of the target variable "y" to train and test.
    GPU_flag: if your machine has a GPU set this flag and it will use XGBoost GPU to speed up processing.
    scaler : default is StandardScaler(). But you can send in MinMaxScaler() as input to change it or any other scaler.
    enc_method: default is 'label' encoding. But you can choose 'glmm' as an alternative. But those are the only two.
    verbose: default = 0. Choosing 1 will give you lot more output.

    Outputs:
    ------------
    y_preds: Predicted values for your X_XGB_test dataframe.
        It has been averaged after repeatedly predicting on X_XGB_test. So likely to be better than one model.
    """
    columns =  X_XGB.columns
    if isinstance(scaler, str):
        if not scaler == '':
            scaler = scaler.lower()
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    #########     G P U     P R O C E S S I N G      B E G I N S    ############
    ###### This is where we set the CPU and GPU parameters for XGBoost
    if GPU_flag:
        GPU_exists = check_if_GPU_exists()
    else:
        GPU_exists = False
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    cpu_params['tree_method'] = 'hist'
    cpu_params['gpu_id'] = 0
    cpu_params['updater'] = 'grow_colmaker'
    cpu_params['predictor'] = 'cpu_predictor'
    if GPU_exists:
        param['tree_method'] = 'gpu_hist'
        param['gpu_id'] = 0
        param['updater'] = 'grow_gpu_hist' #'prune'
        param['predictor'] = 'gpu_predictor'
        print('    Running XGBoost using GPU parameters')
    else:
        param = copy.deepcopy(cpu_params)
        print('    Running XGBoost using CPU parameters')
    #################################################################################
    if modeltype == 'Regression':
        if log_y:
            Y_XGB.loc[Y_XGB==0] = 1e-15  ### just set something that is zero to a very small number
        xgb = XGBRegressor(
                          booster = 'gbtree',
                          colsample_bytree=0.5,
                          alpha=0.015,
                          gamma=4,
                          learning_rate=0.1,
                          max_depth=15,
                          min_child_weight=2,
                          n_estimators=1000,
                          reg_lambda=0.5,
          	              #reg_alpha=8,
                          subsample=0.7,
                          random_state=99,
                          objective='reg:squarederror',
          	              eval_metric='rmse',
                          verbosity = 0,
                          n_jobs=-1,
                          silent = True)
    else:
        if Y_XGB.nunique() <= 2:
            objective='binary:logistic'
            eval_metric = 'logloss'
        else:
            objective='multi:softmax'
            eval_metric = 'mlogloss'
        xgb = XGBClassifier(
                         booster = 'gbtree',
                         colsample_bytree=0.5,
                         alpha=0.015,
                         gamma=4,
                         learning_rate=0.1,
                         max_depth=15,
                         min_child_weight=2,
                         n_estimators=1000,
                         reg_lambda=0.5,
                         objective=objective,
                         subsample=0.7,
                         random_state=99,
                         n_jobs=-1,
                         verbosity = 0,
                         silent = True)

    #testing for GPU
    model = xgb.set_params(**param)
    if X_XGB.shape[0] >= 1000000:
        hyper_frac = 0.1
    elif X_XGB.shape[0] >= 100000:
        hyper_frac = 0.2
    elif X_XGB.shape[0] >= 10000:
        hyper_frac = 0.3
    else:
        hyper_frac = 0.4
    #### now select a random sample from X_XGB ##
    if modeltype == 'Regression':
        X_XGB_sample = X_XGB[:int(hyper_frac*X_XGB.shape[0])]
        Y_XGB_sample = Y_XGB[:int(hyper_frac*X_XGB.shape[0])]
    else:
        X_XGB_sample = X_XGB.sample(frac=hyper_frac, random_state=99)
        Y_XGB_sample = Y_XGB.sample(frac=hyper_frac, random_state=99)
    #########  Now set the number of rows we need to tune hyper params ###
    nums = int(X_XGB_sample.shape[0]*0.9)
    X_train = X_XGB_sample[:nums]
    X_valid = X_XGB_sample[nums:]
    Y_train = Y_XGB_sample[:nums]
    Y_valid = Y_XGB_sample[nums:]
    scoreFunction = { "precision": "precision_weighted","recall": "recall_weighted"}
    params = {
        'learning_rate': sp.stats.uniform(scale=1),
        'gamma': sp.stats.randint(0, 32),
       'n_estimators': sp.stats.randint(100,500),
        "max_depth": sp.stats.randint(3, 15),
            },
    model = RandomizedSearchCV(xgb.set_params(**param),
                                       param_distributions = params,
                                       n_iter = 10,
                                       return_train_score = True,
                                       random_state = 99,
                                       n_jobs=-1,
                                       #cv = 3,
                                        verbose = False)

    X_train, X_valid = data_transform(X_train, Y_train, X_valid,
                                scaler=scaler, enc_method=enc_method)

    gbm_model = xgb_model_fit(model, X_train, Y_train, X_valid, Y_valid, modeltype,
                         log_y, params, cpu_params)
    model = gbm_model.best_estimator_
    #############################################################################
    n_splits = 10
    ls=[]
    if modeltype == 'Regression':
        fold = KFold(n_splits=n_splits)
    else:
        fold = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=99)
    scores=[]
    if not isinstance(X_XGB_test, str):
        pred_xgbs = np.zeros(len(X_XGB_test))
        pred_probas = np.zeros(len(X_XGB_test))
    else:
        pred_xgbs = []
        pred_probas = []
    #### First convert test data into numeric using train data ###
    if not isinstance(X_XGB_test, str):
        _, X_XGB_test_enc = data_transform(X_XGB, Y_XGB, X_XGB_test,
                                scaler=scaler, enc_method=enc_method)
    #### now run all the folds each one by one ##################################
    start_time = time.time()
    for folds, (train_index, test_index) in tqdm(enumerate(fold.split(X_XGB,Y_XGB))):
        x_train, x_test = X_XGB.iloc[train_index], X_XGB.iloc[test_index]
        if modeltype == 'Regression':
            if log_y:
                y_train, y_test = np.log(Y_XGB.iloc[train_index]), Y_XGB.iloc[test_index]
            else:
                y_train, y_test = Y_XGB.iloc[train_index], Y_XGB.iloc[test_index]
        else:
            y_train, y_test = Y_XGB.iloc[train_index], Y_XGB.iloc[test_index]

        ##  scale the x_train and x_test values - use all columns -
        x_train, x_test = data_transform(x_train, y_train, x_test,
                                scaler=scaler, enc_method=enc_method)

        model = gbm_model.best_estimator_
        model = xgb_model_fit(model, x_train, y_train, x_test, y_test, modeltype,
                                log_y, params, cpu_params)

        #### now make predictions on validation data ##
        if modeltype == 'Regression':
            if log_y:
                preds = np.exp(model.predict(x_test))
            else:
                preds = model.predict(x_test)
        else:
            preds = model.predict(x_test)

        feature_importances = pd.DataFrame(model.feature_importances_,
                                           index = X_XGB.columns,
                                            columns=['importance'])
        sum_all=feature_importances.values
        ls.append(sum_all)
        ######  Time to consolidate the predictions on test data #########
        if modeltype == 'Regression':
            if not isinstance(X_XGB_test, str):
                if log_y:
                    pred_xgb=np.exp(model.predict(X_XGB_test_enc[columns]))
                else:
                    pred_xgb=model.predict(X_XGB_test_enc[columns])
                pred_xgbs = np.vstack([pred_xgbs, pred_xgb])
                pred_xgbs = pred_xgbs.mean(axis=0)
            if log_y:
                score = np.sqrt(mean_squared_log_error(y_test, preds))
            else:
                score = np.sqrt(mean_squared_error(y_test, preds))
            print('RMSE score in fold %d = %s' %(folds+1, score))
        else:
            if not isinstance(X_XGB_test, str):
                pred_xgb=model.predict(X_XGB_test_enc[columns])
                pred_proba = model.predict_proba(X_XGB_test_enc[columns])
                if folds == 0:
                    pred_xgbs = copy.deepcopy(pred_xgb)
                    pred_probas = copy.deepcopy(pred_proba)
                else:
                    pred_xgbs = np.vstack([pred_xgbs, pred_xgb])
                    pred_xgbs = stats.mode(pred_xgbs, axis=0)[0][0]
                    pred_probas = np.mean( np.array([ pred_probas, pred_proba ]), axis=0 )
            score = balanced_accuracy_score(y_test, preds)
            print('Balanced Accuracy score in fold %d = %0.1f%%' %(folds+1, score*100))
        scores.append(score)
    print('    Time taken for training XGB (in minutes) = %0.1f' %(
             (time.time()-start_time)/60))
    if verbose:
        plot_importances_XGB(train_set=X_XGB, labels=Y_XGB, ls=ls, y_preds=pred_xgbs,
                            modeltype=modeltype)
    print("Average scores are: ", np.sum(scores)/len(scores))
    print('\nReturning the following:')
    print('    Model = %s' %model)
    if modeltype == 'Regression':
        print('    final predictions', pred_xgbs[:10])
        return (pred_xgbs, model)
    else:
        print('    final predictions', pred_xgbs[:10])
        print('    predicted probabilities', pred_probas[:1])
        return (pred_xgbs, pred_probas, model)
##################################################################################
def plot_importances_XGB(train_set, labels, ls, y_preds, modeltype):
    add_items=0
    for item in ls:
        add_items +=item
    df_cv=pd.DataFrame(add_items/len(ls),index=train_set.columns,columns=["importance"]).sort_values('importance', ascending=False)
    df_cv=df_cv.reset_index()
    feat_imp = pd.Series(df_cv.importance.values, index=df_cv.drop(["importance"], axis=1)).sort_values(axis='index',ascending=False)
    #
    feat_imp2=feat_imp[feat_imp>0.00005]
    imp_columns=[]
    for item in pd.DataFrame(feat_imp2).reset_index()["index"].tolist():
        fcols=re.sub("[(),]","",str(item))
        try:
            columns= int(re.sub("['']","",fcols))
            imp_columns.append(columns)
        except:
            columns= re.sub("['']","",fcols)
            imp_columns.append(columns)
    # X_UPDATED=X_GB[imp_columns]
    len(imp_columns)
    fig = plt.figure(figsize=(15,8))
    ax1=plt.subplot(2, 2, 1)
    feat_imp.nlargest(5).plot(kind='barh', ax=ax1)
    if modeltype == 'Regression':
        ax2=plt.subplot(2, 2, 2)
        pd.Series(y_preds).plot(ax=ax2, color='b');
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
def FE_create_interaction_vars(df, intxn_vars):
    """
    This handy function creates interaction variables among pairs of numeric vars you send in.
    Your input must be a dataframe and a list of tuples. Each tuple must contain a pair of variables.
    All variables must be numeric. Double check your input before sending them in.
    """
    df = df.copy(deep=True)
    for (each_intxn1,each_intxn2)  in intxn_vars:
        new_col = each_intxn1 + '_x_' + each_intxn2
        try:
            df[new_col] = df[each_intxn1] * df[each_intxn2]
        except:
            continue
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
            max_col_length = df[col].map(len).max()
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
    float_targets = 'float16' in  train_dataframe[copy_targets].dtypes.values or 'float32' in  train_dataframe[copy_targets].dtypes.values
    if float_targets:
        modeltype = 'Regression'
    else:
        modeltype = 'Classification'
    print('    Since number of rows > maxrows, loading a random sample of %d rows into pandas for EDA' %nrows_limit)
    ####### If it is a classification problem, you need to stratify and select sample ###
    if modeltype != 'Regression':
        copy_targets = copy.deepcopy(targets)
        for each_target in copy_targets:
            ### You need to remove rows that have very class samples - that is a problem while splitting train_small
            list_of_few_classes = train_dataframe[each_target].value_counts()[train_dataframe[each_target].value_counts()<=10].index.tolist()
            train_small = train_dataframe.loc[~(train_dataframe[each_target].isin(list_of_few_classes))]
        train_small, _ = train_test_split(train_dataframe, test_size=test_size, stratify=train_dataframe[targets])
    else:
        ### For Regression problems: load a small sample of data into a pandas dataframe ##
        train_small = train_dataframe.sample(n=nrows_limit, random_state=99)
    return train_small
################################################################################################