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
###############           F E A T U R E   W I Z A R D          ##################
################  featurewiz library developed by Ram Seshadri  #################
# featurewiz utilizes SULOV METHOD which is a fast method for feature selection #
#####  SULOV also means Searching for Uncorrelated List Of Variables (:-)  ######
###############     A L L   R I G H T S  R E S E R V E D         ################
#################################################################################
##### This project is not an official Google project. It is not supported by ####
##### Google and Google specifically disclaims all warranties as to its quality,#
##### merchantability, or fitness for a particular purpose.  ####################
#################################################################################
import pandas as pd
import numpy as np
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
####################################################################################
import re
import pdb
import pprint
from itertools import cycle, combinations
from collections import defaultdict, OrderedDict
import copy
import time
import sys
import random
import xlrd
import statsmodels
from io import BytesIO
import base64
from functools import reduce
import copy
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
        print('        %d variable(s) removed since they were ID or low-information variables'
                                %len(IDcols+cols_delete+discrete_string_vars))
        if verbose >= 1:
            print('    List of variables removed: %s' %(IDcols+cols_delete+discrete_string_vars))
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
    features_dict = dict([('IDcols',IDcols),('cols_delete',cols_delete),('bool_vars',bool_vars),('categorical_vars',categorical_vars),
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
from collections import Counter
import time
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
##################################################################################
def load_file_dataframe(dataname, sep=",", header=0, verbose=0):
    start_time = time.time()
    ###########################  This is where we load file or data frame ###############
    if isinstance(dataname,str):
        #### this means they have given file name as a string to load the file #####
        if dataname != '' and dataname.endswith(('csv')):
            codex = ['utf-8', 'iso-8859-1', 'cp1252', 'latin1']
            for code in codex:
                try:
                    dfte = pd.read_csv(dataname,sep=sep,index_col=None,encoding=code)
                    print('    Encoder %s chosen to read CSV file' %code)
                    print('Shape of your Data Set loaded: %s' %(dfte.shape,))
                    if len(np.array(list(dfte))[dfte.columns.duplicated()]) > 0:
                        print('You have duplicate column names in your data set. Removing duplicate columns now...')
                        dfte = dfte[list(dfte.columns[~dfte.columns.duplicated(keep='first')])]
                    return dfte
                except:
                    print('Encoding codex %s does not work for this file' %code)
                    continue
        elif dataname.endswith(('xlsx','xls','txt')):
            #### It's very important to get header rows in Excel since people put headers anywhere in Excel#
            dfte = pd.read_excel(dataname,header=header)
            print('Shape of your Data Set loaded: %s' %(dfte.shape,))
            return dfte
        else:
            print('File not able to be loaded')
            return
    if isinstance(dataname,pd.DataFrame):
        #### this means they have given a dataframe name to use directly in processing #####
        dfte = copy.deepcopy(dataname)
        print('Shape of your Data Set loaded: %s' %(dfte.shape,))
        if len(np.array(list(dfte))[dfte.columns.duplicated()]) > 0:
            print('You have duplicate column names in your data set. Removing duplicate columns now...')
            dfte = dfte[list(dfte.columns[~dfte.columns.duplicated(keep='first')])]
        return dfte
    else:
        print('Dataname input must be a filename with path to that file or a Dataframe')
        return
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
    elif  train[targ].dtype in ['float']:
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
def remove_variables_using_fast_correlation(df, numvars, modeltype, target,
                                corr_limit = 0.70,verbose=0):
    """
    ##########################################################################################
    #####              SULOV stands for Searching Uncorrelated List Of Variables  ############
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
    ##############  YOU MUST INCLUDE THE ABOVE MESSAGE IF YOU COPY THIS CODE IN YOUR LIBRARY #####
    """
    import copy
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
        try:
            fs.fit(df[corr_list].astype(np.float16), df[target])
        except:
            fs.fit(df[corr_list].astype(np.float32), df[target])
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
        corr = df[corr_list].corr()
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
        import copy
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
def convert_train_test_cat_col_to_numeric(start_train, start_test, col):
    """
    ####  This is the easiest way to label encode object variables in both train and test
    #### This takes care of some categories that are present in train and not in test
    ###     and vice versa
    """
    start_train = copy.deepcopy(start_train)
    start_test = copy.deepcopy(start_test)
    if start_train[col].isnull().sum() > 0:
        start_train[col] = start_train[col].fillna("NA")
    train_categs = list(pd.unique(start_train[col].values))
    if not isinstance(start_test,str) :
        test_categs = list(pd.unique(start_test[col].values))
        categs_all = train_categs+test_categs
        dict_all =  return_factorized_dict(categs_all)
    else:
        dict_all = return_factorized_dict(train_categs)
    start_train[col] = start_train[col].map(dict_all)
    if not isinstance(start_test,str) :
        if start_test[col].isnull().sum() > 0:
            start_test[col] = start_test[col].fillna("NA")
        start_test[col] = start_test[col].map(dict_all)
    return start_train, start_test
###############################################################################
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
############## CONVERSION OF STRING COLUMNS TO NUMERIC WITHOUT LABEL ENCODER #########
#######################################################################################
import copy
import pdb
def convert_a_column_to_numeric(x, col_dict=""):
    '''Function converts any pandas series (or column) consisting of string chars,
       into numeric values. It converts an all-string column to an all-number column.
       This is an amazing function which performs exactly like a Label Encoding
       except that it is simpler and faster'''
    if isinstance(col_dict, str):
        values = np.unique(x)
        values2nums = dict(zip(values,range(len(values))))
        convert_dict = dict(zip(range(len(values)),values))
        return x.replace(values2nums), convert_dict
    else:
        convert_dict = copy.deepcopy(col_dict)
        keys  = col_dict.keys()
        newkeys = np.unique(x)
        rem_keys = left_subtract(newkeys, keys)
        max_val = max(col_dict.values()) + 1
        for eachkey in rem_keys:
            convert_dict.update({eachkey:max_val})
            max_val += 1
        return x.replace(convert_dict)
#######################################################################################
def convert_a_mixed_object_column_to_numeric(x, col_dict=''):
    """
    This is the main utility that converts any string column to numeric.
    It does not need Label Encoder since it picks up an string that may not be in test data.
    """
    x = x.astype(str)
    if isinstance(col_dict, str):
        x, convert_dict = convert_a_column_to_numeric(x)
        convert_dict = dict([(v,k) for (k,v) in convert_dict.items()])
        return x, convert_dict
    else:
        x = convert_a_column_to_numeric(x, col_dict)
        return x, ''
######################################################################################
def convert_all_object_columns_to_numeric(train, test=""):
    """
    #######################################################################################
    This is a utility that converts string columns to numeric WITHOUT LABEL ENCODER.
    Make sure test and train have the same number of columns. If you have target in train,
    remove it before sending it through this utility. Otherwise, might blow up during test transform.
    The beauty of this utility is that it does not blow up when it finds strings in test not in train.
    #######################################################################################
    """
    train = copy.deepcopy(train)
    lis = []
    lis = train.select_dtypes('object').columns.tolist() + train.select_dtypes('category').columns.tolist()
    if not isinstance(test, str):
        lis_test = test.select_dtypes('object').columns.tolist() + test.select_dtypes('category').columns.tolist()
        if len(left_subtract(lis, lis_test)) > 0:
            ### if there is an extra column in train that is not in test, then remove it from consideration
            lis = copy.deepcopy(lis_test)
    if not (len(lis)==0):
        for everycol in lis:
            #print('    Converting %s to numeric' %everycol)
            try:
                train[everycol], train_dict = convert_a_mixed_object_column_to_numeric(train[everycol])
                if not isinstance(test, str):
                    test[everycol],_ = convert_a_mixed_object_column_to_numeric(test[everycol], train_dict)
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
                      test_data='', feature_engg='', category_encoders='', **kwargs):
    """
    This is a fast utility that uses XGB to find top features. You
    It returns a list of important features.
    Since it is XGB, you dont have to restrict the input to just numeric vars.
    You can send in all kinds of vars and it will take care of transforming it. Sweet!
    #####           Featurewiz arguments           #############################
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
            Recommend you do not use more than two of these. Featurewiz will automatically select only two from your list.
            Default is empty string (which means no encoding of your categorical features)
                ['HashingEncoder', 'SumEncoder', 'PolynomialEncoder', 'BackwardDifferenceEncoder',
                'OneHotEncoder', 'HelmertEncoder', 'OrdinalEncoder', 'FrequencyEncoder', 'BaseNEncoder',
                'TargetEncoder', 'CatBoostEncoder', 'WOEEncoder', 'JamesSteinEncoder']
        ############       C A V E A T !  C A U T I O N !   W A R N I N G ! #######################################
        In general: Be very careful with featurewiz. Don't use it mindlessly to generate un-interpretable features!
    ########           Featurewiz Output           #############################
    Output: Tuple
    Featurewiz can output either a list of features or one dataframe or two depending on what you send in.
        1. features: featurewiz will return just a list of important features in your data if you send in just a dataset.
        2. trainm: modified train dataframe is the dataframe that is modified with engineered and selected features from dataname.
        3. testm: modified test dataframe is the dataframe that is modified with engineered and selected features from test_data
    """
    ### set all the defaults here ##############################################
    dataname = copy.deepcopy(dataname)
    max_nums = 30
    max_cats = 15
    RANDOM_SEED = 42
    ############################################################################
    cat_encoders_list = list(settings.cat_encoders_names.keys())
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
    ##################    L O A D     D A T A N A M E   ########################
    train = load_file_dataframe(dataname, sep, header, verbose)
    train_index = train.index
    if isinstance(test_data, str):
        if test_data != '':  ### if test data is not an empty string, then load it as a file
            test = load_file_dataframe(test_data, sep, header, verbose)
            test_index = test.index
        else:
            test = None ### set test as None so that it can be skipped processing
    elif isinstance(test_data, pd.DataFrame): ### If it is a dataframe, simply copy it to test
        test = copy.deepcopy(test_data)
        test_index = test.index
    else:
        test = None ### If it is none of the above, set test as None
    #############    X G B O O S T     F E A T U R E    S E L E C T I O N ######
    #### If there are more than 30 categorical variables in a data set, it is worth reducing features.
    ####  Otherwise. XGBoost is pretty good at finding the best features whether cat or numeric !
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
    features_dict = classify_features(train, target)
    if len(features_dict['date_vars']) > 0:
        date_time_vars = features_dict['date_vars']
        date_cols = copy.deepcopy(date_time_vars)
        #### Do this only if date time columns exist in your data set!
        for date_col in date_cols:
            print('Processing %s column for date time features....' %date_col)
            date_df_train = fe_create_time_series_features(train, date_col)
            date_col_adds_train = left_subtract(date_df_train.columns.tolist(),date_col)
            print('    Adding %d column(s) from date-time column %s in train' %(len(date_col_adds_train),date_col))
            train.drop(date_col,axis=1,inplace=True)
            train = train.join(date_df_train)
            if test is not None:
                print('        Adding same time series features to test data...')
                date_df_test = fe_create_time_series_features(test, date_col)
                date_col_adds_test = left_subtract(date_df_test.columns.tolist(),date_col)
                ### Now time to remove the date time column from all further processing ##
                test.drop(date_col,axis=1,inplace=True)
                test = test.join(date_df_test)
    ### Now time to continue with our further processing ##
    cols_to_remove = features_dict['cols_delete'] + features_dict['IDcols'] + features_dict['discrete_string_vars']
    preds = [x for x in list(train) if x not in target+cols_to_remove]
    numvars = train[preds].select_dtypes(include = 'number').columns.tolist()
    if len(numvars) > max_nums:
        if feature_gen:
            print('    Warning: Too many extra features will be generated by feature generation. This may take time...')
    catvars = left_subtract(preds, numvars)
    if len(catvars) > max_cats:
        if feature_type:
            print('    Warning: Too many extra features will be generated by category encoding. This may take time...')
    rem_vars = copy.deepcopy(catvars)
    ########## Now we need to select the right model to run repeatedly ####
    if target is None or len(target) == 0:
        cols_list = list(train)
        settings.modeltype = 'Clustering'
    else:
        settings.modeltype = analyze_problem_type(train, target)
        cols_list = left_subtract(list(train),target)
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
        print('Starting feature engineering...this will take time...')
        if test is None:
            if settings.multi_label:
                ### if it is a multi_label problem, leave target as it is - a list!
                X_train, X_test, y_train, y_test = train_test_split(train[preds],
                                                                train[target],
                                                                test_size=0.2,
                                                                random_state=RANDOM_SEED)
            else:
                ### if it not a multi_label problem, make target as target[0]
                X_train, X_test, y_train, y_test = train_test_split(train[preds],
                                                            train[target[0]],
                                                            test_size=0.2,
                                                            random_state=RANDOM_SEED)
        else:
            X_train = train[preds]
            if settings.multi_label:
                y_train = train[target]
            else:
                y_train = train[target[0]]
            X_test = test[preds]
            try:
                y_test = test[target]
            except:
                y_test = None
        X_train_index = X_train.index
        X_test_index = X_test.index
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
                    random_state=RANDOM_SEED)
        #### Now you can process the tuple this way #########
        data1 = data_tuple.X_train.join(y_train) ### data_tuple does not have a y_train, remember!
        if test is None:
            ### Since you have done a train_test_split using randomized split, you need to put it back again.
            data2 = data_tuple.X_test.join(y_test)
            train = data1.append(data2)
            train = train.reindex(train_index)
        else:
            try:
                test = data_tuple.X_test.join(y_test)
            except:
                test = copy.deepcopy(data_tuple.X_test)
            test = test.reindex(test_index)
            train = copy.deepcopy(data1)
        print('    Completed feature engineering. Shape of Train (with target) = %s' %(train.shape,))
        preds = [x for x in list(train) if x not in target]
        numvars = train[preds].select_dtypes(include = 'number').columns.tolist()
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
    ########    Drop Missing value rows since XGB for some reason  #########
    ########    can't handle missing values in early stopping rounds #######
    train.dropna(axis=0,subset=preds+target,inplace=True)
    if len(numvars) > 1:
        final_list = remove_variables_using_fast_correlation(train,numvars,settings.modeltype,target,
                         corr_limit,verbose)
    else:
        final_list = copy.deepcopy(numvars)
    ####### This is where you draw how featurewiz works when the verbose = 2 ###########
    print('    Adding %s categorical variables to reduced numeric variables  of %d' %(
                            len(important_cats),len(final_list)))
    if  isinstance(final_list,np.ndarray):
        final_list = final_list.tolist()
    preds = final_list+important_cats
    #######You must convert category variables into integers ###############
    if len(important_cats) > 0:
        train, traindict = convert_all_object_columns_to_numeric(train,  "")
        if test is not None:
            test, _ = convert_all_object_columns_to_numeric(test, traindict)
    ########   Dont move this train and y definition anywhere else ########
    y = train[target]
    print('############## F E A T U R E   S E L E C T I O N  ####################')
    important_features = []
    ########## This is for Single_Label problems ######################
    if settings.modeltype == 'Regression':
        objective = 'reg:squarederror'
        model_xgb = XGBRegressor( n_estimators=100,subsample=subsample,objective=objective,
                                colsample_bytree=col_sub_sample,reg_alpha=0.5, reg_lambda=0.5,
                                 seed=1,n_jobs=-1,random_state=1)
        eval_metric = 'rmse'
    else:
        #### This is for Classifiers only
        classes = np.unique(train[target].values)
        if len(classes) == 2:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                colsample_bytree=col_sub_sample,gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=100,
                n_jobs=-1, nthread=None, objective='binary:logistic',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5,
                seed=1)
            eval_metric = 'logloss'
        else:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                        colsample_bytree=col_sub_sample, gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=100,
                n_jobs=-1, nthread=None, objective='multi:softmax',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5,
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
                    important_features += pd.Series(model_xgb.get_booster().get_score(
                            importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                #######  order this in the same order in which they were collected ######
                important_features = list(OrderedDict.fromkeys(important_features))
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
                    important_features += pd.Series(model_xgb.get_booster().get_score(
                            importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                important_features = list(OrderedDict.fromkeys(important_features))
    except:
        print('Finding top features using XGB is crashing. Continuing with all predictors...')
        important_features = copy.deepcopy(preds)
        return important_features, train[important_features+target]
    important_features = list(OrderedDict.fromkeys(important_features))
    print('Selected %d important features from your dataset' %len(important_features))
    numvars = [x for x in numvars if x in important_features]
    important_cats = [x for x in important_cats if x in important_features]
    print('    Time taken (in seconds) = %0.0f' %(time.time()-start_time))
    if feature_gen or feature_type:
        print(f'Returning list of {len(important_features)} important features and dataframe.')
        if test is None:
            return important_features, train[important_features+target]
        else:
            print('Returning 2 dataframes: train and test with %d important features.' %len(important_features))
            if target in list(test): ### see if target exists in this test data
                return train[important_features+target], test[important_features+target]
            else:
                return train[important_features+target], test[important_features]
    else:
        print(f'Returning list of {len(important_features)} important features and dataframe.')
        return important_features, train[important_features+target]
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
    try:
        from tensorflow.python.client import device_lib
        dev_list = device_lib.list_local_devices()
        print('Number of GPUs = %d' %len(dev_list))
        for i in range(len(dev_list)):
            if 'GPU' == dev_list[i].device_type:
                GPU_exists = True
                print('%s available' %dev_list[i].device_type)
    except:
        pass
    if not GPU_exists:
        try:
            os.environ['NVIDIA_VISIBLE_DEVICES']
            print('    GPU active on this device')
            return True
        except:
            print('    No GPU active on this device')
            return False
    else:
        return True
#############################################################################################
from itertools import combinations
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
######################################################################################
# Removes duplicates from a list to return unique values - USED ONLYONCE
def find_remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
################################################################################
def add_date_time_features(smalldf, startTime, endTime, splitter_date_string="/",splitter_hour_string=":"):
    """
    If you have start date time stamp and end date time stamp, this module will create additional features for such fields.
    You must provide a start date time stamp field and if you have an end date time stamp field, you must use it.
    Otherwise, you are better off using the create_date_time_features module which is also in this library.
    You must provide the following:
    smalldf: Dataframe containing your date time fields
    startTime: this is hopefully a string field which converts to a date time stamp easily. Make sure it is a string.
    endTime: this also must be a string field which converts to a date time stamp easily. Make sure it is a string.
    splitter_date_string: usually there is a string such as '/' or '.' between day/month/year etc. Default is assumed / here.
    splitter_hour_string: usually there is a string such as ':' or '.' between hour:min:sec etc. Default is assumed : here.
    """
    smalldf = smalldf.copy()
    add_cols = []
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
def split_one_field_into_many(df, field, splitter, filler, new_names_list, add_count_field=False):
    """
    This little function takes any data frame field (string variables only) and splits
    it into as many fields as you want in the new_names_list.
    You can also specify what string to split on using the splitter argument.
    You can also fill Null values that occur due to your splitting by specifying a filler.
    if no new_names_list is given, then we use the name of the field itself to split.
    add_count_field: False (default). If True, it will count the number of items in
        the "field" column before the split. This may be needed in nested dictionary fields.
    """
    import warnings
    warnings.filterwarnings("ignore")
    df = df.copy()
    ### First print the maximum number of things in that field
    max_things = df[field].map(lambda x: len(x.split(splitter))).max()
    if len(new_names_list) == 0:
        print('    Max. columns created by splitting %s field is %d.' %(
                            field,max_things))
    else:
        if not max_things == len(new_names_list):
            print('    Max. columns created by splitting %s field is %d but you have given %d variable names only. Selecting first %d' %(
                        field,max_things,len(new_names_list),len(new_names_list)))
    ### This creates a new field that counts the number of things that are in that field.
    if add_count_field:
        num_products_viewed = 'count_things_in_'+field
        df[num_products_viewed] = df[field].map(lambda x: len(x.split(";"))).values
    ### Clean up the field such that it has the right number of split chars otherwise add to it
    df[field] = df[field].map(lambda x: x+splitter*(max_things-len(x.split(";"))) if len(x.split(";")) < max_things else x)
    ###### Now you create new fields by split the one large field ########
    if new_names_list == '':
        new_names_list = [field+'_'+str(i) for i in range(1,max_things+1)]
    try:
        for i in range(len(new_names_list)):
            df[field].fillna(filler, inplace=True)
            df.loc[df[field] == splitter, field] = filler
            df[new_names_list[i]] = df[field].map(lambda x: x.split(splitter)[i]
                                          if splitter in x else x)
    except:
        ### Check if the column is a string column. If not, give an error message.
        print('Cannot split the column. Getting an error. Check the column again')
        return df
    return df, new_names_list
###########################################################################
def add_aggregate_primitive_features(dft, agg_types, id_column, ignore_variables=[]):
    """
    ###   Modify Dataframe by adding computational primitive Features using Feature Tools ####
    ###   What are aggregate primitives? they are to "mean""median","mode","min","max", etc. features
    ### Inputs:
    ###   df: Just sent in the data frame df that you want features added to
    ###   agg_types: list of computational types: 'mean','median','count', 'max', 'min', 'sum', etc.
    ###         One caveat: these agg_types must be found in the agg_func of numpy or pandas groupby statement.
    ###         for example: numpy has 'median','prod','sum','std','var', etc. - they will work!
    ###   idcolumn: this is to create an index for the dataframe since FT runs on index variable. You can leave it empty string.
    ###   ignore_variables: list of variables to ignore among numeric variables in data since they may be ID variables.
    """
    import copy
    ### Make sure the list of functions they send in are acceptable functions. If not, the aggregate will blow up!
    func_set = {'count','sum','mean','mad','median','min','max','mode','abs','prod','std','var','sem','skew','kurt','quantile','cumsum','cumprod','cummax','cummin'}
    agg_types = list(set(agg_types).intersection(func_set))
    ### If the ignore_variables list is empty, make sure you add the id_column to it so it can be dropped from aggregation.
    if len(ignore_variables) == 0:
        ignore_variables = [id_column]
    ### Select only integer and float variables to do this aggregation on. Be very careful if there are too many vars.
    ### This will take time to run in that case.
    dft_index = copy.deepcopy(dft[id_column])
    dft_cont = copy.deepcopy(dft.select_dtypes('number').drop(ignore_variables,axis=1))
    dft_cont[id_column] = dft_index
    try:
        dft_full = dft_cont.groupby(id_column).agg(agg_types)
    except:
        ### if for some reason, the groupby blows up, then just return the dataframe as is - no changes!
        return dft
    cols = [x+'_'+y+'_by_'+id_column for (x,y) in dft_full.columns]
    dft_full.columns = cols
    ###  Not every column has useful values. If it is full of just the same value, remove it
    _, list_unique_col_ids = np.unique(dft_full, axis = 1, return_index=True)
    dft_full = dft_full.iloc[:, list_unique_col_ids]
    return dft_full
################################################################################################################################
import copy
##############################################################
def create_ts_features(df, tscol):
    """
    This takes in input a dataframe and a date variable.
    It then creates time series features using the pandas .dt.weekday kind of syntax.
    It also returns the data frame of added features with each variable as an integer variable.
    """
    df = copy.deepcopy(df)
    dt_adds = []
    try:
        df[tscol+'_hour'] = df[tscol].dt.hour.astype(int)
        df[tscol+'_minute'] = df[tscol].dt.minute.astype(int)
        dt_adds.append(tscol+'_hour')
        dt_adds.append(tscol+'_minute')
    except:
        print('    Error in creating hour-second derived features. Continuing...')
    try:
        df[tscol+'_dayofweek'] = df[tscol].dt.dayofweek.astype(int)
        dt_adds.append(tscol+'_dayofweek')
        df[tscol+'_quarter'] = df[tscol].dt.quarter.astype(int)
        dt_adds.append(tscol+'_quarter')
        df[tscol+'_month'] = df[tscol].dt.month.astype(int)
        dt_adds.append(tscol+'_month')
        df[tscol+'_year'] = df[tscol].dt.year.astype(int)
        dt_adds.append(tscol+'_year')
        today = date.today()
        df[tscol+'_age_in_years'] = today.year - df[tscol].dt.year.astype(int)
        dt_adds.append(tscol+'_age_in_years')
        df[tscol+'_dayofyear'] = df[tscol].dt.dayofyear.astype(int)
        dt_adds.append(tscol+'_dayofyear')
        df[tscol+'_dayofmonth'] = df[tscol].dt.day.astype(int)
        dt_adds.append(tscol+'_dayofmonth')
        df[tscol+'_weekofyear'] = df[tscol].dt.weekofyear.astype(int)
        dt_adds.append(tscol+'_weekofyear')
        weekends = (df[tscol+'_dayofweek'] == 5) | (df[tscol+'_dayofweek'] == 6)
        df[tscol+'_weekend'] = 0
        df.loc[weekends, tscol+'_weekend'] = 1
        df[tscol+'_weekend'] = df[tscol+'_weekend'].astype(int)
        dt_adds.append(tscol+'_weekend')
    except:
        print('    Error in creating date time derived features. Continuing...')
    df = df[dt_adds].fillna(0).astype(int)
    return df
################################################################
from dateutil.relativedelta import relativedelta
from datetime import date
##### This is a little utility that computes age from year ####
def compute_age(year_string):
    today = date.today()
    age = relativedelta(today, year_string)
    return age.years
#################################################################
def fe_create_time_series_features(dtf, ts_column):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    This creates between 8 and 10 date time features for each date variable. The number of features
    depends on whether it is just a year variable or a year+month+day and whether it has hours and mins+secs.
    So this can create all these features using just the date time column that you send in.
    It returns the entire dataframe with added variables as output.
    """
    dtf = copy.deepcopy(dtf)
    #### If for some reason ts_column is just a number, make sure it is a string so it does not blow up and concatenated
    if not isinstance(ts_column,str):
        ts_column = str(ts_column)
    try:
        ### In some extreme cases, date time vars are not processed yet and hence we must fill missing values here!
        if dtf[ts_column].isnull().sum() > 0:
            missing_flag = True
            new_missing_col = ts_column + '_Missing_Flag'
            dtf[new_missing_col] = 0
            dtf.loc[dtf[ts_column].isnull(),new_missing_col]=1
            dtf[ts_column] = dtf[ts_column].fillna(method='ffill')
        if dtf[ts_column].dtype in [np.float64,np.float32,np.float16]:
            dtf[ts_column] = dtf[ts_column].astype(int)
        ### if we have already found that it was a date time var, then leave it as it is. Thats good enough!
        date_items = dtf[ts_column].apply(str).apply(len).values
        #### In some extreme cases,
        if all(date_items[0] == item for item in date_items):
            if date_items[0] == 4:
                ### If it is just a year variable alone, you should leave it as just a year!
                age_col = ts_column+'_age_in_years'
                dtf[age_col] = dtf[ts_column].map(lambda x: pd.to_datetime(x,format='%Y')).apply(compute_age).values
                return dtf[[age_col]]
            else:
                ### if it is not a year alone, then convert it into a date time variable
                dtf[ts_column] = pd.to_datetime(dtf[ts_column], infer_datetime_format=True)
        else:
            dtf[ts_column] = pd.to_datetime(dtf[ts_column], infer_datetime_format=True)
        dtf = create_ts_features(dtf,ts_column)
    except:
        print('Error in Processing %s column for date time features. Continuing...' %ts_column)
    return dtf
######################################################################################
def fe_get_latest_status_from_date(dft, id_col, date_col, cols, ascending=False):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    This function gets you the latest status of the columns in cols from the date column you specify in dataframe, dft.
    ######################################################################################
    date_col: this must be a valid pandas date-time column. If it is a string, make sure you change it to a date-time column.
    cols: these are the list of columns you want their latest value based on the date-col you specify.
    ascending: you can set this as True or False depending on whether you want the smallest or the biggest
    ######################################################################################
    Returns a dataframe that is smaller than the original dataframe since it groups everything by ID column and date-col.
    It then sorts each group with the latest date on top and selects that top row. It then returns the cols.
    You should get a dataframe that has the cols specified in your input with fewer rows than your input dft.
    """
    dft = copy.deepcopy(dft)
    try:
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
def fe_split_add_column(dft, col, splitter=',', action='+'):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    This function will split a column's values based on a splitter you specify and
    will either add them or concatenate them as you specify in the action argument.
    ###########################################################################
    splitter: splitter can be any string that is found in your column and that you want to split by.
    action: can be anything, add, subtract, multiply, concatenate, whatever you specify.
    ################################################################################
    Returns a dataframe with a new column that is a modification of the old column
    """
    dft = copy.deepcopy(dft)
    new_col = col + '_split_apply'
    print('Creating column = %s using split_add feature engineering...' %new_col)
    if action in ['+','-','*','/','add','subtract','multiply','divide']:
        if action == 'add':
            sign = '+'
        elif action == 'subtract':
            sign = '-'
        elif action == 'multiply':
            sign = '*'
        elif action == 'divide':
            sign = '/'
        else:
            sign = '+'
        # using reduce to compute sum of list
        try:
            trainx = dft[col].map(lambda x:  0 if x is np.nan else 0 if x == '' else x.split(splitter)).map(
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
def fe_add_age_by_date_col(dft, date_col, age_format):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    This handy function gets you age from the date_col to today. It can be counted in months or years or days.
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
def fe_count_rows_for_all_columns_by_group(dft, id_col):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    This handy function gives you a count of all rows by groups based on id_col in your dataframe.
    Remember that it counts only non-null rows. Hence it is a different count than other count function.
    It returns a dataframe with id_col as the index and a bunch of new columns that give you counts of groups.
    """
    new_col = 'row_count_'
    groupby_columns =  [id_col]
    grouped_count = dft.groupby(groupby_columns).count().add_prefix(new_col)
    return grouped_count
#################################################################################
def fe_count_rows_by_group_incl_nulls(dft, id_col):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    This function gives you the count of all the rows including null rows in your data.
    It returns a dataframe with id_col as the index and the counts of rows (incl null rows) as a new column
    """
    new_col = 'row_count_incl_null_rows'
    groupby_columns =  [id_col]
    ### len gives you count of all the rows including null rows in your data
    grouped_len = dft.groupby(groupby_columns).apply(len)
    grouped_val = grouped_len.values
    grouped_len = pd.DataFrame(grouped_val, columns=[new_col],index=grouped_len.index)
    return grouped_len
#################################################################################
import copy
import time
import pdb
def fe_create_groupby_features(dft, groupby_columns, numeric_columns, agg_types):
    """
    FE means FEATURE ENGINEERING - That means this function will create new features
    Beware: this function will return a smaller dataframe than what you send in since it groups rows by keys.
    #########################################################################################################
    Function groups rows in a dft dataframe by the groupby_columns and returns multiple columns for the numeric column aggregated.
    Do not send in more than one column in the numeric column since beyond the first column it will be ignored!
    agg_type can be any numpy function such as mean, median, sum, count, etc.
    ##########################################################################################################
    Returns: a smaller dataframe with rows grouped by groupby_columns and aggregated for the numeric_column
    """
    start_time = time.time()
    grouped_sep = pd.DataFrame()
    print('Autoviml Feature Engineering: creating groupby features using %s' %groupby_columns)
    ##########  This is where we create new columns by each numeric column grouped by group-by columns given.
    if isinstance(numeric_columns, list):
        pass
    elif isinstance(numeric_column, str):
        numeric_columns = [numeric_columns]
    else:
        print('    Numeric column must be a string not a number Try again')
        return pd.DataFrame()
    grouped_list = pd.DataFrame()
    for iteration, numeric_column in zip(range(len(numeric_columns)),numeric_columns):
        grouped = dft.groupby(groupby_columns)[[numeric_column]]
        try:
            agg_type = agg_types[iteration]
        except:
            print('    No aggregation type given, hence mean is chosen by default')
            agg_type = 'mean'
        try:
            prefix = numeric_column + '_'
            if agg_type in ['Sum', 'sum']:
                grouped_agg = grouped.sum()
            elif agg_type in ['Mean', 'mean','Average','average']:
                grouped_agg = grouped.mean()
            elif agg_type in ['count', 'Count']:
                grouped_agg = grouped.count()
            elif agg_type in ['Median', 'median']:
                grouped_agg = grouped.median()
            elif agg_type in ['Maximum', 'maximum','max', 'Max']:
                ## maximum of the amounts
                grouped_agg = grouped.max()
            elif agg_type in ['Minimum', 'minimum','min', 'Min']:
                ## maximum of the amounts
                grouped_agg = grouped.min()
            else:
                grouped_agg = grouped.mean()
            grouped_sep = grouped_agg.unstack().add_prefix(prefix).fillna(0)
        except:
            print('    Error in creating groupby features...returning with null dataframe')
            grouped_sep = pd.DataFrame()
        if iteration == 0:
            grouped_list = copy.deepcopy(grouped_sep)
        else:
            grouped_list = pd.concat([grouped_list,grouped_sep],axis=1)
        print('    After grouped features added by %s, number of columns = %d' %(numeric_column, grouped_list.shape[1]))
    #### once everything is done, you can close it here
    print('Time taken for creation of groupby features (in seconds) = %0.0f' %(time.time()-start_time))
    try:
        grouped_list.columns = grouped_list.columns.get_level_values(1)
        grouped_list.columns.name = None ## make sure the name on columns is removed
        grouped_list = grouped_list.reset_index() ## make sure the ID column comes back
    except:
        print('   Error in setting column names. Please reset column names after this step...')
    return grouped_list
################################################################################
# Can we see if a feature or features has some outliers and how can we cap them?
from collections import Counter
def FE_capping_outliers_beyond_IQR_Range(df, features, cap_at_nth_largest=5, IQR_multiplier=1.5, drop=False):
    """
    FE at the beginning of function name stands for Feature Engineering. FE functions add or drop features.
    #########################################################################################
    Typically we think of outliers as being observations beyond the 1.5 Inter Quartile Range (IQR)
    But this function will allow you to cap any observation that is multiple of that range, such as 1.5, 2. etc.
    In addition, this utility helps you select the number to cap it at. The value to be capped at
    is based on "n" that you input. It selects the nth_largest number after the maximum value to cap at!
    "cap_at_nth_largest" specifies the max number below the largest (max) number in your column to cap that feature.
    Optionally, you can drop certain observations that have too many outliers in at least 3 columns.
    #########################################################################################
    It returns the same dataframe as you input unless you change drop to True in the input argument.
    If drop=True, it will return a smaller number of rows in your dataframe than what you sent in. Be careful!
    """
    outlier_indices = []
    df = df.copy(deep=True)
    # iterate over features(columns)
    for col in features:
        ### this is how the column looks now before capping outliers
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
        df.drop(multiple_outliers, axis=0, inplace=True)
        print('Shape of dataframe after outliers being dropped: %s' %(df.shape,))
        print('\nNumber_of_rows with multiple outliers in more than 3 columns which were dropped = %d' %(number_of_rows-df.shape[0]))
    return df
#################################################################################
from sklearn.base import TransformerMixin
from collections import defaultdict
import pdb
class My_LabelEncoder(TransformerMixin):
    """
    ######  Label Encode any column in a data frame using this neat function #######################
    This not only Label Encodes a column but also accounts for NaN's and unknown values.
    Returns the same column label encoded as well as a dictionary mapping previous and new values.
    ################################################################################################
    """
    def __init__(self):
        self.transformer = defaultdict(str)
        self.reverse_transformer = defaultdict(str)

    def fit(self,testx):
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            return testx
        outs = np.unique(testx.factorize()[0])
        ins = np.unique(testx.factorize()[1]).tolist()
        if -1 in outs:
            ins.insert(0,np.nan)
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
        ins = np.unique(testx.factorize()[1]).tolist()
        missing = [x for x in ins if x not in self.transformer.keys()]
        if len(missing) > 0:
            for each_missing in missing:
                max_val = np.max(list(self.transformer.values())) + 1
                self.transformer[each_missing] = max_val
                self.inverse_transformer[max_val] = each_missing
        ### now convert the input to transformer dictionary values
        outs = testx.map(self.transformer).values
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
            if df1[each_cat].map(len).mean() >=20:
                nlpcols.append(each_cat)
                catcols.remove(each_cat)
        except:
            continue
    intcols = df1.select_dtypes(include='integer').columns.tolist()
    # let's find all the float numeric columns in data
    floatcols = df1.select_dtypes(include='float').columns.tolist()
    return catcols, intcols, floatcols, nlpcols
############################################################################################
def EDA_classify_features(train, target, idcols):
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
