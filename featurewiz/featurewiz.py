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
####  C O D E  C A N  B E  RE-U S E D  W I T H   C I T A T I O N   B E L O W ####
#################################################################################
###############              F E A T U R E   W I Z                ###############
################  featurewiz library developed by Ram Seshadri  #################
#### THIS METHOD IS KNOWN AS SULOV METHOD in HONOR OF my mom, SULOCHANA #########
#####       SULOV means Searching for Uncorrelated List Of Variables  ###########
###############                  v 0.0.1                         ################
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
        print('        %d variables removed since they were ID or low-information variables'
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
        print('        This does not include the Target column(s)')
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
                    print('Encoder %s chosen to read CSV file' %code)
                    print('Shape of your Data Set loaded: %s' %(dfte.shape,))
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
        print('''################ %s %s Feature Selection Started #####################''' %(
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
    #### THIS METHOD IS KNOWN AS SULOV METHOD in HONOR OF my mother SULOCHANA SESHADRI #######
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
                                labels = dict(zip(nodelist,[x+' (selected)' if x in selected+' (removed)' else x for x in nodelist])),
                                font_color='black')
        plt.box(True)
        plt.title("""In SULOV, we repeatedly remove features with lower mutual info scores among highly correlated pairs (see figure),
                    SULOV selects the feature with higher mutual info score related to target when choosing between a pair. """, fontsize=10)
        plt.suptitle('How SULOV Method of Removing Highly Correlated Features in a Data Set works', fontsize=20,y=1.03)
        red_patch = mpatches.Patch(color='blue', label='Bigger size of circle denotes higher mutual info score with target')
        blue_patch = mpatches.Patch(color='lightblue', label='Thicker line width denotes higher correlation between two variables')
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
    return x
######################################################################################
def convert_all_object_columns_to_numeric(train, test=""):
    """
    #######################################################################################
    This is a utility that converts string columns to numeric WITHOUT LABEL ENCODER.
    The beauty of this utility is that it does not blow up when it finds strings in test not in train.
    #######################################################################################
    """
    train = copy.deepcopy(train)
    if object in train.dtypes.values:
        lis=[]
        for row,column in train.dtypes.iteritems():
            if column == object:
                lis.append(row)
        #print('%d string variables identified' %len(lis))
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
def featurewiz(dataname, target, corr_limit=0.7, verbose=0, sep=",", header=0):
    """
    This is a fast utility that uses XGB to find top features. You
    It returns a list of important features.
    Since it is XGB, you dont have to restrict the input to just numeric vars.
    You can send in all kinds of vars and it will take care of transforming it. Sweet!
    """
    train = load_file_dataframe(dataname, sep, header, verbose)
    start_time = time.time()
    #### If there are more than 30 categorical variables in a data set, it is worth reducing features.
    ####  Otherwise. XGBoost is pretty good at finding the best features whether cat or numeric !
    n_splits = 5
    max_depth = 8
    max_cats = 5
    ######################   I M P O R T A N T ####################################
    subsample =  0.7
    col_sub_sample = 0.7
    test_size = 0.2
    seed = 1
    early_stopping = 5
    ####### All the default parameters are set up now #########
    kf = KFold(n_splits=n_splits, random_state=33)
    ###### This is where we set the CPU and GPU parameters for XGBoost
    GPU_exists = check_if_GPU_exists()
    #####   Set the Scoring Parameters here based on each model and preferences of user ##############
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
    ###############################################################################
    if isinstance(target, str):
        target = [target]
        multi_label = False
    else:
        if len(target) <= 1:
            multi_label = False
        else:
            multi_label = True
    ###### Now we detect the various types of variables to see how to convert them to numeric
    features_dict = classify_features(train, target)
    cols_to_remove = features_dict['cols_delete'] + features_dict['IDcols'] + features_dict['discrete_string_vars']+features_dict['date_vars']
    preds = [x for x in list(train) if x not in target+cols_to_remove]
    numvars = train[preds].select_dtypes(include = 'number').columns.tolist()
    catvars = left_subtract(preds, numvars)
    rem_vars = copy.deepcopy(catvars)
    ########## Now we need to select the right model to run repeatedly ####
    if target is None or len(target) == 0:
        cols_list = list(train)
        modeltype = 'Clustering'
    else:
        modeltype = analyze_problem_type(train, target)
        cols_list = left_subtract(list(train),target)
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
        final_list = remove_variables_using_fast_correlation(train,numvars,modeltype,target,
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
        train, _ = convert_all_object_columns_to_numeric(train, "")
    ########   Dont move this train and y definition anywhere else ########
    y = train[target]
    print('############## F E A T U R E   S E L E C T I O N  ####################')
    important_features = []
    ########## This is for Single_Label problems ######################
    if modeltype == 'Regression':
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
    if multi_label:
        ########## This is for multi_label problems ###############################
        if modeltype == 'Regression':
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
                if modeltype == 'Regression':
                    train_part = int((1-test_size)*X.shape[0])
                    X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
                else:
                    X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                                test_size=test_size, random_state=seed)
                try:
                    if multi_label:
                        eval_set = [(X_train.values,y_train.values),(X_cv.values,y_cv.values)]
                    else:
                        eval_set = [(X_train,y_train),(X_cv,y_cv)]
                    if multi_label:
                        model_xgb.fit(X_train,y_train)
                    else:
                        model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,eval_set=eval_set,
                                            eval_metric=eval_metric,verbose=False)
                except:
                    #### On Colab, even though GPU exists, many people don't turn it on.
                    ####  In that case, XGBoost blows up when gpu_predictor is used.
                    ####  This is to turn it back to cpu_predictor in case GPU errors!
                    if GPU_exists:
                        print('Error: GPU exists but it is not turned on. Using CPU for predictions...')
                        if multi_label:
                            new_xgb.estimator.set_params(**cpu_params)
                            new_xgb.fit(X_train,y_train)
                        else:
                            new_xgb.set_params(**cpu_params)
                            new_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,eval_set=eval_set,
                                        eval_metric=eval_metric,verbose=False)
                #### This is where you collect the feature importances from each run ############
                if multi_label:
                    ### doing this for multi-label is a little different for single label #########
                    imp_feats = [model_xgb.estimators_[i].feature_importances_ for i in range(len(target))]
                    imp_feats_df = pd.DataFrame(imp_feats).T
                    imp_feats_df.columns = target
                    imp_feats_df.index = cols_sel
                    imp_feats_df['sum'] = imp_feats_df.sum(axis=1).values
                    important_features += imp_feats_df.sort_values(by='sum',ascending=False)[:top_num].index.tolist()
                else:
                    ### doing this for single-label is a little different from multi_label #########
                    important_features += pd.Series(model_xgb.get_booster().get_score(
                            importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                #######  order this in the same order in which they were collected ######
                important_features = list(OrderedDict.fromkeys(important_features))
            else:
                X = train_p[list(train_p.columns.values)[i:train_p.shape[1]]]
                cols_sel = X.columns.tolist()
                #### Split here into train and test #####
                if modeltype == 'Regression':
                    train_part = int((1-test_size)*X.shape[0])
                    X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
                else:
                    X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                                test_size=test_size, random_state=seed)
                ### set the validation data as arrays in multi-label case #####
                if multi_label:
                    eval_set = [(X_train.values,y_train.values),(X_cv.values,y_cv.values)]
                else:
                    eval_set = [(X_train,y_train),(X_cv,y_cv)]
                ########## Try training the model now #####################
                try:
                    if multi_label:
                        model_xgb.fit(X_train,y_train)
                    else:
                        model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,
                                  eval_set=eval_set,eval_metric=eval_metric,verbose=False)
                except:
                    #### On Colab, even though GPU exists, many people don't turn it on.
                    ####  In that case, XGBoost blows up when gpu_predictor is used.
                    ####  This is to turn it back to cpu_predictor in case GPU errors!
                    if GPU_exists:
                        print('Error: GPU exists but it is not turned on. Using CPU for predictions...')
                        if multi_label:
                            new_xgb.estimator.set_params(**cpu_params)
                            new_xgb.fit(X_train,y_train)
                        else:
                            new_xgb.set_params(**cpu_params)
                            new_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,
                                  eval_set=eval_set,eval_metric=eval_metric,verbose=False)
                ### doing this for multi-label is a little different for single label #########
                if multi_label:
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
        return important_features
    important_features = list(OrderedDict.fromkeys(important_features))
    print('Found %d important features' %len(important_features))
    print('    Time taken (in seconds) = %0.0f' %(time.time()-start_time))
    numvars = [x for x in numvars if x in important_features]
    important_cats = [x for x in important_cats if x in important_features]
    return important_features
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
        print('')
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
