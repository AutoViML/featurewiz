###############################################################################
# MIT License
#
# Copyright (c) 2020 Alex Lekov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
#####  This amazing Library was created by Alex Lekov: Many Thanks to Alex! ###
#####                https://github.com/Alex-Lekov/AutoML_Alex              ###
###############################################################################
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from category_encoders import HashingEncoder, SumEncoder, PolynomialEncoder, BackwardDifferenceEncoder
from category_encoders import OneHotEncoder, HelmertEncoder, OrdinalEncoder, CountEncoder, BaseNEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder, JamesSteinEncoder
from category_encoders.glmm import GLMMEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.wrapper import PolynomialWrapper
from .encoders import FrequencyEncoder
from . import settings

import pdb
# disable chained assignments
pd.options.mode.chained_assignment = None
import copy
import dask
import dask.dataframe as dd

class DataBunch(object):
    """
    Сlass for storing, cleaning and processing your dataset
    """
    def __init__(self,
                X_train=None,
                y_train=None,
                X_test=None,
                y_test=None,
                cat_features=None,
                clean_and_encod_data=True,
                cat_encoder_names=None,
                clean_nan=True,
                num_generator_features=True,
                group_generator_features=True,
                target_enc_cat_features=True,
                normalization=True,
                random_state=42,
                verbose=1):
        """
        Description of __init__

        Args:
            X_train=None (undefined): dataset
            y_train=None (undefined): y
            X_test=None (undefined): dataset
            y_test=None (undefined): y
            cat_features=None (list or None):
            clean_and_encod_data=True (undefined):
            cat_encoder_names=None (list or None):
            clean_nan=True (undefined):
            num_generator_features=True (undefined):
            group_generator_features=True (undefined):
            target_enc_cat_features=True (undefined)
            random_state=42 (undefined):
            verbose = 1 (undefined)
        """
        self.random_state = random_state

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_predicts = None
        self.X_test_predicts = None
        self.cat_features = None

        # Encoders
        self.cat_encoders_names = settings.cat_encoders_names
        self.target_encoders_names = settings.target_encoders_names


        self.cat_encoder_names = cat_encoder_names
        self.cat_encoder_names_list = list(self.cat_encoders_names.keys())  + list(self.target_encoders_names.keys())
        self.target_encoders_names_list = list(self.target_encoders_names.keys())

        # check X_train, y_train, X_test
        if self.check_data_format(X_train):
            if type(X_train) == dask.dataframe.core.DataFrame:
                self.X_train_source = X_train.compute()
            else:
                self.X_train_source = pd.DataFrame(X_train)
            self.X_train_source = remove_duplicate_cols_in_dataset(self.X_train_source)
        if X_test is not None:
            if self.check_data_format(X_test):
                if type(X_test) == dask.dataframe.core.DataFrame:
                    self.X_test_source = X_test.compute()
                else:
                    self.X_test_source = pd.DataFrame(X_test)
                self.X_test_source = remove_duplicate_cols_in_dataset(self.X_test_source)


        ### There is a chance for an error in this - so worth watching!
        if y_train is not None:
            le = LabelEncoder()
            if self.check_data_format(y_train):
                if settings.multi_label:
                    ### if the model is mult-Label, don't transform it since it won't work
                    self.y_train_source = y_train
                else:
                    if not isinstance(y_train, pd.DataFrame):
                        if y_train.dtype == 'object' or str(y_train.dtype) == 'category':
                            self.y_train_source =  le.fit_transform(y_train)
                        else:
                            if settings.modeltype == 'Multi_Classification':
                                rare_class = find_rare_class(y_train)
                                if rare_class != 0:
                                    ### if the rare class is not zero, then transform it using Label Encoder
                                    y_train =  le.fit_transform(y_train)
                            self.y_train_source =  copy.deepcopy(y_train)
                    else:
                        print('Error: y_train should be a series. Skipping target encoding for dataset...')
                        target_enc_cat_features = False
            else:
                if settings.multi_label:
                    self.y_train_source = pd.DataFrame(y_train)
                else:
                    if y_train.dtype == 'object' or str(y_train.dtype) == 'category':
                        self.y_train_source = le.fit_transform(pd.DataFrame(y_train))
                    else:
                        self.y_train_source =  copy.deepcopy(y_train)
        else:
            print("No target data found!")
            return

        if y_test is not None:
            self.y_test = y_test

        if verbose > 0:
            print('Source X_train shape: ', self.X_train_source.shape)
            if not X_test is None:
                print('| Source X_test shape: ', self.X_test_source.shape)
            print('#'*50)

        # add categorical features in DataBunch
        if cat_features is None:
            self.cat_features = self.auto_detect_cat_features(self.X_train_source)
            if verbose > 0:
                print('Auto detect cat features: ', len(self.cat_features))

        else:
            self.cat_features = list(cat_features)
        
        # preproc_data in DataBunch
        if clean_and_encod_data:
            if verbose > 0:
                print('> Start preprocessing with %d variables' %self.X_train_source.shape[1])
            self.X_train, self.X_test = self.preproc_data(self.X_train_source,
                                                            self.X_test_source,
                                                            self.y_train_source,
                                                            cat_features=self.cat_features,
                                                            cat_encoder_names=cat_encoder_names,
                                                            clean_nan=clean_nan,
                                                            num_generator_features=num_generator_features,
                                                            group_generator_features=group_generator_features,
                                                            target_enc_cat_features=target_enc_cat_features,
                                                            normalization=normalization,
                                                            verbose=verbose,)
        else:
            self.X_train, self.X_test = X_train, X_test


    def check_data_format(self, data):
        """
        Description of check_data_format:
            Check that data is not pd.DataFrame or empty

        Args:
            data (undefined): dataset
        Return:
            True or Exception
        """
        data_tmp = pd.DataFrame(data)
        if data_tmp is None or data_tmp.empty:
            raise Exception("data is not pd.DataFrame or empty")
        else:
            if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
                return True
            elif isinstance(data, np.ndarray):
                return True
            elif type(data) == dask.dataframe.core.DataFrame:
                return True
            else:
                False

    def clean_nans(self, data, cols=None):
        """
        Fill Nans and add column, that there were nans in this column

        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Return:
            Clean data (pd.DataFrame, shape (n_samples, n_features))

        """
        if cols is not None:
            nan_columns = list(data[cols].columns[data[cols].isnull().sum() > 0])
            if nan_columns:
                for nan_column in nan_columns:
                    data[nan_column+'_isNAN'] = pd.isna(data[nan_column]).astype('uint8')
                data.fillna(data.median(), inplace=True)
        return(data)


    def auto_detect_cat_features(self, data):
        """
        Description of _auto_detect_cat_features:
            Auto-detection categorical_features by simple rule:
            categorical feature == if feature nunique low 1% of data

        Args:
            data (pd.DataFrame): dataset

        Returns:
            cat_features (list): columns names cat features

        """
        #object_features = list(data.columns[data.dtypes == 'object'])
        cat_features = data.columns[(data.nunique(dropna=False) < len(data)//100) & \
            (data.nunique(dropna=False) >2)]
        #cat_features = list(set([*object_features, *cat_features]))
        return (cat_features)


    def gen_cat_encodet_features(self, data, cat_encoder_name):
        """
        Description of _encode_features:
            Encode car features

        Args:
            data (pd.DataFrame):
            cat_encoder_name (str): cat Encoder name

        Returns:
            pd.DataFrame

        """
        if isinstance(cat_encoder_name, str):
            if cat_encoder_name in self.cat_encoder_names_list and cat_encoder_name not in self.target_encoders_names_list:
                if cat_encoder_name == 'HashingEncoder':
                    encoder = self.cat_encoders_names[cat_encoder_name][0](cols=self.cat_features, n_components=int(np.log(len(data.columns))*1000),
                                                            drop_invariant=True)
                else:
                    encoder = self.cat_encoders_names[cat_encoder_name][0](cols=self.cat_features, drop_invariant=True)
                data_encodet = encoder.fit_transform(data)
                data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
            else:
                print(f"{cat_encoder_name} is not supported!")
                return ('', '')
        else:
            encoder = copy.deepcopy(cat_encoder_name)
            data_encodet = encoder.transform(data)
            data_encodet = data_encodet.add_prefix(str(cat_encoder_name).split("(")[0] + '_')

        return (data_encodet, encoder)


    def gen_target_encodet_features(self, x_data, y_data=None, cat_encoder_name=''):
        """
        Description of _encode_features:
            Encode car features

        Args:
            data (pd.DataFrame):
            cat_encoder_name (str): cat Encoder name

        Returns:
            pd.DataFrame

        """

        if isinstance(cat_encoder_name, str):
            ### If it is the first time, it will perform fit_transform !
            if cat_encoder_name in self.target_encoders_names_list:
                encoder = self.target_encoders_names[cat_encoder_name][0](cols=self.cat_features, drop_invariant=True)
                if settings.modeltype == 'Multi_Classification':
                    ### you must put a Polynomial Wrapper on the cat_encoder in case the model is multi-class
                    if cat_encoder_name in ['WOEEncoder']:
                        encoder = PolynomialWrapper(encoder)
                ### All other encoders TargetEncoder CatBoostEncoder GLMMEncoder don't need
                ### Polynomial Wrappers since they handle multi-class (label encoded) very well!
                data_encodet = encoder.fit_transform(x_data, y_data)
                data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
            else:
                print(f"{cat_encoder_name} is not supported!")
                return ('', '')
        else:
            ### if it is already fit, then it will only do transform here !
            encoder = copy.deepcopy(cat_encoder_name)
            data_encodet = encoder.transform(x_data)
            data_encodet = data_encodet.add_prefix(str(cat_encoder_name).split("(")[0] + '_')

        return (data_encodet, encoder)

    def gen_numeric_interaction_features(self,
                                        df,
                                        columns,
                                        operations=['/','*','-','+'],):
        """
        Description of numeric_interaction_terms:
            Numerical interaction generator features: A/B, A*B, A-B,

        Args:
            df (pd.DataFrame):
            columns (list): num columns names
            operations (list): operations type

        Returns:
            pd.DataFrame

        """
        copy_columns = copy.deepcopy(columns)
        fe_df = pd.DataFrame()
        for combo_col in combinations(columns,2):
            if '/' in operations:
                fe_df['{}_div_by_{}'.format(combo_col[0], combo_col[1]) ] = (df[combo_col[0]]*1.) / df[combo_col[1]]
            if '*' in operations:
                fe_df['{}_mult_by_{}'.format(combo_col[0], combo_col[1]) ] = df[combo_col[0]] * df[combo_col[1]]
            if '-' in operations:
                fe_df['{}_minus_{}'.format(combo_col[0], combo_col[1]) ] = df[combo_col[0]] - df[combo_col[1]]
            if '+' in operations:
                fe_df['{}_plus_{}'.format(combo_col[0], combo_col[1]) ] = df[combo_col[0]] + df[combo_col[1]]

        for each_col in copy_columns:
            fe_df['{}_squared'.format(each_col) ] = df[each_col].pow(2)
        return (fe_df)


    def gen_groupby_cat_encode_features(self, data, cat_columns, num_column,
                                cat_encoder_name='JamesSteinEncoder'):
        """
        Description of group_encoder

        Args:
            data (pd.DataFrame): dataset
            cat_columns (list): cat columns names
            num_column (str): num column name

        Returns:
            pd.DataFrame

        """

        if isinstance(cat_encoder_name, str):
            if cat_encoder_name in self.cat_encoder_names_list:
                encoder = JamesSteinEncoder(cols=self.cat_features, model='beta', return_df = True, drop_invariant=True)
                encoder.fit(X=data[cat_columns], y=data[num_column].values)
            else:
                print(f"{cat_encoder_name} is not supported!")
                return ('', '')
        else:
            encoder = copy.deepcopy(cat_encoder_name)

        data_encodet = encoder.transform(X=data[cat_columns], y=data[num_column].values)
        data_encodet = data_encodet.add_prefix('GroupEncoded_' + num_column + '_')

        return (data_encodet, encoder)

    def preproc_data(self, X_train=None,
                        X_test=None,
                        y_train=None,
                        cat_features=None,
                        cat_encoder_names=None,
                        clean_nan=True,
                        num_generator_features=True,
                        group_generator_features=True,
                        target_enc_cat_features=True,
                        normalization=True,
                        verbose=1,):
        """
        Description of preproc_data:
            dataset preprocessing function

        Args:
            X_train=None (pd.DataFrame):
            X_test=None (pd.DataFrame):
            y_train=None (pd.DataFrame):
            cat_features=None (list):
            cat_encoder_names=None (list):
            clean_nan=True (Bool):
            num_generator_features=True (Bool):
            group_generator_features=True (Bool):

        Returns:
            X_train (pd.DataFrame)
            X_test (pd.DataFrame)

        """
        #### Sometimes there are duplicates in column names. You must remove them here. ###
        cat_features = find_remove_duplicates(cat_features)

        # concat datasets for correct processing.
        df_train = X_train.copy()

        if X_test is None:
            data = df_train
            test_data = None ### Set test_data to None if X_test is None
        else:
            test_data = X_test.copy()
            test_data = remove_duplicate_cols_in_dataset(test_data)
            data = copy.deepcopy(df_train)

        data = remove_duplicate_cols_in_dataset(data)

        # object & num features
        object_features = list(data.columns[(data.dtypes == 'object') | (data.dtypes == 'category')])
        num_features = list(set(data.columns) - set(cat_features) - set(object_features) - {'test'})
        encodet_features_names = list(set(object_features + list(cat_features)))

        original_number_features = len(encodet_features_names)
        count_number_features = df_train.shape[1]

        self.encodet_features_names = encodet_features_names
        self.num_features_names = num_features
        self.binary_features_names = []

        # LabelEncode all Binary Features - leave the rest alone
        cols = data.columns.tolist()
        #### This sometimes errors because there are duplicate columns in a dataset ###
        for feature in cols:
            if (feature != 'test') and (data[feature].nunique(dropna=False) < 3):
                data[feature] = data[feature].astype('category').cat.codes
                if test_data is not None:
                    test_data[feature] = test_data[feature].astype('category').cat.codes
                self.binary_features_names.append(feature)

        # Convert all Category features "Category" type variables if no encoding is specified
        cat_only_encoders = [x for x in self.cat_encoder_names if x in self.cat_encoders_names]
        if len(cat_only_encoders) > 0:
            ### Just skip if this encoder is not in the list of category encoders ##
            if encodet_features_names:
                if cat_encoder_names is None:
                    for feature in encodet_features_names:
                        data[feature] = data[feature].astype('category').cat.codes
                        if test_data is not None:
                            test_data[feature] = test_data[feature].astype('category').cat.codes
                else:
                    #### If an encoder is specified, then use that encoder to transform categorical variables
                    if verbose > 0:
                        print('> Generate Categorical Encoded features')

                    copy_cat_encoder_names = copy.deepcopy(cat_encoder_names)
                    for encoder_name in copy_cat_encoder_names:
                        if verbose > 0:
                            print(' + To know more, click: %s' %self.cat_encoders_names[encoder_name][1])
                        data_encodet, train_encoder = self.gen_cat_encodet_features(data[encodet_features_names],
                                                                    encoder_name)
                        if not isinstance(data_encodet, str):
                            data = data.join(data_encodet)
                        if test_data is not None:
                            test_encodet, _ = self.gen_cat_encodet_features(test_data[encodet_features_names],
                                                                    train_encoder)
                            if not isinstance(test_encodet, str):
                                test_data = test_data.join(test_encodet)

                        if verbose > 0:
                            if not isinstance(data_encodet, str):
                                addl_features = data_encodet.shape[1] - original_number_features
                                count_number_features += addl_features
                                print(' + added ', addl_features, ' additional Features using',encoder_name)

        # Generate Target related Encoder features for cat variables:
        target_encoders = [x for x in self.cat_encoder_names if x in self.target_encoders_names_list]
        if len(target_encoders) > 0:
            target_enc_cat_features =  True
        if target_enc_cat_features:
            if encodet_features_names:
                if verbose > 0:
                    print('> Generate Target Encoded categorical features')

                if len(target_encoders) == 0:
                    target_encoders = ['TargetEncoder'] ### set the default as TargetEncoder if nothing is specified
                copy_target_encoders = copy.deepcopy(target_encoders)
                for encoder_name in copy_target_encoders:
                    if verbose > 0:
                        print(' + To know more, click: %s' %self.target_encoders_names[encoder_name][1])
                    data_encodet, train_encoder = self.gen_target_encodet_features(data[encodet_features_names],
                                                                self.y_train_source, encoder_name)
                    if not isinstance(data_encodet, str):
                        data = data.join(data_encodet)

                    if test_data is not None:
                        test_encodet, _ = self.gen_target_encodet_features(test_data[encodet_features_names],'',
                                                                train_encoder)
                        if not isinstance(test_encodet, str):
                            test_data = test_data.join(test_encodet)


                if verbose > 0:
                    if not isinstance(data_encodet, str):
                        addl_features = data_encodet.shape[1] - original_number_features
                        count_number_features += addl_features
                        print(' + added ', len(encodet_features_names) , ' additional Features using ', encoder_name)

        # Clean NaNs in Numeric variables only
        if clean_nan:
            if verbose > 0:
                print('> Cleaned NaNs in numeric features')
            data = self.clean_nans(data, cols=num_features)
            if test_data is not None:
                test_data = self.clean_nans(test_data, cols=num_features)
            ### Sometimes, train has nulls while test doesn't and vice versa
            if test_data is not None:
                rem_cols = left_subtract(list(data),list(test_data))
                if len(rem_cols) > 0:
                    for rem_col in rem_cols:
                        test_data[rem_col] = 0
                elif len(left_subtract(list(test_data),list(data))) > 0:
                    rem_cols = left_subtract(list(test_data),list(data))
                    for rem_col in rem_cols:
                        data[rem_col] = 0
                else:
                    print(' + test and train have similar NaN columns')
        # Generate interaction features for Numeric variables
        if num_generator_features:
            if len(num_features) > 1:
                if verbose > 0:
                    print('> Generate Interactions features among Numeric variables')
                fe_df = self.gen_numeric_interaction_features(data[num_features],
                                                            num_features,
                                                            operations=['/','*','-','+'],)
                if not isinstance(fe_df, str):
                    data = data.join(fe_df)
                if test_data is not None:
                    fe_test = self.gen_numeric_interaction_features(test_data[num_features],
                                                            num_features,
                                                            operations=['/','*','-','+'],)
                    if not isinstance(fe_test, str):
                        test_data = test_data.join(fe_test)

                if verbose > 0:
                    if not isinstance(fe_df, str):
                        addl_features = fe_df.shape[1]
                        count_number_features += addl_features
                        print(' + added ', addl_features, ' Interaction Features ',)

        # Generate Group Encoded Features for Numeric variables only using all Categorical variables
        if group_generator_features:
            if encodet_features_names and num_features:
                if verbose > 0:
                    print('> Generate Group-by Encoded Features')
                    print(' + To know more, click: %s' %self.target_encoders_names['JamesSteinEncoder'][1])

                for num_col in num_features:
                    data_encodet, train_group_encoder = self.gen_groupby_cat_encode_features(
                                                            data,
                                                            encodet_features_names,
                                                            num_col,)
                    if not isinstance(data_encodet, str):
                        data = data.join(data_encodet)
                    if test_data is not None:
                        test_encodet, _ = self.gen_groupby_cat_encode_features(
                                                            data,
                                                            encodet_features_names,
                                                            num_col,train_group_encoder)
                        if not isinstance(test_encodet, str):
                            test_data = test_data.join(test_encodet)

                if verbose > 0:
                    addl_features = data_encodet.shape[1]*len(num_features)
                    count_number_features += addl_features
                    print(' + added ', addl_features, ' Group-by Encoded Features using JamesSteinEncoder')

        # Drop source cat features
        if not len(cat_encoder_names) == 0:
            ### if there is no categorical encoding, then let the categorical_vars pass through.
            ### If they have been transformed into Cat Encoded variables, then you can drop them!
            data.drop(columns=encodet_features_names, inplace=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        #data.fillna(0, inplace=True)
        if test_data is not None:
            if not len(cat_encoder_names) == 0:
                ### if there is no categorical encoding, then let the categorical_vars pass through.
                test_data.drop(columns=encodet_features_names, inplace=True)
            test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            #test_data.fillna(0, inplace=True)

        X_train = copy.deepcopy(data)
        X_test = copy.deepcopy(test_data)

        # Normalization Data
        if normalization:
            if verbose > 0:
                print('> Normalization Features')
            columns_name = X_train.columns.values
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=columns_name)
            X_test = pd.DataFrame(X_test, columns=columns_name)

        if verbose > 0:
            print('#'*50)
            print('> Final Number of Features: ', (X_train.shape[1]))
            print('#'*50)
            print('New X_train shape: ', X_train.shape, '| X_test shape: ', X_test.shape)
            if len(left_subtract(X_test.columns, X_train.columns)) > 0:
                print("""There are more columns in test than train 
                    due to missing columns being more in test than train. Continuing...""")

        return (X_train, X_test)
################################################################################
def find_rare_class(series, verbose=0):
    ######### Print the % count of each class in a Target variable  #####
    """
    Works on Multi Class too. Prints class percentages count of target variable.
    It returns the name of the Rare class (the one with the minimum class member count).
    This can also be helpful in using it as pos_label in Binary and Multi Class problems.
    """
    return series.value_counts().index[-1]
#################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#################################################################################
def remove_duplicate_cols_in_dataset(df):
    df = copy.deepcopy(df)
    cols = df.columns.tolist()
    number_duplicates = df.columns.duplicated().astype(int).sum()
    if  number_duplicates > 0:
        print('Detected %d duplicate columns in dataset. Removing duplicates...' %number_duplicates)
        df = df.loc[:,~df.columns.duplicated()]
    return df
###########################################################################
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
