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
#####   This amazing Library was created by Alex Lekov: Many Thanks to Alex ###
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

import pdb
# disable chained assignments
pd.options.mode.chained_assignment = None 
import copy

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
        self.cat_encoders_names = {
                'HashingEncoder': HashingEncoder,
                'SumEncoder': SumEncoder,
                'PolynomialEncoder': PolynomialEncoder,
                'BackwardDifferenceEncoder': BackwardDifferenceEncoder,
                'OneHotEncoder': OneHotEncoder,
                'HelmertEncoder': HelmertEncoder,
                'OrdinalEncoder': OrdinalEncoder,
                'BaseNEncoder': BaseNEncoder,
                }

        self.target_encoders_names = {
                'CountEncoder': CountEncoder,
                'TargetEncoder': TargetEncoder,
                'CatBoostEncoder': CatBoostEncoder,
                'WOEEncoder': WOEEncoder,
                'JamesSteinEncoder': JamesSteinEncoder,
                'GLMMEncoder': GLMMEncoder
                }


        self.cat_encoder_names = cat_encoder_names
        self.cat_encoder_names_list = list(self.cat_encoders_names.keys())  + list(self.target_encoders_names.keys())
        self.target_encoders_names_list = list(self.target_encoders_names.keys())

        # check X_train, y_train, X_test
        if self.check_data_format(X_train):
            self.X_train_source = pd.DataFrame(X_train)
        if X_test is not None:
            if self.check_data_format(X_test):
                self.X_test_source = pd.DataFrame(X_test)
                
        if y_train is not None:
            if self.check_data_format(X_test):
                self.y_train_source = pd.DataFrame(y_train)
            else:
                self.y_train_source = y_train
        else:
            print("No target data found!")
            return 

        if y_test is not None:
            self.y_test = y_test
        
        if verbose > 0:   
            print('Source X_train shape: ', X_train.shape, '| X_test shape: ', X_test.shape)
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
                print('> Start preprocessing Data')
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
        return(True)


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
                    data[nan_column+'isNAN'] = pd.isna(data[nan_column]).astype('uint8')
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
                    encoder = self.cat_encoders_names[cat_encoder_name](cols=self.cat_features, n_components=int(np.log(len(data.columns))*1000), 
                                                            drop_invariant=True)
                else:
                    encoder = self.cat_encoders_names[cat_encoder_name](cols=self.cat_features, drop_invariant=True)
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
            if cat_encoder_name in self.target_encoders_names_list:
                encoder = self.target_encoders_names[cat_encoder_name](cols=self.cat_features, drop_invariant=True)
                data_encodet = encoder.fit_transform(x_data, y_data)
                data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
            else:
                print(f"{cat_encoder_name} is not supported!")
                return ('', '')
        else:
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
        
        # concat datasets for correct processing.
        df_train = X_train.copy()
        
        if X_test is None:
            data = df_train
            test_data = None ### Set test_data to None if X_test is None
        else: 
            test_data = X_test.copy()
            data = copy.deepcopy(df_train)

        # object & num features
        object_features = list(data.columns[(data.dtypes == 'object') | (data.dtypes == 'category')])
        num_features = list(set(data.columns) - set(cat_features) - set(object_features) - {'test'})
        encodet_features_names = list(set(object_features + list(cat_features)))
        
        self.encodet_features_names = encodet_features_names
        self.num_features_names = num_features
        self.binary_features_names = []

        # LabelEncoded Binary Features
        for feature in data.columns:
            if (feature != 'test') and (data[feature].nunique(dropna=False) < 3):
                data[feature] = data[feature].astype('category').cat.codes
                if test_data is not None:
                    test_data[feature] = test_data[feature].astype('category').cat.codes                    
                self.binary_features_names.append(feature)
                        
        # Generator cat encodet features
        if encodet_features_names:
            if cat_encoder_names is None:
                for feature in encodet_features_names:
                    data[feature] = data[feature].astype('category').cat.codes
                    if test_data is not None:
                        test_data[feature] = test_data[feature].astype('category').cat.codes                    
            else:
                if verbose > 0:
                    print('> Generate cat encoder features')

                copy_cat_encoder_names = copy.deepcopy(cat_encoder_names)
                for encoder_name in copy_cat_encoder_names:
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
                            print(' + ', data_encodet.shape[1], ' Features from ', encoder_name)

        # Generate Target related Encoder features for cat variables:
        
        target_encoders = [x for x in self.cat_encoder_names if x in self.target_encoders_names_list]
        if len(target_encoders) > 0:
            target_enc_cat_features =  True

        if target_enc_cat_features:            
            if encodet_features_names:
                if verbose > 0:
                    print('> Generate Target Encoded cat features')

                copy_target_encoders = copy.deepcopy(target_encoders)
                for encoder_name in copy_target_encoders:
                    data_encodet, train_encoder = self.gen_target_encodet_features(data[encodet_features_names], 
                                                                y_train, encoder_name)
                    if not isinstance(data_encodet, str):
                        data = data.join(data_encodet)

                    if test_data is not None:
                        test_encodet, _ = self.gen_target_encodet_features(test_data[encodet_features_names],'',
                                                                train_encoder)                   
                        if not isinstance(test_encodet, str):
                            test_data = test_data.join(test_encodet)

                
                if verbose > 0:
                    if not isinstance(data_encodet, str):
                        print(' + ', data_encodet.shape[1], ' Frequency Encode Num Features ',)

        # Nans
        if clean_nan:
            if verbose > 0:
                print('> Clean NaNs in num features')
            data = self.clean_nans(data, cols=num_features)
            if test_data is not None:
                test_data = self.clean_nans(test_data, cols=num_features)

        # Generator interaction Num Features
        if num_generator_features:
            if len(num_features) > 1:
                if verbose > 0:
                    print('> Generate interaction Num Features')
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
                        print(' + ', fe_df.shape[1], ' Interaction Features')

        # Generator Group Encoder Features
        if group_generator_features:
            if encodet_features_names and num_features:
                if verbose > 0:
                    print('> Generate Group Encoder Features')
                count = 0
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

                    if not isinstance(data_encodet, str):
                        count += data_encodet.shape[1]
                if verbose > 0:
                    print(' + ', count, ' Group cat Encoder Features')

        # Drop source cat features
        data.drop(columns=encodet_features_names, inplace=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)
        if test_data is not None:
            test_data.drop(columns=encodet_features_names, inplace=True)
            test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            test_data.fillna(0, inplace=True)

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
        
        return (X_train, X_test)
