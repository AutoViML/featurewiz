import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin #gives fit_transform method for free
import pdb
import copy
from sklearn.base import TransformerMixin
from collections import defaultdict
from category_encoders import OneHotEncoder
from category_encoders import HashingEncoder, SumEncoder, PolynomialEncoder, BackwardDifferenceEncoder
from category_encoders import HelmertEncoder, OrdinalEncoder, CountEncoder, BaseNEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder, JamesSteinEncoder
#from sklearn.preprocessing import OneHotEncoder
from category_encoders.glmm import GLMMEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.wrapper import PolynomialWrapper
from sklearn.preprocessing import FunctionTransformer
from pandas.api.types import is_datetime64_any_dtype

#################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#################################################################################
class My_LabelEncoder(BaseEstimator, TransformerMixin):
    """
    ################################################################################################
    ######     The My_LabelEncoder class works just like sklearn's Label Encoder but better! #######
    #####  It label encodes any cat var in your dataset. It also handles NaN's in your dataset! ####
    ##  The beauty of this function is that it takes care of NaN's and unknown (future) values.#####
    ##################### This is the BEST working version - don't mess with it!! ##################
    ################################################################################################
    Usage:
          le = My_LabelEncoder()
          le.fit_transform(train[column]) ## this will give your transformed values as an array
          le.transform(test[column]) ### this will give your transformed values as an array
              
    Usage in Column Transformers and Pipelines:
          No. It cannot be used in pipelines since it need to produce two columns for the next stage in pipeline.
          See my other module called My_LabelEncoder_Pipe() to see how it can be used in Pipelines.
    """
    def __init__(self):
        self.transformer = defaultdict(str)
        self.inverse_transformer = defaultdict(str)
        self.max_val = 0
        
    def fit(self,testx, y=None):
        ### Do not change this since Rare class combiner requires this test ##
        if isinstance(testx, tuple):
            y = testx[1]
            testx = testx[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            #### There is no way to transform dataframes since you will get a nested renamer error if you try ###
            ### But if it is a one-dimensional dataframe, convert it into a Series
            if testx.shape[1] == 1:
                testx = pd.Series(testx.values.ravel(),name=testx.columns[0])
            else:
                #### Since it is multi-dimensional, So in this case, just return the data as is
                return self
        ins = np.unique(testx.factorize()[1]).tolist()
        outs = np.unique(testx.factorize()[0]).tolist()
        #ins = testx.value_counts(dropna=False).index        
        if -1 in outs:
        #   it already has nan if -1 is in outs. No need to add it.
            if not np.nan in ins:
                ins.insert(0,np.nan)
        self.transformer = dict(zip(ins,outs))
        self.inverse_transformer = dict(zip(outs,ins))
        return self

    def transform(self, testx, y=None):
        ### Do not change this since Rare class combiner requires this test ##
        if isinstance(testx, tuple):
            y = testx[1]
            testx = testx[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            #### There is no way to transform dataframes since you will get a nested renamer error if you try ###
            ### But if it is a one-dimensional dataframe, convert it into a Series
            if testx.shape[1] == 1:
                testx = pd.Series(testx.values.ravel(),name=testx.columns[0])
            else:
                #### Since it is multi-dimensional, So in this case, just return the data as is
                return testx, y
        ### now convert the input to transformer dictionary values
        new_ins = np.unique(testx.factorize()[1]).tolist()
        missing = [x for x in new_ins if x not in self.transformer.keys()]
        if len(missing) > 0:
            for each_missing in missing:
                self.transformer[each_missing] = int(self.max_val + 1)
                self.inverse_transformer[int(self.max_val+1)] = each_missing
                self.max_val = int(self.max_val+1)
        else:
            self.max_val = np.max(list(self.transformer.values()))
        outs = testx.map(self.transformer).values.astype(int)
        ### To handle category dtype you must do the next step #####
        testk = testx.map(self.transformer) ## this must be still a pd.Series
        if testx.dtype not in [np.int16, np.int32, np.int64,np.int8, float, bool, object]:
            if testx.isnull().sum().sum() > 0:
                fillval = self.transformer[np.nan]
                testk = testk.cat.add_categories([fillval])
                testk = testk.fillna(fillval)
                testk = testk.astype(int)
                return testk, y
            else:
                testk = testk.astype(int)
                return testk, y
        else:
            return outs

    def inverse_transform(self, testx, y=None):
        ### now convert the input to transformer dictionary values
        if isinstance(testx, pd.Series):
            outs = testx.map(self.inverse_transformer).values
        elif isinstance(testx, np.ndarray):
            outs = pd.Series(testx).map(self.inverse_transformer).values
        else:
            outs = testx[:]
        return outs
#################################################################################
from collections import defaultdict
# This is needed to make this a regular transformer ###
from sklearn.base import BaseEstimator, TransformerMixin 
class Rare_Class_Combiner_Pipe(BaseEstimator, TransformerMixin ):
    """
    This is the pipeline version of rare class combiner used in sklearn pipelines.
    """
    def __init__(self, transformers={}  ):
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self.transformers =  transformers
        self.zero_low_counts = defaultdict(bool)
        
    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"transformers": self.transformers}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data"""        
        X =  copy.deepcopy(X)
        # transformers need a default name for rare categories ##
        def return_cat_value():
            return "rare_categories"
        ### In this case X itself will only be a pd.Series ###
        each_catvar = X.name
        #### if it is already a list, then leave it as is ###
        self.transformers[each_catvar] = defaultdict(return_cat_value)
        ### Then find the unique categories in the column ###
        self.transformers[each_catvar] = dict(zip(X.unique(), X.unique()))
        low_counts = pd.DataFrame(X).apply(lambda x: x.value_counts()[
                (x.value_counts()<=(0.01*x.shape[0])).values].index).values.ravel()
        
        if len(low_counts) == 0:
            self.zero_low_counts[each_catvar] = True
        else:
            self.zero_low_counts[each_catvar] = False
        for each_low in low_counts:
            self.transformers[each_catvar].update({each_low:'rare_categories'})
        return self
    
    def transform(self, X, y=None, **fit_params):
        X =  copy.deepcopy(X)
        each_catvar = X.name
        if self.zero_low_counts[each_catvar]:
            pass
        else:
            X = X.map(self.transformers[each_catvar])
            ### simply fill in the missing values with the word "missing" ##
            X = X.fillna('missing')
        return X

    def fit_transform(self, X, y=None, **fit_params):
        X =  copy.deepcopy(X)
        ### Since X for yT in a pipeline is sent as X, we need to switch X and y this way ##
        self.fit(X, y)
        each_catvar = X.name
        if self.zero_low_counts[each_catvar]:
            pass
        else:
            X = X.map(self.transformers[each_catvar])
            ### simply fill in the missing values with the word "missing" ##
            X = X.fillna('missing')
        return X

    def inverse_transform(self, X, **fit_params):
        ### One problem with this approach is that you have combined categories into one.
        ###   You cannot uncombine them since they no longer have a unique category. 
        ###   You will get back the last transformed category when you inverse transform it.
        each_catvar = X.name
        transformer_ = self.transformers[each_catvar]
        reverse_transformer_ = dict([(y,x) for (x,y) in transformer_.items()])
        if self.zero_low_counts[each_catvar]:
            pass
        else:
            X[each_catvar] = X[each_catvar].map(reverse_transformer_).values
        return X
    
    def predict(self, X, y=None, **fit_params):
        #print('There is no predict function in Rare class combiner. Returning...')
        return X
######################################################################################
from pandas.api.types import is_numeric_dtype
class Rare_Class_Combiner(BaseEstimator, TransformerMixin):
    """
    This is the general version of combining classes in categorical vars. 
    You cannot use it in sklearn pipelines. You can however use it alone to make changes.
    """
    def __init__(self, transformers={}, categorical_features=[], zero_low_counts=False):
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self.transformers =  transformers
        self.categorical_features = categorical_features
        self.zero_low_counts = {}
        if zero_low_counts:
            for each_cat in categorical_features:
                self.zero_low_counts[each_cat] = zero_low_counts
        else:
            for each_cat in categorical_features:
                self.zero_low_counts[each_cat] = 0
        

    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"transformers": self.transformers, "categorical_features": self.categorical_features,
                    "zero_low_counts": self.zero_low_counts}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data"""        
        X =  copy.deepcopy(X)
        # transformers need a default name for rare categories ##
        # transformers are designed to modify X which is 2d dimensional
        if len(self.categorical_features) == 0:
            if isinstance(X, pd.Series):
                self.categorical_features = [X.name]
            elif isinstance(X, np.ndarray):
                print('Error: Input cannot be a numpy array for transformers')
                return X, y
            else:
                # if X is a dataframe, then you need the list of features ##
                self.categorical_features = X.columns.tolist()
            if isinstance(self.categorical_features, str):
                self.categorical_features = [self.categorical_features]
        #### if it is already a list, then leave it as is ###
        for i, each_catvar in enumerate(self.categorical_features):
            if is_numeric_dtype(X[each_catvar]):
                max_value = X[each_catvar].max()
                save_value = max_value+1
            else:
                save_value = "rare_categories"
            ### Then find the unique categories in the column ###
            self.transformers[each_catvar] = dict(zip(X[each_catvar].unique(),X[each_catvar].unique()))
            low_counts = X[[each_catvar]].apply(lambda x: x.value_counts()[
                    (x.value_counts()<=(0.01*x.shape[0])).values].index).values.ravel()
            ### This is where we find whether cat var has even a single low category ###
            if len(low_counts) == 0:
                self.zero_low_counts[each_catvar] = save_value
            else:
                self.zero_low_counts[each_catvar] = 0
            for each_low in low_counts:
                self.transformers[each_catvar].update({each_low: save_value})
        return self
    
    def transform(self, X, y=None, **fit_params):
        X =  copy.deepcopy(X)
        for i, each_catvar in enumerate(self.categorical_features):
            if self.zero_low_counts[each_catvar]:
                continue
            else:
                X[each_catvar] = X[each_catvar].map(self.transformers[each_catvar]).values
                ### simply fill in the missing values with the word "missing" ##
                ### Remember that fillna only works at dataframe level! ##
                X[[each_catvar]] = X[[each_catvar]].fillna('missing')
        return X

    def fit_transform(self, X, y=None, **fit_params):
        X =  copy.deepcopy(X)
        ### Since X for yT in a pipeline is sent as X, we need to switch X and y this way ##
        self.fit(X, y)
        for i, each_catvar in enumerate(self.categorical_features):
            if self.zero_low_counts[each_catvar]:
                continue
            else:
                X[each_catvar] = X[each_catvar].map(self.transformers[each_catvar]).values
                ### simply fill in the missing values with the word "missing" ##
                ### Remember that fillna only works at dataframe level! ##
                X[[each_catvar]] = X[[each_catvar]].fillna('missing')
        return X

    def inverse_transform(self, X, **fit_params):
        ### One problem with this approach is that you have combined categories into one.
        ###   You cannot uncombine them since they no longer have a unique category. 
        ###   You will get back the last transformed category when you inverse transform it.
        for i, each_catvar in enumerate(self.categorical_features):
            transformer_ = self.transformers[each_catvar]
            reverse_transformer_ = dict([(y,x) for (x,y) in transformer_.items()])
            if self.zero_low_counts[each_catvar]:
                continue
            else:
                X[each_catvar] = X[each_catvar].map(reverse_transformer_).values
        return X
    
    def predict(self, X, y=None, **fit_params):
        #print('There is no predict function in Rare class combiner. Returning...')
        return X
######################################################################################
class My_LabelEncoder_Pipe(BaseEstimator, TransformerMixin):
    """
    ################################################################################################
    ######  The My_LabelEncoder_Pipe class works just like sklearn's Label Encoder but better! #####
    #####  It label encodes any cat var in your dataset. But it can also be used in Pipelines! #####
    ##  The beauty of this function is that it takes care of NaN's and unknown (future) values.#####
    #####  Since it produces an unused second column it can be used in sklearn's Pipelines.    #####
    #####  But for that you need to add a drop_second_col() function to this My_LabelEncoder_Pipe ## 
    #####  and then feed the whole pipeline to a Column_Transformer function. It is very easy. #####
    ##################### This is the BEST working version - don't mess with it!! ##################
    ################################################################################################
    Usage in pipelines:
          le = My_LabelEncoder_Pipe()
          le.fit_transform(train[column]) ## this will give you two columns - beware!
          le.transform(test[column]) ### this will give you two columns - beware!
              
    Usage in Column Transformers:
        def drop_second_col(Xt):
        ### This deletes the 2nd column. Hence col number=1 and axis=1 ###
        return np.delete(Xt, 1, 1)
        
        drop_second_col_func = FunctionTransformer(drop_second_col)
        
        le_one = make_pipeline(le, drop_second_col_func)
    
        ct = make_column_transformer(
            (le_one, catvars[0]),
            (le_one, catvars[1]),
            (imp, numvars),
            remainder=remainder)    

    """
    def __init__(self):
        self.transformer = defaultdict(str)
        self.inverse_transformer = defaultdict(str)
        self.max_val = 0
        
    def fit(self,testx, y=None):
        ### Do not change this since Rare class combiner requires this test ##
        if isinstance(testx, tuple):
            y = testx[1]
            testx = testx[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            #### There is no way to transform dataframes since you will get a nested renamer error if you try ###
            ### But if it is a one-dimensional dataframe, convert it into a Series
                
            if testx.shape[1] == 1:
                testx = pd.Series(testx.values.ravel(),name=testx.columns[0])
            else:
                #### Since it is multi-dimensional, So in this case, just return the data as is
                return self
        ins = np.unique(testx.factorize()[1]).tolist()
        outs = np.unique(testx.factorize()[0]).tolist()
        #ins = testx.value_counts(dropna=False).index        
        if -1 in outs:
        #   it already has nan if -1 is in outs. No need to add it.
            if not np.nan in ins:
                ins.insert(0,np.nan)
        self.transformer = dict(zip(ins,outs))
        self.inverse_transformer = dict(zip(outs,ins))
        return self

    def transform(self, testx, y=None):
        ### Do not change this since Rare class combiner requires this test ##
        if isinstance(testx, tuple):
            y = testx[1]
            testx = testx[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            #### There is no way to transform dataframes since you will get a nested renamer error if you try ###
            ### But if it is a one-dimensional dataframe, convert it into a Series
            if testx.shape[1] == 1:
                testx = pd.Series(testx.values.ravel(),name=testx.columns[0])
            else:
                #### Since it is multi-dimensional, So in this case, just return the data as is
                return testx, y
        ### now convert the input to transformer dictionary values
        new_ins = np.unique(testx.factorize()[1]).tolist()
        missing = [x for x in new_ins if x not in self.transformer.keys()]
        if len(missing) > 0:
            for each_missing in missing:
                self.transformer[each_missing] = int(self.max_val + 1)
                self.inverse_transformer[int(self.max_val+1)] = each_missing
                self.max_val = int(self.max_val+1)
        else:
            self.max_val = np.max(list(self.transformer.values()))
        outs = testx.map(self.transformer).values
        testk = testx.map(self.transformer)
        if testx.dtype not in [np.int16, np.int32, np.int64, float, bool, object]:
            if testx.isnull().sum().sum() > 0:
                fillval = self.transformer[np.nan]
                testk = testk.cat.add_categories([fillval])
                testk = testk.fillna(fillval)
                testk = testk.astype(int)
                return testk, y
            else:
                testk = testk.astype(int)
                return testk, y
        else:
            return np.c_[outs,np.zeros(shape=outs.shape)].astype(int)

    def inverse_transform(self, testx, y=None):
        ### now convert the input to transformer dictionary values
        if isinstance(testx, pd.Series):
            outs = testx.map(self.inverse_transformer).values
        elif isinstance(testx, np.ndarray):
            outs = pd.Series(testx).map(self.inverse_transformer).values
        else:
            outs = testx[:]
        return outs
#################################################################################
class Groupby_Aggregator(BaseEstimator, TransformerMixin):
    """
    #################################################################################################
    ######  This Groupby_Aggregator Class works just like any Transformer in sklearn  ###############
    #####  You can add any groupby features based on categorical columns in a data frame  ###########
    ###  The list of numeric features grouped by each categorical must be given as input ############
    ### You cannot use it in sklearn pipelines but can use it stand-alone to create features. #######
    #####  It uses the same fit() and fit_transform() methods of sklearn's LabelEncoder class.  #####
    #################################################################################################
    ###   This function is a very fast function that will iteratively compute aggregates for numerics
    ###   It returns original dataframe with added features using numeric variables aggregated
    ###   What are aggregate? aggregates can be "count, "mean", "median", "mode", "min", "max", etc.
    ###   What do we aggregrate? all numeric columns in your data
    ###   What do we groupby? categorical columns which are usually object or string varaiables.
    ###   Make sure to select best features afterwards using FE_remove_variables_using_SULOV_method.
    #################################################################################################
    ### Inputs:
    ###   dft: Just sent in the data frame df that you want features added to
    ###   agg_types: list of computational types: 'mean','median','count', 'max', 'min', 'sum', etc.
    ###         One caveat: these agg_types must be found in the following agg_func of numpy 
    ###                    or pandas groupby statements.
    ###         List of aggregates available: {'count','sum','mean','mad','median','min','max',
    ###               'mode','abs', 'prod','std','var','sem','skew','kurt',
    ###                'quantile','cumsum','cumprod','cummax','cummin'}
    ###   categoricals: columns to groupby all the numeric features and compute aggregates by.
    ###   numerics: columns that will be grouped by categoricals above using aggregate types.
    ### Outputs:
    ###     dataframe: The same input dataframe with additional features created by this function.
    #################################################################################################
    Usage:
        MGB = Groupby_Aggregator(categoricals=catcols,aggregates=['mean','skew'], numerics=numerics)
        trainx = MGB.fit_transform(train)
        testx = MGB.transform(test)
    """
    def __init__(self, categoricals=[], aggregates=[], numerics='all'):
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self.transformers =  defaultdict(str)
        self.categoricals = categoricals
        self.agg_types = aggregates
        self.numerics = numerics
        self.train_cols = defaultdict(str)
        self.func_set = {'count','sum','mean','mad','median','min','max','mode',
                        'std','var','sem', 'skew','kurt','abs', 'prod',
                        'quantile','cumsum','cumprod','cummax','cummin'}
        
    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"categoricals": self.categoricals, "aggregates": self.agg_types,
                    "numerics": self.numerics}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, **fit_params):
        """Fit the model according to the given training data"""
        try:
            print('Beware: Potentially creates %d features (some will be dropped due to zero variance)' %(
                len(self.numerics)*len(self.categoricals)*len(self.agg_types)))
        except Exception as e:
            print('Erroring due to %s' %e)
        ##### First make a copy of dataframe ###
        dft_index = X.index
        dft = copy.deepcopy(X)
        # transformers are designed to modify X which must be multi-dimensional
        if isinstance(X, pd.Series) or isinstance(X, np.ndarray):
            print('Data cannot be a numpy array or a pandas Series. Must be dataframe!')
            return X
        if isinstance(self.categoricals, str):
            self.categoricals = [self.categoricals]
        if isinstance(self.numerics, str):
            if self.numerics != 'all':
                self.numerics = [self.numerics]
        ### Make sure the list of functions they send in are acceptable functions ##
        
        ls = X.select_dtypes('number').columns.tolist()
        if self.numerics == 'all':
            self.numerics = copy.deepcopy(ls)
        ### Make sure that the numerics are numeric variables! ##
        #self.numerics = list(set(self.numerics).intersection(ls))
        ### Make sure that the aggregate functions are real aggregators! ##
        self.agg_types = list(set(self.agg_types).intersection(self.func_set))
        copy_cats = copy.deepcopy(self.categoricals)
        #### if categoricals is already a list, then start transforming ###
        for i, each_catvar in enumerate(copy_cats):
            try:
                dft_cont = X[self.numerics+[each_catvar]]
            except:
                print('    %s columns given not found in data. Please correct your input.' %self.numerics)
                return X
            ### Then find the unique categories in the column ###
            try:
                #### This is where we create the aggregated features ########
                dft_full = dft_cont.groupby(each_catvar).agg(self.agg_types)
                cols =  [a +'_by_'+ str(each_catvar) +'_'+ b for (a,b) in dft_full.columns]
                dft_full.columns = cols
            except:
                print('    Error: There are no unique categories in %s column. Skipping it...###' %each_catvar)
                self.categoricals.remove(each_catvar)
                continue            
            # make sure there are no zero-variance cols. If so, drop them #
            #### drop zero variance cols the first time
            copy_cols = copy.deepcopy(cols)
            orig_shape = dft_full.shape[1]
            for each_col in copy_cols:
                if dft_full[each_col].var() == 0:
                    dft_full = dft_full.drop(each_col, axis=1)
            num_cols_dropped = dft_full.shape[1] - orig_shape
            num_cols_created  = orig_shape - num_cols_dropped
            print('    %d features grouped by %s for aggregates %s' %(num_cols_created,
                                each_catvar, self.agg_types))
            self.train_cols[each_catvar] = dft_full.columns.tolist()
            self.transformers[each_catvar] = dft_full.reset_index()
            
        return self
    
    def transform(self, X, **fit_params):
        for i, each_catvar in enumerate(self.categoricals):
            if len(self.train_cols[each_catvar]) == 0:
                ## skip this variable if it has no transformed variables
                continue
            else:
                ### now combine the aggregated variables with given dataset ###
                dft_full = self.transformers[each_catvar]
                ### simply fill in the missing values with the word "missing" ##
                ### Remember that fillna only works at the dataframe level!
                dft_full = dft_full.fillna(0)
                try:
                    X = pd.merge(X, dft_full, on=each_catvar, how='left')
                except:
                    for each_col in dft_full.columns.tolist():
                        X[each_col] = 0.0
                    print('    Erroring on creating aggregate vars for %s. Continuing...' %each_catvar)
                    continue
        ### once all columns have been transferred return the dataframe ##
        return X

    def fit_transform(self, X, **fit_params):
        ### Since X for yT in a pipeline is sent as X, we need to switch X and y this way ##
        self.fit(X)
        for i, each_catvar in enumerate(self.categoricals):
            if len(self.train_cols[each_catvar]) == 0:
                ## skip this variable if it has no transformed variables
                continue
            else:
                ### now combine the aggregated variables with given dataset ###
                dft_full = self.transformers[each_catvar]
                ### simply fill in the missing values with the word "missing" ##
                ### Remember that fillna only works at the dataframe level!
                dft_full = dft_full.fillna(0)
                X = pd.merge(X, dft_full, on=each_catvar, how='left')
        ### once all columns have been transferred return the dataframe ##
        return X

    def inverse_transform(self, X, **fit_params):
        ### One problem with this approach is that you have combined categories into one.
        ###   You cannot uncombine them since they no longer have a unique category. 
        ###   You will get back the last transformed category when you inverse transform it.
        print('There is no inverse transform for this aggregator...')
        return X
    
    def predict(self, X, **fit_params):
        #print('There is no predict function in Rare class combiner. Returning...')
        return X
###################################################################################
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from collections import defaultdict
import pdb
import copy
from sklearn.base import TransformerMixin
from collections import defaultdict
class Ranking_Aggregator(BaseEstimator, TransformerMixin):
    """
    #################################################################################################
    ######  This Ranking_Aggregator Class works just like any Transformer in sklearn  ###############
    #####  You can rank any ID column based on categorical columns in data. Why is it needed?  ######
    ###  If you have a patient in hospital then ranking them by city, state or illness is needed ####
    #####  It uses the same fit() and fit_transform() methods of sklearn's LabelEncoder class.  #####
    ### But you cannot use it in sklearn pipelines since they are more rigit in creating features ###
    #################################################################################################
    ###   This function is a very fast function that will iteratively compute rankings for ID vars ##
    ###   It returns original dataframe with added features using ID variables ranked by cat vars ### 
    ###   What are aggregates? aggregates can be "count, "mean", "median", "mode", "min", "max", etc.
    ###   What do we aggregrate? all numeric columns in your data
    ###   What do we Rank? ID variables which are usually object or string varaiables.
    ###   Make sure to select uncorrelated features afterwards using FE_remove_variables_using_SULOV_method.
    #################################################################################################
    ### Inputs:
    ###   dft: Just sent in the data frame df that you want features added to
    ###   agg_types: list of computational types: 'mean','median','count', 'max', 'min', 'sum', etc.
    ###         One caveat: these agg_types must be found in the following agg_func of numpy 
    ###                    or pandas groupby statements.
    ###         List of aggregates available: {'count','sum','mean','mad','median','min','max',
    ###               'mode','abs', 'prod','std','var','sem','skew','kurt',
    ###                'quantile','cumsum','cumprod','cummax','cummin'}
    ###   categoricals: columns to groupby all the numeric features and compute aggregates by.
    ###   idvars: columns that will ranked by categoricals above using aggregate types.
    ### Outputs:
    ###     dataframe: The same input dataframe with additional features created by this function.
    #################################################################################################
    Usage:
        MGB = Ranking_Aggregator(categoricals=catcols,aggregates=['mean','skew'], idvars=idvars)
        trainx = MGB.fit_transform(train)
        testx = MGB.transform(test)
    """
    def __init__(self, categoricals=[], aggregates=[], idvars=''):
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self.transformers =  defaultdict(str)
        self.categoricals = categoricals
        self.agg_types = aggregates
        self.idvars = idvars
        self.train_cols = defaultdict(str)
        self.func_set = {'average', 'min', 'max', 'dense', 'first'}
        ### ‘first’ is not allowed for non-numeric variables ##
        
    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"categoricals": self.categoricals, "aggregates": self.agg_types,
                    "idvars": self.idvars}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, **fit_params):
        """Fit the model according to the given training data"""
        try:
            print('Beware: Potentially creates %d features (some will be dropped due to zero variance)' %(
                len(self.categoricals)*len(self.agg_types)))
        except Exception as e:
            print('Erroring due to %s' %e)
        ##### First make a copy of dataframe ###
        dft_index = X.index
        dft = copy.deepcopy(X)
        # transformers are designed to modify X which must be multi-dimensional
        if isinstance(X, pd.Series) or isinstance(X, np.ndarray):
            print('Data cannot be a numpy array or a pandas Series. Must be dataframe!')
            return X
        if isinstance(self.categoricals, str):
            self.categoricals = [self.categoricals]
        if isinstance(self.idvars, str):
            if self.idvars == 'all':
                nunique_train = X.nunique().reset_index()
                nunique_min = 0.20
                nunique_max = 0.4
                ID_limit_min = max(10, int(nunique_min*(len(X)))) ### X% of rows must be unique for it to be called ID
                ID_limit_max = max(10, int(nunique_max*(len(X)))) ### X% of rows must be unique for it to be called ID
                ls = nunique_train[(nunique_train[0]<=ID_limit_max) & (nunique_train[0]>=ID_limit_min) ]['index'].tolist()
                if len(ls) > 0:
                    print('    Using first one from %s vars as ID vars since all option was chosen.' %ls)
                    self.idvars = ls[0]
                else:
                    print('    No ID vars found that metet criteria of being %s-%s nuniques of dataset length. Returning' %(nunique_min, nunique_max))
                    return self
            else:
                print('    %s ID variable chosen...' %self.idvars)
        elif isinstance(self.idvars, list):
            print('    only one ID variable can be chosen at a time. Choosing first one from list %s' %self.idvars)
            self.idvars = self.idvars[0]
        else:
            print('    %s ID vars unrecognized. Please check your input and try again.' %self.idvars)
            return self
        ### Make sure the list of functions they send in are acceptable functions ##
        ### Make sure that the aggregate functions are real aggregators! ##
        self.agg_types = list(set(self.agg_types).intersection(self.func_set))
        dft_temp = dft[self.idvars]
        ### Check if non-numeric dtype is used in dataset for ranking ##
        if isinstance(dft_temp, pd.Series):
            if not dft_temp.dtype.kind in 'biufc':
                print('    "first" aggregate type not allowed in non-numeric columns')
                if 'first' in self.agg_types:
                    self.agg_types.remove('first')
        else:
            for col in dft_temp.columns:
                if not dft_temp[col].dtype.kind in 'biufc':
                    print('    "first" aggregate type not allowed in non-numeric columns')
                    if 'first' in self.agg_types:
                        self.agg_types.remove('first')
        copy_cats = copy.deepcopy(self.categoricals)
        #### if categoricals is already a list, then start transforming ###
        for i, each_catvar in enumerate(copy_cats):
            cols_added = []
            group_list = [self.idvars, each_catvar]

            ### Then find the unique categories in the column ###
            
            for each_type in self.agg_types:
                new_col =  str(self.idvars) + '_ranked_by_'+ str(each_catvar) + '_' + each_type
                try:
                    df_temp = dft.groupby(group_list)[self.idvars].rank(method=each_type,ascending=True)
                    if df_temp.nunique() > 1:
                        dft[new_col] = df_temp.values
                        cols_added.append(new_col)
                        continue
                except:
                    print('Error trying to add new aggregate column for %s by %s' %(each_catvar, each_type))
            
            # make sure there are no zero-variance cols. If so, drop them #
            
            if len(cols_added) > 0:
                copy_cols = copy.deepcopy(cols_added)
                dft_full = pd.DataFrame()
                dft_full = dft[[self.idvars,each_catvar]+cols_added].drop_duplicates(subset=[self.idvars,each_catvar],keep='first')
                print('    %s columns added for %s' %(len(cols_added), each_catvar))
                self.train_cols[each_catvar] = cols_added
                self.transformers[each_catvar] = dft_full
            else:
                print('No columns added for %s. Continuing...' %each_catvar)
                continue
                
            del dft_full
            del df_temp
            
        return self
    
    def transform(self, X, **fit_params):
        for i, each_catvar in enumerate(self.categoricals):
            
            if len(self.train_cols[each_catvar]) == 0:
                ## skip this variable if it has no transformed variables
                continue
            else:
                if each_catvar in self.train_cols.keys():
                    dft_full = pd.DataFrame()
                    ### now combine the aggregated variables with given dataset ###
                    cols_added = self.train_cols[each_catvar]
                    dft_full = self.transformers[each_catvar]
                    ### simply fill in the missing values with the word "0" ##
                    ### Remember that fillna only works at the dataframe level!
                    try:
                        X = pd.merge(X, dft_full, on=[self.idvars,each_catvar], how='left')
                        X[cols_added].fillna(0, inplace=True)
                    except:
                        for each_col in cols_added:
                            X[each_col] = 0.0
                        print('    Erroring on creating aggregate vars for %s. Continuing...' %each_catvar)
                        continue
        ### once all columns have been transferred return the dataframe ##
        return X

    def fit_transform(self, X, **fit_params):
        X = copy.deepcopy(X)
        ### Since X for yT in a pipeline is sent as X, we need to switch X and y this way ##
        self.fit(X)
        for i, each_catvar in enumerate(self.categoricals):
            if len(self.train_cols[each_catvar]) == 0:
                ## skip this variable if it has no transformed variables
                continue
            else:
                if each_catvar in self.train_cols.keys():
                    dft_full = pd.DataFrame()
                    cols_added = self.train_cols[each_catvar]
                    ### now combine the aggregated variables with given dataset ###
                    dft_full = self.transformers[each_catvar]
                    ### simply fill in the missing values with the word "0" ##
                    ### Remember that fillna only works at the dataframe level!
                    if len(cols_added) > 0:
                        X = pd.merge(X, dft_full, on=[self.idvars,each_catvar], how='left')
                        X[cols_added].fillna(0, inplace=True)
        ### once all columns have been transferred return the dataframe ##
        return X

    def inverse_transform(self, X, **fit_params):
        ### One problem with this approach is that you have combined categories into one.
        ###   You cannot uncombine them since they no longer have a unique category. 
        ###   You will get back the last transformed category when you inverse transform it.
        print('There is no inverse transform for this aggregator...')
        return X
    
    def predict(self, X, **fit_params):
        #print('There is no predict function in Rare class combiner. Returning...')
        return X
###################################################################################
import copy
class DateTime_Transformer(BaseEstimator, TransformerMixin):
    """
    ################################################################################################
    ######     The DateTime_Transformer class works just like sklearn's Transformers but better! ###
    #####  It creates new features out of any date-time var in your dataset. It also handles NaN's##
    ##  The beauty of this function is that it takes care of NaN's and a variety of date formats.###
    ##################### This is the BEST working version - don't mess with it!! ##################
    ################################################################################################
    Usage:
          ds = DateTime_Transformer(ts_column=col)
          train = ds.fit_transform(train) ## this will give your transformed values as a dataframe
          test = ds.transform(test) ### this will give your transformed values as a dataframe
              
    Usage in Column Transformers and Pipelines:
          No. It cannot be used in pipelines since it need to produce two columns for the next stage in pipeline.
          See if you can change it to fit an sklearn pipeline on your own. I'd be curious to know how.
    """
    def __init__(self, ts_column, verbose=0):
        self.ts_column = ts_column
        self.verbose = verbose
        self.cols_added = []
        self.fitted = False
        self.X_transformed = None
        self.train = False
        
    def fit(self, X, y=None):
        X = copy.deepcopy(X)
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, pd.Series):
            print('    X must be dataframe. Converting it to a pd.DataFrame.')
            X = pd.DataFrame(X.values, columns=[X.name])
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            print('    X must be dataframe. Converting it to a pd.DataFrame.')
        else:
            #### There is no way to transform dataframes since you will get a nested renamer error if you try ###
            ### But if it is a one-dimensional dataframe, convert it into a Series
            #print('    X is a DataFrame...')
            pass
        X_trans, self.cols_added = FE_create_time_series_features(X, self.ts_column, ts_adds_in=[], verbose=self.verbose)
        self.fitted = True
        self.train = True
        self.X_transformed = X_trans
        return self
    
    def transform(self, X, y=None):
        X = copy.deepcopy(X)
        if self.fitted and self.train:
            self.train = False
            return self.X_transformed
        else:
            self.fit(X)
            return self.X_transformed 
        
    def fit_transform(self, X, y=None):
        X = copy.deepcopy(X)
        self.fit(X)
        self.train = False
        return self.X_transformed
######################################################################################################
import copy
def _create_ts_features(df, tscol, verbose=0):
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
        df[[name_col]] = df[[name_col]].fillna(0)
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
    if verbose:
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
def FE_create_time_series_features(dft, ts_column, ts_adds_in=[], verbose=0):
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
    dtf: The original pandas dataframe with new fields created by splitting date-time field
    rem_ts_cols: List of added variables as output. This will be useful for future ts_adds_in
                 This list of columns is useful for matching test with train dataframes.
    ######################################################################################
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
            if verbose:
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
            if verbose:
                print('        dropping %s column after time series done' %ts_column)
            dtf = dtf.drop(ts_column, axis=1)
        else:
            pass
        if verbose:
            print('    After dropping some zero variance cols, shape of data: %s' %(dtf.shape,))
    except Exception as e:
        print('Error in Processing %s column due to %s for date time features. Continuing...' %(ts_column, e))
    return dtf, rem_ts_cols
######################################################################################
from pandas.api.types import is_numeric_dtype
#gives fit_transform method for free
from sklearn.base import BaseEstimator, TransformerMixin 
import copy
import pdb
class Binning_Transformer(BaseEstimator, TransformerMixin):
    """
        ######   This is where we do ENTROPY BINNING OF CONTINUOUS VARS ###########
        #### Best to do binning by using Target variables: that's why we use DT's
        #### Make sure your input is pandas Series or DataFrame with all NUMERICS.
        #### Otherwise Binning canot be done. This transformer ensures you get the
        #### Best Results by generalizing using Regressors and Classifiers.
        ############################################################################
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        ### This is where we set the max depth for setting defaults for clf ##
        self.new_bincols = {}
        self.entropy_threshold = {}
        self.fitted = False
        self.clfs = {}
        self.max_number_of_classes = 1
    
    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"verbose": self.verbose}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def num_classes(self, y):
        """
        ### Returns number of classes in y
        """
        from collections import defaultdict
        from collections import OrderedDict
        y = copy.deepcopy(y)
        if isinstance(y, np.ndarray):
            ls = pd.Series(y).nunique()
        else:
            if isinstance(y, pd.Series):
                ls = y.nunique()
            else:
                if len(y.columns) >= 2:
                    ls = OrderedDict()
                    for each_i in y.columns:
                        ls[each_i] = y[each_i].nunique()
                    return ls
                else:
                    ls = y.nunique()[0]
        return ls

    
    def fit(self, X, y, **fit_params):
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        X = copy.deepcopy(X)
        if isinstance(y, pd.DataFrame):
            if len(y.columns) >= 2:
                number_of_classes = self.num_classes(y)
                for each_i in y.columns:
                    number_of_classes[each_i] = int(number_of_classes[each_i] - 1)
                max_number_of_classes = np.max(list(number_of_classes.values()))
            else:
                number_of_classes = int(self.num_classes(y) - 1)
                max_number_of_classes = np.max(number_of_classes)
        else:
            number_of_classes = int(self.num_classes(y) - 1)
            max_number_of_classes = np.max(number_of_classes)
        self.max_number_of_classes = max_number_of_classes
        seed = 99
        if isinstance(X, np.ndarray):
            print('    X cannot be numpy array. It must be either pandas Series or DataFrame!')
            return self
        elif isinstance(X, pd.Series):
            self.continuous_vars = [X.name]
            X = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            self.continuous_vars = X.columns.tolist()
        else:
            print('Input seems to be of unknown data type. Returning...')
            return self
        ####### This is where we bin each variable through a method known as Entropy Binning ##############
        X = X.fillna(-999)
        for each_num in self.continuous_vars:
            ###   This is an Awesome Entropy Based Binning Method for Continuous Variables ###########
            max_depth = max(2, int(np.log10(X[each_num].max()-X[each_num].min())))
            if is_numeric_dtype(y) and self.max_number_of_classes > 25:
                clf = DecisionTreeRegressor(criterion='mse',min_samples_leaf=2,
                                            max_depth=max_depth,
                                            random_state=seed)
            else:
                clf = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=2,
                                                 max_depth=max_depth,
                                                 random_state=seed)
            try:
                clf.fit(X[each_num].values.reshape(-1,1), y)
                ranges = clf.tree_.threshold[clf.tree_.threshold>-2].tolist()
                ranges.append(np.inf)
                ranges.insert(0, -np.inf)
                self.entropy_threshold[each_num] = np.sort(ranges)        
                self.new_bincols[each_num] = None
                self.clfs[each_num] = clf
                if self.verbose:
                    print('    %d bins created for %s...' %((len(ranges)-1), each_num))
            except:
                self.entropy_threshold[each_num] =  None
                print('Skipping %s column for Entropy Binning due to Error. Check your input and try again' %each_num)
        self.fitted = True
        return self
    
    def transform(self, X, y=None, **fit_params):
        X = copy.deepcopy(X)
        ####### This is where we bin each variable through a method known as Entropy Binning ##############
        for each_num in self.continuous_vars:
            if isinstance(X, pd.Series):
                X = pd.DataFrame(X)
                entropy_threshold = self.entropy_threshold[each_num]
                if entropy_threshold is None:
                    print('skipping binning since there are no bins available for %s' %each_num)
                    continue
                else:
                    try:
                        X[each_num] = np.digitize(X[each_num].values, entropy_threshold)
                        #### We Drop the original continuous variable after you have created the bin when Flag is true
                        self.new_bincols[each_num] = X[each_num].nunique()
                    except:
                        print('Error in %s during Entropy Binning' %each_num)
        return X.values, y
    
    def fit_transform(self, X, y=None, **fit_params):
        X = copy.deepcopy(X)
        self.fit(X, y)
        self.fitted = True
        X_transformed, _ = self.transform(X, y)
        return X_transformed, y    
################################################################################################
from pandas.api.types import is_numeric_dtype, is_integer_dtype
from pandas.api.types import is_datetime64_any_dtype
#gives fit_transform method for free
from sklearn.base import BaseEstimator, TransformerMixin 
from lightgbm import LGBMRegressor
import copy
#from lazytransform import LazyTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from tqdm import tqdm
from sklearn.metrics import r2_score
from collections import defaultdict
import dask
import pdb
class TSLagging_Transformer(BaseEstimator, TransformerMixin):
    """
        ###### T I M E  S E R I E S  L A G G I N G   T R A N S F O R M E R  ############
        ######   This is where we add Lags of Targets to Time Series data ##############
        #### In a time series problems to predict sales, we need to add last week's, ###
        #### last month's, last year's sales data as features to build a model.      ###
        #### Otherwise the model will not be able to learn how to predict future sales #
        #### from past sales. This is a very important feature engineering technique. ##
        ################################################################################
        Inputs:
        ----------------
        X: a dataframe with a time series (pandas date-time variable type) column in it.
        namevars: columns that you want to lag in the data frame. Other columns will be untouched.
        y: target variable(s) you intend to lag. It can be multi_label also
        n_in: Number of lag periods as input (X).
        n_out: Number of future periods (optional) as output for the taget variable (y).
        dropT: Boolean - whether or not to drop columns at time 't'.
        
        Outputs:
        -----------------
        X: This is the transformed data frame with lagged targets added.
    """
    def __init__(self, lags, date_column, hier_vars='', verbose=0):
        ## Not more than 3 lagged values allowed ##
        self.lags = max(3, lags)
        if isinstance(date_column, list):
            print('Only one date column accepted. Taking the first from the list of columns given: %s' %date_column[0])
            self.date_col = date_column[0]
        else:
            self.date_col = date_column
        if isinstance(hier_vars, list):
            if len(hier_vars) == 0:
                print('No hierarchical vars given. Continuing without but results may not be accurate...')
                self.hier_vars = []
            else:
                self.hier_vars = hier_vars
        elif isinstance(hier_vars, str):
            if hier_vars == '':
                print('No hierarchical vars given. Continuing without but results may not be accurate...')
                self.hier_vars = []
            else:
                self.hier_vars = [hier_vars]
        else:
            print('hier_vars must be a string or a list. Returning')
            return
        self.verbose = verbose
        ### This is where we find out whether fit or fit_transform is done ##
        self.X_prior = None
        self.y_prior = None
        self.fitted = False
        ### If ts_column is not a string column, then set its format to an empty string ##
        self.str_format = ''
        self.targets = []
        self.col_adds = defaultdict(list)
        self.X_adds = None
    
    def get_params(self, deep=True):
        # This is to make it scikit-learn compatible ####
        return {"lags": self.lags, "date_column": self.date_col, "verbose": self.verbose}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_names_of_targets(self, y):
        """
        ### Returns names of target variables in y if it is multi-label.
        """
        from collections import defaultdict
        from collections import OrderedDict
        y = copy.deepcopy(y)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, columns=['target'])
            targets = ['target']
        if isinstance(y, pd.Series):
            if y.name is None:
                targets = ['target']
                y = pd.DataFrame(y, columns=targets, index=y.index)
            else:
                targets = [y.name]
                y = pd.DataFrame(y, columns=targets, index=y.index)
        elif isinstance(y, pd.DataFrame):
            targets = y.columns.tolist()
        self.targets = targets
        return y
        
    def convert_X_to_datetime(self, X):
        """
        This utility checks if the date-time column is actually in X.
        Then it checks if the date time column is really a pandas date-time column.
        Then it converts X to a pandas dataframe so it is easier to handle in future processing.
        """
        X = copy.deepcopy(X)
        if isinstance(X, np.ndarray):
            print('    X cannot be a numpy array. It must be either a pandas Series or a DataFrame. Returning')
            return X
        elif isinstance(X, pd.Series):
            if self.date_col == X.name:
                if is_datetime64_any_dtype(X):
                    self.date_col = X.name
                    X = pd.DataFrame(X)
                else:
                    if self.verbose:
                        print('Converting %s into pandas date-time type...' %self.date_col)
                    X = pd.DataFrame(X)
                    self.date_col = X.columns.tolist()[0]
            else:
                print('%s is not found in X. Check your input and try again' %self.date_col)
                return X
        elif isinstance(X, pd.DataFrame):
            if self.date_col in X.columns.tolist():
                if not is_datetime64_any_dtype(X[self.date_col]):
                    if self.verbose:
                        print('Converting %s into pandas date-time type...' %self.date_col)
            else:
                print('%s is not found in X. Check your input and try again' %self.date_col)
                return X
        #### This is where you convert the column to a date-time column ###
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            try:
                ####### This is where fill each date-column in case it is empty ##
                X[[self.date_col]] = X[[self.date_col]].fillna(method='ffill')
                X[[self.date_col]] = X[[self.date_col]].fillna(method='bfill')
                ## Check if it has an index or a column with the name of train time series column ####
                if is_datetime64_any_dtype(X[self.date_col]):
                    if self.verbose:
                        print('    since %s is already datetime column, continuing...' %self.date_col)
                    self.str_format = ''
                else:
                    ### since date_col is a string you need to use values ###
                    str_first_value = X[self.date_col].values[0]
                    str_values = X[self.date_col].values[:12] ### we want to test a big sample of them 
                    if isinstance(str_first_value, str):
                        ### if it is an object column, convert ts_column into datetime and then set as index
                        str_format = infer_date_time_format(str_values)
                        if str_format:
                            ### if there is a format for that date column, then use it!
                            str_format = str_format[0]
                            X[self.date_col] = pd.to_datetime(X[self.date_col], format=str_format)
                        else:
                            ### since there is no format to the date column, leave it out
                            X[self.date_col] = pd.to_datetime(X[self.date_col])
                    elif type(str_first_value) in [np.int8, np.int16, np.int32, np.int64]:
                        ### if it is an integer column, convert ts_column into datetime and then set as index
                        X[self.date_col] = pd.to_datetime(X[self.date_col])
                    else:
                        print('    Type of time series column %s is float or unknown. Must be string or datetime. Please check input and try again.' %self.date_col)
                        return 
                    self.str_format = str_format
            except Exception as e:
                print('    Error: Converting time series column %s into ts index due to %s. Please check input and try again.' %(self.date_col, e))
                return X
        elif type(X) == dask.dataframe.core.DataFrame:
            str_format = ''
            if self.date_col in X.columns:
                print('    %s column exists in dask data frame...' %self.date_col)
                str_first_value = X[self.date_col].compute()[0]
                X.index = dd.to_datetime(X[self.date_col].compute())
                X[self.date_col] = dd.to_datetime(X[self.date_col].compute())
            else:
                if self.verbose:
                    print(f"    Error: Unable to detect column (or index) called '{self.date_col}' in dataset. Continuing...")
                return X
        else:
            if self.verbose:
                print('    Unable to detect type of input data X. Please check your input and try again')                        
            return X
        return X

    def change_datecolumn_to_index(self, X):
        X = copy.deepcopy(X)
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            try:
                ############### Check if it has an index or a column with the name of train time series column ####
                if is_datetime64_any_dtype(X[self.date_col]):
                    ### if it is a datetime column, then set it as index
                    ### if it a datetime index, then just set the index as is 
                    ts_index = X.pop(self.date_col)
                    X.index = ts_index
                else:
                    if self.str_format:
                        ### if there is a format for that date column, then use it!
                        if isinstance(self.str_format, list):
                            str_format = self.str_format[0]
                        else:
                            str_format = self.str_format
                        ts_index = pd.to_datetime(X.pop(self.date_col), format=str_format)
                    else:
                        ### since there is no format to the date column, leave it out
                        ts_index = pd.to_datetime(X.pop(self.date_col))
                    X.index = ts_index
            except Exception as e:
                print('    Error: Converting time series column %s into ts index due to %s. Please check input and try again.' %(self.date_col, e))
        elif type(X) == dask.dataframe.core.DataFrame:
            if self.date_col in X.columns:
                print('    %s column exists in dask data frame...' %self.date_col)
                str_first_value = X[self.date_col].compute()[0]
                X.index = dd.to_datetime(X[self.date_col].compute())
                X = X.drop(self.date_col, axis=1)
            else:
                print(f"    Error: input X must have a column (or index) called '{self.date_col}'.")
        else:
            print('    Unable to detect type of input data X. Please check your input and try again')                        
        return X
    
    def self_imputer(self, X_train, y_train, X_test):
        X_train = copy.deepcopy(X_train)
        X_test = copy.deepcopy(X_test)
        y_train = copy.deepcopy(y_train)
        X_index = X_test.index
        import time
        cols = y_train.columns.tolist()
        preds = []
        int_changes = False
        hier_vars = self.hier_vars
        for i, column in enumerate(cols):
            start_time = time.time()
            if self.verbose:
                print('Using a self imputer based on group means for imputing missing values in %s...' %column)
            ### Since X_train might have some NaN's ##
            X_train_new = X_train[(y_train[column].isna()==False)]
            y_train_new = y_train[(y_train[column].isna()==False)]
            int_changes = is_integer_dtype(y_train)
            X_y_combined  = pd.concat([X_train_new, y_train_new[column]], axis=1)
            season_vars = ['date_month','date_dayofmonth']
            ### Sometimes, hier_vars such as store and item are not found in train and test same!
            no_hier_vars = False
            if len(self.hier_vars) == 0:
                no_hier_vars = True
            if not no_hier_vars > 0:
                ### Suppose there are hier_vars then continue using them here ##
                dfx = pd.merge(X_test, X_y_combined, on=season_vars+hier_vars,how='left')
                dfx = dfx[[column]+season_vars+hier_vars]
                dfxd = dfx.drop_duplicates(subset=season_vars+hier_vars, keep='last')
                ### Suppose there are hier_vars but they don't work well, then reset it ##
                if dfxd[column].isnull().all():
                    no_hier_vars = True
            ### Do not change the next line because there is a reason for it! ####
            if no_hier_vars:
                ### This is a simpler version in case NaN's happen due to lack of no hier_vars categories in test data
                X_test1 = X_test[season_vars].astype(np.int8)
                X_y_combined = X_y_combined[season_vars+[column]]
                X_y_combined[season_vars] = X_y_combined[season_vars].astype(np.int8)
                ### If there is no column value due to this merge, then use a simpler merge!
                dfx = pd.merge(X_test1, X_y_combined, on=season_vars,how='left')
                del X_test1
                del X_y_combined
                dfx = dfx[[column]+season_vars]
                dfxd = dfx.drop_duplicates(subset=season_vars, keep='last')
                if dfxd[column].isnull().all():
                    if self.verbose:
                        print('    There are NaNs in data when trying to create lagged vars. Fix the NaNs and try again.')
                    return np.zeros((X_test.shape[0], y_train_new.shape[1]))
                X_test1 = pd.merge(X_test, dfxd,on=season_vars, how='left')
                pred = X_test1[column].values
                del X_test1
                del dfx
                del dfxd
            else:
                ####### if hier vars exist, then use this #########
                dfxgroup_all = dfx.groupby(hier_vars).mean().reset_index()[hier_vars+[column]]
                X_test1 = pd.merge(X_test,dfxgroup_all,on=hier_vars, how='left')
                X_test2 = pd.merge(X_test1, dfxd, on=season_vars+hier_vars,how='left')
                ### delete useless variables ###
                del X_test1
                del X_test2
                del X_y_combined
                del dfx
                del dfxd
            ### Make sure that whatever vars came in as integers return back as integers!
            if int_changes:
                if self.verbose:
                    print('    ## target is integer dtype ###')
                pred = pd.Series(pred).fillna(0).values.astype(np.int32).values
            preds.append(pred)
            if self.verbose:
                print('    time taken for imputing = %0.0f seconds' %((time.time()-start_time)))
        y_preds = np.array(preds)
        ### You need to flip it to make it right shape ##
        y_preds = y_preds.T
        y_preds = pd.DataFrame(y_preds, index=X_index, columns=cols)
        return y_preds
    
    def fit(self, X, y, **fit_params):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        self.X_prior = copy.deepcopy(X)
        y = self.get_names_of_targets(y)
        y_prior = X[[self.date_col]+self.hier_vars].join(y)
        self.y_prior = y_prior.set_index(pd.to_datetime(y_prior.pop(self.date_col)))
        del y_prior
        self.fitted = True
        return self
    
    def transform(self, X, y=None, **fit_params):
        X = copy.deepcopy(X)
        change_to_imputing = False
        print('Input X shape = %s. Beginning Time Series lagging transformation...' %(X.shape,))
        if y is None:
            ### Since y is not available we have to search for prior_y and see!
            for i in range(self.lags):
                if i == 0:
                    no_of_days = 1
                elif i == 1:
                    no_of_days = 7
                elif i == 2:
                    no_of_days = 30
                rsuffix = '_days_prior_'+str(no_of_days)
                ### Adding 7 days will make the sales on a weekly basis correct such as Monday to Monday
                y_join = copy.deepcopy(self.y_prior)
                y_join.index = self.y_prior.index +  pd.Timedelta(days=+no_of_days)
                y_join = y_join.reset_index()
                X_joined = copy.deepcopy(X)
                X_joined[self.date_col] = pd.to_datetime(X_joined[self.date_col])
                X_joined = pd.merge(X_joined, y_join, on=[self.date_col]+self.hier_vars, how='left')
                del y_join
                target_select = [x for x in X_joined.columns if x in self.targets]
                for each_target in target_select:
                    cols_select = self.col_adds[each_target]
                    col_select = [x for x in cols_select if x.endswith(rsuffix)]
                    col_select = col_select[0]
                    if  X_joined[each_target].isnull().all():
                        if self.verbose:
                            print('    No prior targets available for combination of columns in test vs train')
                            print('        changing method to imputing target values for %s...' %rsuffix)
                        change_to_imputing = True
                        break
                    else:
                        if X_joined[each_target].isnull().any():
                            print('    Error: for some reason there are NaNs in %s. Hence changing to imputing method' %each_target)
                            change_to_imputing = True
                        else:
                            X[col_select] = X_joined[each_target].values
            #### don't change the next line. It is meant to check whether to do imputing ##
            if change_to_imputing:
                if self.verbose:
                    print('Imputing prior values for target using targeted group means...')
                #lazy = LazyTransformer(model=None, encoders='auto', scalers=None, 
                #                       date_to_string=False, transform_target=False,
                #                       imbalanced=False, save=False, combine_rare=False, 
                #                       verbose=0)
                col_adds = self.X_adds.columns.tolist()
                X_old = self.convert_X_to_datetime(self.X_prior)
                X_new = self.convert_X_to_datetime(X)
                ## we are going to turn the prior columns as y to predict future priors
                y_old = self.X_adds
                ## we must convert date_col into date-time features before we convert object vars
                ds = DateTime_Transformer(ts_column=self.date_col)
                X_old = ds.fit_transform(X_old) ## this will give your transformed values as a dataframe
                X_new = ds.transform(X_new) ### this will give your transformed values as a dataframe
                ## Now we can convert all object columns to numeric ######
                #X_old, _ = lazy.fit_transform(X_old, y_old)
                X_old, X_new, _ = FE_convert_all_object_columns_to_numeric(X_old, X_new)
                #X_new = lazy.transform(X_new)
                if self.verbose:
                    print('Imputing begins with input and output shapes = %s %s %s' %(X_old.shape, y_old.shape, X_new.shape,))
                y_new = self.self_imputer(X_old, y_old, X_new)
                ### This is where we combine both imputing and prior_y values ###
                for col_add in col_adds:
                    if col_add in X.columns:
                        mask_loc = X[X[col_add].isna() == True].index
                        if len(mask_loc) > 0:
                            X.iloc[mask_loc, col_add] = y_new.iloc[mask_loc, col_add]
                        else:
                            X[col_add] = y_new[col_add].values
                    else:
                        X[col_add] = y_new[col_add].values
                print('    completed with new X shape = %s' %(X.shape,))
                return X
        else:
            ### In this case, y is available and can add prior day columns ##
            y = copy.deepcopy(y)
            ### find target variables to transform ###
            y = self.get_names_of_targets(y)
            df = self.convert_X_to_datetime(X)
            df = self.change_datecolumn_to_index(df)
        int_vars  = y.select_dtypes(include='integer').columns.tolist()
        # Notice that we will create a lagged columns from name vars
        all_target_col_adds = []
        int_changes = []
        n_in = self.lags
        ### we will only add lags to target variables here ###
        namevars = self.targets
        ### You have to add max. 3 columns to X using y variable shifted by 7, 30 and 365 days ###
        for i in range(n_in):
            if i == 0:
                no_of_days = 1
            elif i == 1:
                no_of_days = 7
            elif i == 2:
                no_of_days = 30
            else:
                continue
            rsuffix = '_days_prior_'+str(no_of_days)
            for var in namevars:
                addname = var + rsuffix
                df[addname] = y[var].shift(no_of_days).values
                if var in int_vars:
                    int_changes.append(addname)
                self.col_adds[var].append(addname)
                all_target_col_adds.append(addname)
                X[addname] = df[addname].values
        # if fit_transform is done, then fitted is False since test is next ##
        self.fitted = False
        self.X_adds = X[all_target_col_adds]
        print('    completed with new X shape = %s' %(X.shape,))
        return X
    
    def fit_transform(self, X, y, **fit_params):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        self.fit(X, y)
        X_trans = self.transform(X, y)
        return X_trans

################################################################################
# THIS IS A MORE COMPLEX ALGORITHM THAT CHECKS MORE SPECIFICALLY FOR A DATE AND TIME FIELD
import datetime as dt
from datetime import datetime, date, time

### This tests if a string is date field and returns a date type object if successful and
##### a null list if it is unsuccessful
def is_date(txt):
    fmts = ('%Y-%m-%d', '%d/%m/%Y', '%d-%b-%Y', '%d/%b/%Y', '%b/%d/%Y', '%m/%d/%Y', '%b-%d-%Y', '%m-%d-%Y',
 '%Y/%m/%d', '%m/%d/%y', '%d/%m/%y', '%Y-%b-%d', '%Y-%B-%d', '%d-%m-%y', '%a, %d %b %Y', '%a, %d %b %y',
 '%d %b %Y', '%d %b %y', '%a, %d/%b/%y', '%d-%b-%y', '%m-%d-%y', '%d-%m-%Y', '%b%d%Y', '%d%b%Y',
 '%Y', '%b %d, %Y', '%B %d, %Y', '%B %d %Y', '%b %Y', '%B%Y', '%b %d,%Y')
    parsed=None
    for fmt in fmts:
        try:
            t = dt.datetime.strptime(txt, fmt)
            Year=t.year
            if Year > 2040 or Year < 1900:
                pass
            else:
                parsed = fmt
                return fmt
                break
        except ValueError as err:
            pass
    return parsed



#### This tests if a string is time field and returns a time type object if successful and
##### a null list if it is unsuccessful
def is_time(txt):
    fmts = ('%H:%M:%S.%f','%M:%S.%fZ','%Y-%m-%dT%H:%M:%S.%fZ','%h:%M:%S.%f','%-H:%M:%S.%f',
            '%H:%M','%I:%M','%H:%M:%S','%I:%M:%S','%H:%M:%S %p','%I:%M:%S %p',
           '%H:%M %p','%I:%M %p')
    parsed=None
    for fmt in fmts:
        try:
            t = dt.datetime.strptime(txt, fmt)
            parsed=fmt
            return parsed
            break
        except ValueError as err:
            pass
    return parsed

#### This tests if a string has both date and time in it. Returns a date-time object and null if it is not

def is_date_and_time(txt):
    fmts = ('%d/%m/%Y  %I:%M:%S %p', '%d/%m/%Y %I:%M:%S %p', '%d-%b-%Y %I:%M:%S %p',
 '%d/%b/%Y %I:%M:%S %p', '%b/%d/%Y %I:%M:%S %p', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ',
 '%m/%d/%Y %I:%M %p', '%m/%d/%Y %H:%M %p', '%d/%m/%Y  %I:%M:%S', '%d/%m/%Y  %H:%M', '%m/%d/%Y %H:%M',
 '%m/%d/%Y  %H:%M', '%d/%m/%Y  %I:%M', '%d/%m/%Y  %I:%M %p', '%m/%d/%Y  %I:%M', '%d/%b/%Y  %I:%M',
 '%b/%d/%Y  %I:%M', '%m/%d/%Y  %I:%M:%S', '%b-%d-%Y %I:%M:%S %p', '%m-%d-%Y %H:%M:%S %p',
 '%b-%d-%Y %H:%M:%S %p', '%m/%d/%Y %H:%M:%S %p', '%b/%d/%Y %H:%M:%S %p', '%Y-%m-%d %H:%M:%S %Z',
 '%Y-%m-%d %H:%M:%S %Z%z', '%Y-%m-%d %H:%M:%S %z', '%Y/%m/%d %H:%M:%S %Z%z', '%m/%d/%y %H:%M:%S %Z%z',
 '%d/%m/%Y %H:%M:%S %Z%z', '%m/%d/%Y %H:%M:%S %Z%z', '%d/%m/%y %H:%M:%S %Z%z', '%Y-%b-%d %H:%M:%S %Z%z',
 '%Y-%B-%d %H:%M:%S %Z%z', '%d-%b-%Y %H:%M:%S %Z%z', '%d-%m-%y %H:%M:%S %Z%z', '%Y-%m-%d %H:%M',
 '%Y-%b-%d %H:%M', '%a, %d %b %Y %T %z', '%a, %d %b %y %T %z', '%d %b %Y %T %z', '%d %b %y %T %z',
 '%d/%b/%Y %T %z', '%a, %d/%b/%y %T %z', '%d-%b-%Y %T %z', '%d-%b-%y %T %z', '%m-%d-%Y %I:%M %p',
 '%m-%d-%y %I:%M %p', '%m-%d-%Y %I:%M:%S %p', '%d-%m-%Y %H:%M:%S %p', '%m-%d-%y %H:%M:%S %p',
 '%d-%b-%Y %H:%M:%S %p', '%d-%m-%y %H:%M:%S %p', '%d-%b-%y %I:%M:%S %p', '%d-%b-%y %I:%M %p',
 '%d-%b-%Y %I:%M %p', '%d-%m-%Y %H:%M %p', '%d-%m-%y %H:%M %p', '%d/%m/%Y %H:%M:%p', '%d/%m/%Y %H:%M:%S',
 '%Y-%m-%d %H:%M:%S')
    parsed=None
    for fmt in fmts:
        try:
            t = dt.datetime.strptime(txt, fmt)
            parsed=fmt
            return parsed
            break
        except ValueError as err:
            pass
    return parsed

# FIND DATE TIME VARIABLES

# This checks if a field in general is a date or time field

def infer_date_time_format(list_dates):
    """
    This is a generic algorithm that can infer date and time formats by checking repeatedly against a list.
    Make sure you give it a list of datetime formats since there can be many formats in a list.
    You can take the first of the returned list of formats or the majority or whatever you wish.
    # THE DATE FORMATS tested so far by this algorithm are:
        # 19JAN1990
        # JAN191990
        # 19/jan/1990
        # jan/19/1990
        # Jan 19, 1990
        # January 19, 1990
        # Jan 19,1990
        # 01/19/1990
        # 01/19/90
        # 1990
        # Jan 1990
        # January1990 
        # YOU CAN ADD MORE FORMATS above IN THE "fmts" section.
    """
    
    date_time_fmts = []
    try: 
        for each_datetime in list_dates:
            date1 = is_date(each_datetime)
            if date1 and not date1 in date_time_fmts:
                date_time_fmts.append(date1)
            else:
                date2 = is_time(each_datetime)
                if date2 and not date2 in date_time_fmts:
                    date_time_fmts.append(date2)
                else:
                    date3 = is_date_and_time(each_datetime)
                    if date3 and not date3 in date_time_fmts:
                        date_time_fmts.append(date3)
            if not date1 and not date2 and not date3 :
                print('date time format cannot be inferred. Please check input and try again.')
    except:
        print('Error in inferring date time format. Returning...')
    return date_time_fmts
#################################################################################################
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
###############################################################################################
