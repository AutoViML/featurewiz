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
        reverse_transformer_ = {y: x for (x, y) in transformer_.items()}
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
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
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
          No. It cannot be used in pipelines since it needs to produce two columns for the next stage in pipeline.
          See My_Label_Encoder_Pipe for an example of how to change this to use in sklearn pipelines.
    """
    def __init__(self, ts_column, verbose=0):
        self.ts_column = ts_column
        self.verbose = verbose
        self.cols_added = []
        self.fitted = False
        
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
        return self
    
    def transform(self, X, y=None):
        X = copy.deepcopy(X)
        if self.fitted:
            ### This is for test data ##########
            X_trans, _ = FE_create_time_series_features(X, self.ts_column,
                                                    ts_adds_in=self.cols_added, verbose=self.verbose)
            return X_trans
        else:
            ### This is for train data #########
            self.fit(X)
            X_trans, self.cols_added = FE_create_time_series_features(X, self.ts_column,
                                                    ts_adds_in=[], verbose=self.verbose)
            self.fitted = True
            return X_trans
        
    def fit_transform(self, X, y=None):
        X = copy.deepcopy(X)
        if self.fitted:
            X_transformed = self.transform(X, y)
        else:
            self.fit(X, y)
            X_transformed = self.transform(X, y)
        return X_transformed
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
    FE stands for FEATURE ENGINEERING - That means this function will create new features!
    #######        B E W A R E  : H U G E   N U M B E R   O F  F E A T U R E S  ###########
    This creates between 10 to 100 date time features for each date variable!! The number
    of features created depends on whether it is just a year variable or a year+month+day variable 
    and has hours and minutes or seconds also. So this can create a huge number of features
    using pandas date time column that you can send in. Optionally, you can send in a list
    of columns that you want returned. It will use those same columns to ensure train and test
    have the same number of columns and returns them in that order.
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
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.metrics import r2_score
from collections import defaultdict
import pdb
import copy
##############################################################################################
class TS_Lagging_Transformer(BaseEstimator, TransformerMixin):
    """
        #####################################################################################
        ####   This is a PERFECT lagger based on matching current date to exact lag date ####
        #####################################################################################
        ######### T I M E  S E R I E S  L A G G I N G   T R A N S F O R M E R  ##############
        ####        This is where we add Lags of Targets to Time Series data     ############
        #### This only adds lags based on days. So if you want 365 days lag, you set ########
        #### lags = 365. The time series data can be in hourly, daily or weekly periods #####
        #### Lags help a model to learn how to predict future sales based on past data ######
        #### This is a very important feature engineering technique in time series data######
        #####################################################################################
        Inputs:
        ----------------
        lags: number of lags based on days. So if you want 365 days lag, you set lags = 365.
        ts_column: name of the date-time column. It should be a pandas date time column dtype.
                It should be in your X dataframe. This will be used to set the time series index.
        hier_vars: Names of hierarchical vars (id variables) such as user_id, store_id, item_id.
                This is needed when you have 1000's of time series in your data.
                Adding hier_vars will make your time series lags more accurate.
        time_period: you can set it as "daily", "weekly", "hourly". This tells the lagger 
                that the time series is in daily, weekly or hourly format. 
        #############    This is where you use the Lagger to transform X and y ##############
        X: is a dataframe with a pandas date-time variable and the hierarchical vars (optional)
        y: You must send in a pandas series or dataframe. It must have the target column.
                It will use y and join it with X to lag. This can be multi_label target also.

        Outputs:
        -----------------
        X_transformed: This is the transformed data frame with lagged targets added.
    """
    def __init__(self, lags, ts_column, hier_vars = [], time_period="", verbose=0):
        self.lags = lags
        self.ts_column = ts_column
        if isinstance(ts_column, list):
            print('Only one date column accepted. Taking the first col from list of cols given: %s' %ts_column[0])
            self.ts_column = ts_column[0]
        else:
            self.ts_column = ts_column
        if isinstance(hier_vars, list):
            if len(hier_vars) == 0:
                print('No hierarchical vars given. Continuing without it but results may not be accurate...')
                self.hier_vars = []
            else:
                self.hier_vars = hier_vars
        elif isinstance(hier_vars, str):
            if hier_vars == '':
                print('No hierarchical vars given. Continuing without it but results may not be accurate...')
                self.hier_vars = []
            else:
                self.hier_vars = [hier_vars]
        else:
            print('hier_vars must be a string or a list. Returning')
            return
        if time_period in ["daily", "weekly", "hourly"]:
            self.time_period = time_period
        else:
            print("time period input must be either daily or weekly or hourly. Returning...")
            return
        self.verbose = verbose
        self.targets = []
        self.fitted = False
        self.train = None
        self.X_index = None
        self.ratio_col_adds = []
        self.y_prior = None
        self.columns = []
        self.ratios = None

    def get_names_of_targets(self, y):
        """
        ### Returns names of target variables in y if it is multi-label.
        """
        y = copy.deepcopy(y)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, columns=['target'], index=self.X_index)
            targets = ['target']
        if isinstance(y, pd.Series):
            if y.name is None:
                targets = ['target']
                y = pd.DataFrame(y, columns=targets, index=self.X_index)
            else:
                targets = [y.name]
                y = pd.DataFrame(y, columns=targets, index=self.X_index)
        elif isinstance(y, pd.DataFrame):
            targets = y.columns.tolist()
        self.targets = targets
        return y

        
    def fit(self, X, y):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        ts_columns = [self.ts_column]
        self.y_prior = copy.deepcopy(y)
        self.X_index = X.index
        self.columns = X.columns.tolist()
        self.fitted = True
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, pd.Series):
            print('X must be dataframe and cannot be a pandas Series. Returning...')
            return self
        elif isinstance(X, np.ndarray):
            print('X must be dataframe and cannot be a numpy array. Returning...')
            return self
        ########## you must save the product uniques so that train and test have consistent columns ##
        print('Before adding lag column, shape of data set = %s' %(X.shape,))
        if not is_datetime64_any_dtype(X[self.ts_column]):
            print('%s is not a pandas date-time dtype column. Converting it now.' %self.ts_column)
            X[self.ts_column] = pd.to_datetime(X[self.ts_column])
        try:
            self.train =  X.join(self.get_names_of_targets(y))
        except:
            print("Cannot join X to y. Check your inputs and try again...")
            return self
        if len(self.hier_vars) == 0:
            self.ratios = self.train.groupby(ts_columns)[self.targets].sum()
        else:
            self.ratios = self.train.groupby(self.hier_vars+ts_columns)[self.targets].sum()
        self.fitted = True
        return self

    def transform(self, X, y=None):
        X = copy.deepcopy(X)
        X_trans = None
        if self.fitted and y is None:
            #############################################
            ##### This is for test data #################
            #############################################
            if not is_datetime64_any_dtype(X[self.ts_column]):
                print('%s is not a pandas date-time dtype column. Converting it now.' %self.ts_column)
                X[self.ts_column] = pd.to_datetime(X[self.ts_column])
            numtrans = Numeric_Transformer(ts_columns = [self.ts_column])
            #new_columns = left_subtract(self.X_transformed.columns.tolist(), self.ratio_col_adds)
            new_columns = X.columns.tolist()
            for each_ratio_col_add in self.ratio_col_adds:
                X_train_1 = numtrans.fit_transform(X[new_columns], self.y_prior)
                y_1 = self.X_transformed[each_ratio_col_add]
                X_test_1 = numtrans.transform(X[new_columns])
                print('##### Training model to create %s column in test ############' %each_ratio_col_add)
                xgbr = XGBRegressor(random_state=0)
                # Train model using XGBRegressor
                xgbr.fit(X_train_1, y_1)
                # Make predictions
                preds1 = xgbr.predict(X_test_1)
                X[each_ratio_col_add] = preds1
            print('    Completed')
            return X[self.columns + self.ratio_col_adds]
        else:
            ##############################################
            ##### This is for train data #################
            ##############################################
            try:
                if self.time_period == 'daily':
                    self.train['new_'+self.ts_column] = self.train[self.ts_column] - pd.Timedelta(days=self.lags)
                elif self.time_period == 'weekly':
                    self.train['new_'+self.ts_column] = self.train[self.ts_column] - pd.Timedelta(weeks=self.lags)
                else:
                    self.train['new_'+self.ts_column] = self.train[self.ts_column] - pd.Timedelta(hours=self.lags)
                #### Now we must transform using the new lag column ############
                for each_target in self.targets:
                    newcol = each_target+'_lag_'+str(self.lags)
                    if len(self.hier_vars) == 0:
                        self.train[newcol] = self.train.set_index(['new_'+self.ts_column]).index.map(self.ratios[each_target].get).fillna(0)
                    else:
                        self.train[newcol] = self.train.set_index(self.hier_vars+['new_'+self.ts_column]).index.map(self.ratios[each_target].get).fillna(0)
                    self.ratio_col_adds.append(newcol)
            except:
                print('    Error occured in adding lag feature. Check your inputs. Returning...')
                return X
            ##### This is where you set the end of training and return values ###
            self.fitted = True
            X_trans = self.train[self.columns + self.ratio_col_adds ]
            print('After adding lag column, shape of data: %s' %(X_trans.shape,))
            return X_trans
    
        
    def fit_transform(self, X, y, **fit_params):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        if self.fitted:
            return self.transform(X, y)
        else:
            self.fit(X, y)
            return self.transform(X, y)

################################################################################
# THIS IS A PIPE version of the Transformers - it is useful for sklearn pipelines
class TS_Lagging_Transformer_Pipe(BaseEstimator, TransformerMixin):
    """
        #####################################################################
        ####   This is a PIPELINE version suitable for sklearn pipelines ####
        #####################################################################
    """
    def __init__(self, lags, ts_column, hier_vars = [], time_period="", verbose=0):
        self.lags = lags
        self.ts_column = ts_column
        self.hier_vars = hier_vars
        self.time_period = time_period
        self.verbose = verbose

    def fit(self, X, y):
        tslag = TS_Lagging_Transformer(lags=self.lags, ts_column=self.ts_column, 
            hier_vars = self.hier_vars, time_period=self.time_period, verbose=self.verbose)
        self.fitted = True
        return tslag

    def transform(self, X, y=None):
        if self.fitted:
            return self.transform(X, y), y
        else:
            tslag = self.fit(X, y)
            return tslag.transform(X), y

    def fit_transform(X, y, **fit_params):
        tslag = self.fit(X, y)
        return tslag.transform(X), y
#################################################################################################
class TS_Fourier_Transformer_Pipe(BaseEstimator, TransformerMixin):
    """
        #########################################################################################
        ####   This is a Fourier Transformer PIPELINE version suitable for sklearn pipelines ####
        #########################################################################################
    """
    def __init__(self, ts_column, id_column='', time_period='daily', seasonality='1year', verbose=0):
        self.ts_column = ts_column
        self.id_column = id_column
        self.time_period = time_period
        self.seasonality = seasonality
        self.verbose = verbose

    def fit(self, X, y):
        tslag = TS_Fourier_Transformer(ts_column=self.ts_column, id_column=self.id_column, 
                time_period=self.time_period, seasonality=self.seasonality, verbose=self.verbose)
        self.fitted = True
        return tslag

    def transform(self, X, y=None):
        if self.fitted:
            return self.transform(X, y), y
        else:
            tslag = self.fit(X, y)
            return tslag.transform(X), y

    def fit_transform(X, y, **fit_params):
        tslag = self.fit(X, y)
        return tslag.transform(X), y

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
    if len(features) == 0:
        features = train.columns.tolist()
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
                train_result = MLB.fit_transform(train[everycol])
                if isinstance(train_result, tuple):
                    train_result = train_result[0]

                train[everycol] = train_result
                
                if not isinstance(test, str):
                    if test is None:
                        pass
                    else:
                        test_result = MLB.transform(test[everycol])
                        if isinstance(test_result, tuple):
                            test_result = test_result[0]
                        
                        test[everycol] = test_result
                        
            except Exception as e:
                print(f'    error converting {everycol} column from string to numeric, deteail : {e}. Continuing...')
                error_columns.append(everycol)
                continue
    
    return train, test, error_columns
###############################################################################################
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import copy
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
class TS_Trend_Seasonality_Transformer(BaseEstimator, TransformerMixin):
    """
    This Transformer class will add a Trend and Seasonality column to your time series dataset.
    In many retail use cases, you will need to add a column that aggregates sales 
    for each item as a percent of that store's sales that day or week. Similarly, 
    you may need to know sales of each item as a percent of the product category by date.
    In these cases, you need a target column ("y") and a date column ("ts_column")
    and finally you need a category column ("categorical_var") to aggregate.
    We will always use "sum" as the aggregation function. The result will be a percent 
    column which you can add to your time series data set!
    
    Remember that this Transformer is very complex. Since we don't have the same dates 
    in test data as in train (since time series data are usually for forecasting problems).
    We will do 2 things since test data has unseen dates and unseen stores compared to train data. 
    So you cannot do a left join to transfer data. You need to predict those columns in test using train data. 
    1. First use X_transformed and set the y to be the ratios column. Then we use a linearn regression model
    to train on train data and predict on test data. This will form the ratios column in test.
    2. Second use X_transformed and set y to the percent column. Then we use an XGBoost regression model 
    to train on train data and predict on test data. This will form the ratios column in test.
    
    Input:
    ts_column: string. Must be an object or a pandas date-time dtype column.
        Column must be found in the X input. Otherwise it will error. 
    categorical_var: string. default is "". group_id or product_id or store ID 
        that defines a group in the time series dataset.
    verbose: default is 0. If set to 1, it will print more verbose output.
    X: pandas dataframe. Must contain the time series column (as an object dtype)
       and the categorical_var column which must be of object dtype.
    y: pandas series or dataframe. Must contain the target (as an integer or float dtype).
    """
    def __init__(self, ts_column, categorical_var="", verbose=0):
        self.ts_column = ts_column
        self.categorical_var = categorical_var
        self.verbose = verbose
        self.targets = []
        self.fitted = False
        self.train = None
        self.X_index = None
        self.ratio_col_adds = []
        self.percent_col_adds = []
        self.columns = []
        self.ratios = None

    def get_names_of_targets(self, y):
        """
        ### Returns names of target variables in y if it is multi-label.
        """
        y = copy.deepcopy(y)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, columns=['target'], index=self.X_index)
            targets = ['target']
        if isinstance(y, pd.Series):
            if y.name is None:
                targets = ['target']
                y = pd.DataFrame(y, columns=targets, index=self.X_index)
            else:
                targets = [y.name]
                y = pd.DataFrame(y, columns=targets, index=self.X_index)
        elif isinstance(y, pd.DataFrame):
            targets = y.columns.tolist()
        self.targets = targets
        return y

        
    def fit(self, X, y):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        self.y_prior = copy.deepcopy(y)
        self.X_index = X.index
        self.columns = X.columns.tolist()
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, pd.Series):
            print('X must be dataframe and cannot be a pandas Series. Returning...')
            return self
        elif isinstance(X, np.ndarray):
            print('X must be dataframe and cannot be a numpy array. Returning...')
            return self
        ########## you must save the product uniques so that train and test have consistent columns ##
        if not is_datetime64_any_dtype(X[self.ts_column]):
            print('%s is not a pandas date-time dtype column. Converting it now.' %self.ts_column)
            X[self.ts_column] = pd.to_datetime(X[self.ts_column])
        print('Before adding trend and seasonality columns, shape of data set = %s' %(X.shape,))
        try:
            self.train =  X.join(self.get_names_of_targets(y))
        except:
            print("Cannot join X to y. Check your inputs and try again...")
            return self
        self.ratios = (self.train.groupby([self.categorical_var,self.ts_column]).sum()/self.train.groupby(
                                [self.ts_column]).sum())[self.targets].to_dict()
        self.fitted = True
        return self

    def transform(self, X, y=None):
        X = copy.deepcopy(X)
        X_trans = None
        ##### Then you should transform here ############
        if self.fitted and y is None:
            if not is_datetime64_any_dtype(self.X_transformed[self.ts_column]):
                print('%s is not a pandas date-time dtype column. Converting it now.' %self.ts_column)
                self.X_transformed[self.ts_column] = pd.to_datetime(self.X_transformed[self.ts_column])
            numtrans = Numeric_Transformer(ts_columns = [self.ts_column])
            new_columns = left_subtract(self.X_transformed.columns.tolist(), self.ratio_col_adds + self.percent_col_adds)
            X_train_1 = numtrans.fit_transform(self.X_transformed[new_columns], self.y_prior)
            X_test_1 = numtrans.transform(X[new_columns])
            ### Since ts_column has been dropped, we need to subtract it from all columns ##
            for each_ratio_col_add in self.ratio_col_adds:
                y_1 = self.X_transformed[each_ratio_col_add]
                print('##### Training model to create %s column in test ############' %each_ratio_col_add)
                model1 = LinearRegression()
                model1.fit(X_train_1, y_1)
                preds1 = model1.predict(X_test_1)
                X[each_ratio_col_add] = preds1
            print('    Completed')
            for each_percent_col_add in self.percent_col_adds:
                print('##### Training model to create %s column in test ############' %each_percent_col_add)
                y_2 = self.X_transformed[each_percent_col_add]
                X_train_2 = copy.deepcopy(X_train_1)
                X_test_2 = copy.deepcopy(X_test_1)
                xgbr = XGBRegressor(random_state=0)
                # Train model using XGBRegressor
                xgbr.fit(X_train_2, y_2)
                # Make predictions
                preds2 = xgbr.predict(X_test_2)
                X[each_percent_col_add] = preds2
            print('    Completed')
            return X[self.columns + self.ratio_col_adds + self.percent_col_adds]
        try:
            for each_target in self.targets:
                self.train[each_target+'_'+self.categorical_var+'_trend'] = self.train.set_index(
                    [self.categorical_var, self.ts_column]).index.map(self.ratios[each_target].get)
                self.train[each_target+'_'+self.categorical_var+'_seasonality'] = self.train[
                        each_target]/self.train[each_target+'_'+self.categorical_var+'_trend']
                self.ratio_col_adds.append(each_target+'_'+self.categorical_var+'_trend')
                self.percent_col_adds.append(each_target+'_'+self.categorical_var+'_seasonality')
        except:
            print('    Error occured in adding trend and seasonality features. Check your inputs. Returning...')
            return X
        ##### This is where you set the end of training and return values ###
        self.fitted = True
        X_trans = self.train[self.columns + self.ratio_col_adds + self.percent_col_adds]
        print('After adding trend and seasonality columns, shape of data: %s' %(X_trans.shape,))
        return X_trans
    
        
    def fit_transform(self, X, y):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        if self.fitted:
            self.X_transformed = self.transform(X, y)
        else:
            self.fit(X, y)
            self.X_transformed = self.transform(X, y)
        return self.X_transformed        
############################################################################################################
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import copy
class TS_Fourier_Transformer(BaseEstimator, TransformerMixin):
    """
    This Transformer class will add fourier transform features to daily and weekly time series data.
        WARNING: You cannot use it for hourly or any other kind of time series data yet.
    The time series must have a time column and an (optional) group identifier column.
    It will return a dataset with Fourier transforms added for every day in a year and by group.
    WARNING: If your test data does not contain any items (group_ids) from train data, 
        then all columns will be zero in test data since it cannot learn from train data.
        In that case, you are better off selecting another group that is common to both 
        train and test data. The key to success is finding common groups within both!
        
    Input:
    ts_column: string. Must be a date-time column. 
        Column must be in pandas date-time format. Otherwise will error. 
    id_column: string. default is "". group_id or product_id or store ID 
        that defines a group in the time series dataset.
    time_period: string. default="daily". It will produce features based on dayofyear.
        Use "weekly" to produce features based on weekofyear. 
    seasonality: string. default="1year". It will produce features for up to 1 year.
        Use "2years" value will produce features for 2 years (max). 
    verbose: default is 0. If set to 1, it will print more verbose output.
    """
    def __init__(self, ts_column, id_column="", time_period="daily", seasonality="1year", verbose=0):
        self.ts_column = ts_column
        self.id_column = id_column
        self.time_period = time_period
        self.seasonality = seasonality
        if isinstance(self.seasonality, str):
            if self.seasonality == "":
                self.seasonality = "1year"
        if isinstance(self.time_period, str):
            if self.time_period == "":
                self.time_period = "daily"
        self.verbose = verbose
        self.fitted = False
        self.train = False
        self.products = []
        self.listofyears = []
        self.dayofbiyear = None
        
    def fit(self, X, y=None):
        X = copy.deepcopy(X)
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, pd.Series):
            if self.verbose:
                print('X must be dataframe. Converting it to a pd.DataFrame.')
            X = pd.DataFrame(X.values, columns=[X.name])
        elif isinstance(X, np.ndarray):
            if self.verbose:
                print('X must be dataframe and cannot be numpy array. Returning...')
            return self
        else:
            #### There is no way to transform dataframes in an sklearn pipeline  
            ####    since you will get a nested renamer error if you try ###
            #### But if it is a one-dimensional dataframe, you can convert into Series
            if self.verbose:
                print('X is a DataFrame...')
            pass
        ########## you must save the product uniques so that train and test have consistent columns ##
        self.products = X[self.id_column].unique().tolist()
        print('Before Fourier features engg, shape of data = %s' %(X.shape,))
        self.fitted = True
        return self

    def transform(self, X, y=None):
        X = copy.deepcopy(X)
        if self.fitted and self.train:
            self.train = False
            return self.X_transformed
        ##### Then you should transform here ############
        # Time period could be 1year or 2year: otherwise it will assume 1 year.
        if self.time_period == 'daily':
            if self.seasonality == '1year':
                self.dayofbiyear = X[self.ts_column].dt.dayofyear  # 1 to 365    
                self.listofyears = [2, 4]
            elif self.seasonality == '2year':
                self.dayofbiyear = X[self.ts_column].dt.dayofyear + 365*(1-(X[self.ts_column].dt.year%2))  # 1 to 730
                self.listofyears = [1, 2, 4]
        elif self.time_period == 'weekly':
            if self.seasonality == '1year':
                self.dayofbiyear = X[self.ts_column].dt.weekofyear  # 1 to 365    
                self.listofyears = [2, 4]
            elif self.seasonality == '2year':
                self.dayofbiyear = X[self.ts_column].dt.weekofyear + 52*(1-(X[self.ts_column].dt.year%2))  # 1 to 104
                self.listofyears = [1, 2, 4]
        ##### You need to reset the number of days above for each dataset ###
        try:
            print('    will create %s unique features...' %(len(self.products)*len(self.listofyears)*2))
            # k=1 -> 2 years, k=2 -> 1 year, k=4 -> 6 months
            for k in self.listofyears:
                if self.time_period == 'daily':
                    if self.seasonality == '1year':
                        X[f'sin{k}'] = np.sin(2 * np.pi * k * self.dayofbiyear / (1* 365))
                        X[f'cos{k}'] = np.cos(2 * np.pi * k * self.dayofbiyear / (1* 365))
                    else:
                        X[f'sin{k}'] = np.sin(2 * np.pi * k * self.dayofbiyear / (2* 365))
                        X[f'cos{k}'] = np.cos(2 * np.pi * k * self.dayofbiyear / (2* 365))
                elif self.time_period == 'weekly':
                    if self.seasonality == '1year':
                        X[f'sin{k}'] = np.sin(2 * np.pi * k * self.dayofbiyear / (1* 52))
                        X[f'cos{k}'] = np.cos(2 * np.pi * k * self.dayofbiyear / (1* 52))
                    else:
                        X[f'sin{k}'] = np.sin(2 * np.pi * k * self.dayofbiyear / (2* 52))
                        X[f'cos{k}'] = np.cos(2 * np.pi * k * self.dayofbiyear / (2* 52))

                if self.id_column:  ### only do this if they send in an ID column ####
                    #### we do this for Different items since each 
                    ####     has a different seasonality pattern
                    for product in self.products:
                        X[f'sin_{k}_{product}'] = X[f'sin{k}'] * (X[self.id_column] == product)
                        X[f'cos_{k}_{product}'] = X[f'cos{k}'] * (X[self.id_column] == product)

                X = X.drop([f'sin{k}', f'cos{k}'], axis=1)
            print('After Fourier features engg, shape of data: %s' %(X.shape,))
        except:
            print('    Error occured in adding Fourier features. Check your inputs. Returning...')
            self.train = False
            self.X_transformed = X
            return self.X_transformed
        ##### This is where you set the end of training and return values ###
        self.fitted = True
        self.train = True
        self.X_transformed = X
        return self.X_transformed
    
        
    def fit_transform(self, X, y=None):
        X = copy.deepcopy(X)
        self.fit(X)
        self.transform(X)
        self.train = False
        return self.X_transformed
#########################################################################################################
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import copy
import pdb
class Column_Names_Transformer(BaseEstimator, TransformerMixin):
    """
    This Transformer class will make your column names unique. 
    Just fit on train data and transform on test data to make them same.
        
    Input:
    train or X_train: a dataframe. Must have column names - should not be an array. 
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.old_column_names = []
        self.new_column_names = []
        self.rename_dict = {}
        self.train = False
        self.fitted = False
        self.transformed_flag = False
        
    def fit(self, X):
        X = copy.deepcopy(X)
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, pd.Series):
            if self.verbose:
                print('X must be dataframe. Converting it to a pd.DataFrame.')
            X = pd.DataFrame(X.values, columns=[X.name])
        elif isinstance(X, np.ndarray):
            if self.verbose:
                print('X must be dataframe and cannot be numpy array. Returning...')
            return self
        else:
            #### There is no way to transform dataframes in an sklearn pipeline  
            ####    since you will get a nested renamer error if you try ###
            #### But if it is a one-dimensional dataframe, you can convert into Series
            if self.verbose:
                print('X is a DataFrame...')
            pass
        ########## you must save the product uniques so that train and test have consistent columns ##
        self.old_column_names = X.columns.tolist()
        if self.verbose:
            print('Before making column names unique, shape of data = %s' %(X.shape,))
        self.new_column_names, self.transformed_flag = EDA_make_column_names_unique(X)
        self.rename_dict = dict(zip(self.old_column_names, self.new_column_names))
        self.fitted = True
        return self

    def transform(self, X, y=None):
        
        if self.fitted and self.train:
            self.train = False
            return self.X_transformed
        ##### Then you should transform here ############
        if self.verbose:
            print('    will make features unique...')
        try:
            self.X_transformed = copy.deepcopy(X)
            self.X_transformed.rename(columns=self.rename_dict, inplace=True)            
        except:
            print('    Error occured in making unique features. Check your inputs. Returning...')
            self.train = False
            self.X_transformed = X
            return self.X_transformed
        ##### This is where you set the end of training and return values ###
        self.fitted = True
        self.train = True
        return self.X_transformed
    
        
    def fit_transform(self, X, y=None):
        X = copy.deepcopy(X)
        self.fit(X)
        self.transform(X)
        self.train = False
        return self.X_transformed
################################################################################
import random
import collections
import re
import copy
def EDA_make_column_names_unique(data_input):
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
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import copy
class Numeric_Transformer(BaseEstimator, TransformerMixin):
    """
    This Transformer class will convert all date-time, object and categorical columns to numeric.
        It will leave numeric columns as is.
        
    Input:
    ts_columns: list of names of date-time columns. These columns must be a pandas date-time dtype columns.
        Columns must be found in the X input. Otherwise it will error. You can leave it as empty list.
    verbose: default is 0. If set to 1, it will print more verbose output.
    X: pandas dataframe. Must contain the time series columns (as date-time dtypes). Otherwise error!
       All your object columns must be of object or categorical dtypes.
    y: pandas series or dataframe. Must contain the target (as an integer or float dtype).
    
    Outputs:
    X_transformed: It will return a transformed dataframe with all numeric columns.
    
    """
    def __init__(self, ts_columns=[], verbose=0):
        self.ts_columns = ts_columns
        self.verbose = verbose
        self.lis = []
        self.columns = []
        self.error_columns = []
        self.mlbs = {}
        self.dss = {}
        self.fitted = False
      
    def fit(self, X, y=None):
        X = copy.deepcopy(X)
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
        ### Now you can check if the parts of tuple are dataframe series, etc.
        if isinstance(X, pd.Series):
            print('X is a pandas Series. Converting...')
            X = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            print('X must be dataframe and cannot be a numpy array. Returning...')
            return self
        ########## you must save the product uniques so that train and test have consistent columns ##
        print('Before converting columns, shape of data set = %s' %(X.shape,))
        self.lis = X.select_dtypes('object').columns.tolist() + X.select_dtypes('category').columns.tolist()
        self.columns = X.columns.tolist()
        return self

    def transform(self, X, y=None):
        X = copy.deepcopy(X)
        ### This is for both train and test data ########
        ### First we must convert all date-time columns to their parts of time ####
        if len(self.ts_columns) > 0:
            for every_ts in self.ts_columns:
                ## we must convert date_col into date-time features before we convert object vars
                if self.fitted:
                    ds1 = self.dss[every_ts]
                    X = ds1.transform(X) ### this will give your transformed values as a dataframe
                else:
                    ds = DateTime_Transformer(ts_column=every_ts, verbose=1)
                    X = ds.fit_transform(X) ## this will give your transformed values as a dataframe
                    self.dss[every_ts] = ds
            print('After converting all date columns to numeric, shape of data: %s' %(X.shape,))
        #### Now you have to convert all the columns into numeric ###
        if not self.fitted:
            self.columns = X.columns.tolist()
            self.lis = X.select_dtypes('object').columns.tolist() + X.select_dtypes('category').columns.tolist()
        if not (len(self.lis)==0):
            for everycol in self.lis:
                try:
                    if self.fitted:
                        MLB = self.mlbs[everycol]
                        train_result = MLB.transform(X[everycol])
                    else:
                        MLB = My_LabelEncoder()
                        train_result = MLB.fit_transform(X[everycol])
                        self.mlbs[everycol] =  MLB
                except Exception as e:
                    print(f'    error converting {everycol} column from string to numeric, deteail : {e}. Continuing...')
                    self.error_columns.append(everycol)
                    continue
                if isinstance(train_result, tuple):
                    train_result = train_result[0]
                #### This is where you store the transformed column #####
                X[everycol] = train_result
        else:
            print('No categorical columns in X. Returning...')
            return 
        ##### This is where you set the end of training and return values ###
        self.fitted = True
        self.columns = left_subtract(self.columns, self.error_columns)
        print('After converting all date, object and category columns to numeric, shape of data: %s' %(X[self.columns].shape,))
        return X[self.columns]
    
        
    def fit_transform(self, X, y):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        if self.fitted:
            X_transformed = self.transform(X, y)
        else:
            self.fit(X, y)
            X_transformed = self.transform(X, y)
        return X_transformed

#####################################################################################
#####################################################################################
