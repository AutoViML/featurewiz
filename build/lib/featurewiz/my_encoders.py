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
        each_catvar = X.name
        if self.zero_low_counts[each_catvar]:
            pass
        else:
            X = X.map(self.transformers[each_catvar])
            ### simply fill in the missing values with the word "missing" ##
            X = X.fillna('missing')
        return X

    def fit_transform(self, X, y=None, **fit_params):
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
        self.zero_low_counts = zero_low_counts
        
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
        # transformers need a default name for rare categories ##
        def return_cat_value():
            return "rare_categories"
        # transformers are designed to modify X which is 2d dimensional
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
            self.transformers[each_catvar] = defaultdict(return_cat_value)
            ### Then find the unique categories in the column ###
            self.transformers[each_catvar] = dict(zip(X[each_catvar].unique(),X[each_catvar].unique()))
            low_counts = X[[each_catvar]].apply(lambda x: x.value_counts()[
                    (x.value_counts()<=(0.01*x.shape[0])).values].index).values.ravel()
            if len(low_counts) == 0:
                self.zero_low_counts[each_catvar] = True
            else:
                self.zero_low_counts[each_catvar] = False
            for each_low in low_counts:
                self.transformers[each_catvar].update({each_low:'rare_categories'})
        return self
    
    def transform(self, X, y=None, **fit_params):
        for i, each_catvar in enumerate(self.categorical_features):
            if self.zero_low_counts[each_catvar]:
                continue
            else:
                X[each_catvar] = X[each_catvar].map(self.transformers[each_catvar]).values
                ### simply fill in the missing values with the word "missing" ##
                ### Remember that fillna only works at dataframe level! ##
                X[[each_catvar]] = X[[each_catvar]].fillna('missing')
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
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
        return X, y

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
        return X, y
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
    ###   What do we groupby? a groupby column which is usually a categorical varaiable.
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
    ###   groupby_column: this is to groupby all the numeric features and compute aggregates by.
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
        else:
            ### Make sure that the numerics are numeric variables! ##
            self.numerics = list(set(self.numerics).intersection(ls))
        ### Make sure that the aggregate functions are real aggregators! ##
        self.agg_types = list(set(self.agg_types).intersection(self.func_set))
        copy_cats = copy.deepcopy(self.categoricals)
        #### if categoricals is already a list, then start transforming ###
        for i, each_catvar in enumerate(copy_cats):
            try:
                dft_cont = X[self.numerics+[each_catvar]]
            except:
                print('    %s columns given not found in data. Please correct your input.')
                return X
            ### Then find the unique categories in the column ###
            dft_full = dft_cont.groupby(each_catvar).agg(self.agg_types)
            cols =  [a +'_by_'+ str(each_catvar) +'_'+ b for (a,b) in dft_full.columns]
            dft_full.columns = cols
            
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
######################################################################################
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
        ### there are certain functions that give only error:
        ### We need to test and make sure all these functions work.
        self.func_set = {'count','sum','mean','mad','median','min','max','mode',
                        'std','var','sem', 'skew','kurt','abs', 'prod',
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
        dft_index = dft.index
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
                        dft_full = dft_full.drop(each_col, axis=1)
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
                    print('\nWarning: train and test have different number of columns. Continuing...')
            
            dft = dft.merge(dft_full, on=self.groupby_column, how='left')
            
            #### Now change the label encoded columns back to original status ##
            copy_cols = copy.deepcopy(self.groupby_column)
            for each_col in copy_cols:
                MLB = self.MLB_dict[each_col]
                dft[each_col] = MLB.inverse_transform(dft[each_col])

            ### provide the index the same as before ####
            dft.index = dft_index

        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly,
            ### if for some reason, the groupby blows up, then just return the dataframe as is - no changes!
            print('Error in groupby function: returning dataframe as is')
            return dft
        return dft
###################################################################################
