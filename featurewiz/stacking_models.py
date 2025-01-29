import numpy as np
np.random.seed(42)
import random
random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.base import ClassifierMixin
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error,auc
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LassoLarsCV
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
#from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.multioutput import ClassifierChain, RegressorChain
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.model_selection import train_test_split
import pathlib
from scipy import stats
from scipy.stats import norm, skew
import time
import copy
import pdb
from collections import Counter
from collections import defaultdict
from collections import OrderedDict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

#########################################################################################
# Calculate class weight
import copy
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

def get_class_distribution(y_input, verbose=0):
    y_input = copy.deepcopy(y_input)
    if isinstance(y_input, np.ndarray):
        class_weights = compute_class_weight('balanced', classes=np.unique(y_input), y=y_input.reshape(-1))
    elif isinstance(y_input, pd.Series):
        class_weights = compute_class_weight('balanced', classes=np.unique(y_input.values), y=y_input.values.reshape(-1))
    elif isinstance(y_input, pd.DataFrame):
        ### if it is a dataframe, return only if it s one column dataframe ##
        y_input = y_input.iloc[:,0]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_input.values), y=y_input.values.reshape(-1))
    else:
        ### if you cannot detect the type or if it is a multi-column dataframe, ignore it
        return None
    if len(class_weights[(class_weights> 10)]) > 0:
        class_weights = (class_weights/10)
    else:
        class_weights = (class_weights)
    #print('    class_weights = %s' %class_weights)
    classes = np.unique(y_input)
    xp = Counter(y_input)
    class_weights[(class_weights<1)]=1
    class_rows = class_weights*[xp[x] for x in classes]
    class_rows = class_rows.astype(int)
    min_rows = np.min(class_rows)
    class_weighted_rows = dict(zip(classes,class_rows))
    ### sometimes the middle classes are not found in the dictionary ###
    for x in range(np.unique(y_input).max()+1):
        if x not in list(class_weighted_rows.keys()):
            class_weighted_rows.update({x:min_rows})
        else:
            pass
    ### return the updated dictionary with reduced weights > 1000 ###
    keys = np.array(list(class_weighted_rows.keys()))
    val = np.array(list(class_weighted_rows.values()))
    class_weighted_rows = dict(zip(keys, np.where(val>=1000,val//10,val)))
    if verbose:
        print('    class_weighted_rows = %s' %class_weighted_rows)
    return class_weighted_rows
############################################################################
def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))
##################################################
### Define the input models here #######
###################################################
class Stacking_Classifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    #################################################################################
    ############          Credit for Stacking Classifier        #####################
    #################################################################################
    #### Greatly indebted to Gilbert Tanner who explained Stacked Models here
    ####   https://gilberttanner.com/blog/introduction-to-ensemble-learning
    #### I used the blog to create a Stacking Classifier that can handle multi-label targets
    #################################################################################
    """
    def __init__(self):
        n_folds = 5
        use_features = False
        self.base_models = []
        self.meta_model = None
        self.n_folds = n_folds
        self.use_features = use_features
        self.target_len = 1

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        import lightgbm as lgb
        models_dict = stacking_models_list(X_train=X, y_train=y, modeltype='Classification', verbose=1)
        self.base_models = list(models_dict.values())
        self.base_models_ = [list() for x in self.base_models]
        if y.ndim >= 2:
            if y.shape[1] == 1:
                self.meta_model = lgb.LGBMClassifier(n_estimators=100, random_state=99, n_jobs=-1)
            else:
                stump = lgb.LGBMClassifier(n_estimators=50, random_state=99)
                self.meta_model = MultiOutputClassifier(stump, n_jobs=-1)
        else:
            self.meta_model = lgb.LGBMClassifier(n_estimators=100, random_state=99, n_jobs=-1)
        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        start_time = time.time()
        model_name = str(self.meta_model).split("(")[0]
        print('Stacking model %s training started. This will take time...' %model_name)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        # Train cloned base models and create out-of-fold predictions
        if y.ndim <= 1:
            self.target_len = 1
        else:
            self.target_len = y.shape[1]
        out_of_fold_predictions = np.zeros((X.shape[0], self.target_len*len(self.base_models)))
        for i, model in enumerate(self.base_models):
            start_time = time.time()
            print('  %s model training and prediction...' %str(model).split("(")[0])
            
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                if y.ndim == 1:
                    out_of_fold_predictions[holdout_index, i] = y_pred.ravel()
                elif y.ndim <= 2:
                    if y.shape[1] == 1:
                        out_of_fold_predictions[holdout_index, i] = y_pred.ravel()
                    else:
                        next_i = int(i+self.target_len)
                        out_of_fold_predictions[holdout_index,i:next_i] = y_pred
                else:
                    next_i = int(i+self.target_len)
                    out_of_fold_predictions[holdout_index,i:next_i] = y_pred
            print('    Time taken = %0.0f seconds' %(time.time()-start_time))
        
        if self.use_features:
            self.meta_model_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_model_.fit(out_of_fold_predictions, y)
            
        return self

    #Do the predictions of all base models on the test data and use the averaged predictions as
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        if self.target_len == 1:
            meta_features = np.column_stack([
                np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                for base_models in self.base_models_ ])
        else:
            max_len = self.target_len
            base_models = self.base_models_[0]
            for each_m, model in enumerate(base_models):
                if each_m == 0:
                    stump_pred = model.predict(X)
                    pred = stump_pred[:]
                else:
                    addl_pred = model.predict(X)
                    stump_pred = np.column_stack([stump_pred, addl_pred])
                    for each_i in range(max_len):
                        next_i = int(each_i+self.target_len)                        
                        #pred[:,each_i] = np.column_stack([stump_pred[:,each_i],stump_pred[:,next_i]]).mean(axis=1)
                        pred[:,each_i] = (np.column_stack([stump_pred[:,each_i],stump_pred[:,next_i]]).mean(axis=1)>=0.5).astype(int)
            meta_features = pred[:]

        if self.use_features:
            return self.meta_model_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_model_.predict(meta_features)
###################################################################
class Stacking_Regressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    #################################################################################
    ############          Credit for Stacking Regressor        ######################
    #################################################################################
    #### Greatly indebted to Gilbert Tanner who explained Stacked Models here
    ####   https://gilberttanner.com/blog/introduction-to-ensemble-learning
    #### I used the blog to create a Stacking Regressor that can handle multi-label targets
    #################################################################################
    """
    def __init__(self, use_features=True):
        n_folds = 5
        self.base_models = []
        self.meta_model = None
        self.n_folds = n_folds
        self.use_features = use_features
        self.target_len = 1
        
    def fit(self, X, y):
        """Fit all the models on the given dataset"""

        import lightgbm as lgb
        models_dict = stacking_models_list(X_train=X, y_train=y, modeltype='Regression', verbose=1)
        self.base_models = list(models_dict.values())
        self.base_models_ = [list() for x in self.base_models]
        if y.ndim >= 2:
            if y.shape[1] == 1:
                self.meta_model = lgb.LGBMRegressor(n_estimators=50, random_state=99)
            else:
                stump = lgb.LGBMRegressor(n_estimators=50, random_state=99)
                self.meta_model = MultiOutputRegressor(stump, n_jobs=-1)
        else:
            self.meta_model = lgb.LGBMRegressor(n_estimators=50, random_state=99)
        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        start_time = time.time()
        model_name = str(self.meta_model).split("(")[0]
        print('Stacking model %s training started. This will take time...' %model_name)
        
        # Train cloned base models and create out-of-fold predictions
        if y.ndim <= 1:
            self.target_len = 1
        else:
            self.target_len = y.shape[1]
        out_of_fold_predictions = np.zeros((X.shape[0], self.target_len*len(self.base_models)))
        for i, model in enumerate(self.base_models):
            print('  %s model training and prediction...' %str(model).split("(")[0])
            
            start_time = time.time()
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                
                if y.ndim == 1:
                    out_of_fold_predictions[holdout_index, i] = y_pred.ravel()
                elif y.ndim <= 2:
                    if y.shape[1] == 1:
                        out_of_fold_predictions[holdout_index, i] = y_pred.ravel()
                    else:
                        next_i = int(i+self.target_len)
                        out_of_fold_predictions[holdout_index,i:next_i] = y_pred
                else:
                    next_i = int(i+self.target_len)
                    out_of_fold_predictions[holdout_index,i:next_i] = y_pred
            print('    Time taken = %0.0f seconds' %(time.time()-start_time))
        
        if self.use_features:
            self.meta_model_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_model_.fit(out_of_fold_predictions, y)
            
        return self
    
    def predict(self, X):
        if self.target_len == 1:
            meta_features = np.column_stack([
                np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                for base_models in self.base_models_ ])
        else:
            max_len = self.target_len
            base_models = self.base_models_[0]
            for each_m, model in enumerate(base_models):
                if each_m == 0:
                    stump_pred = model.predict(X)
                    pred = stump_pred[:]
                else:
                    addl_pred = model.predict(X)
                    stump_pred = np.column_stack([stump_pred, addl_pred])
                    for each_i in range(max_len):
                        next_i = int(each_i+self.target_len)                        
                        pred[:,each_i] = np.column_stack([stump_pred[:,each_i],stump_pred[:,next_i]]).mean(axis=1)
            meta_features = pred[:]

        if self.use_features:
            return self.meta_model_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_model_.predict(meta_features)
###############################################################################
class Blending_Regressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    #################################################################################
    ############          Credit for Blending Regressor        ######################
    #################################################################################
    #### Greatly indebted to Gilbert Tanner who explained Stacked Models here
    ####   https://gilberttanner.com/blog/introduction-to-ensemble-learning
    #### I used the blog to create a Blending Regressor that can handle multi-label targets
    #################################################################################
    """
    def __init__(self, holdout_pct=0.2, use_features=True):
        # create models
        n_folds = 5
        self.base_models = []
        self.meta_model = None
        self.n_folds = n_folds
        self.holdout_pct = holdout_pct
        self.use_features = use_features
        self.target_len = 1

    def fit(self, X, y):
        import lightgbm as lgb
        models_dict = stacking_models_list(X_train=X, y_train=y, modeltype='Regression', verbose=1)
        self.base_models = list(models_dict.values())
        self.base_models_ = [clone(x) for x in self.base_models]
        
        if y.ndim >= 2:
            if y.shape[1] == 1:
                self.meta_model = lgb.LGBMRegressor(n_estimators=50, random_state=99, n_jobs=-1)
            else:
                stump = lgb.LGBMRegressor(n_estimators=50, random_state=99)
                self.meta_model = MultiOutputRegressor(stump, n_jobs=-1)
        else:
            self.meta_model = lgb.LGBMRegressor(n_estimators=50, random_state=99, n_jobs=-1)
        self.meta_model_ = clone(self.meta_model)

        start_time = time.time()
        model_name = str(self.meta_model).split("(")[0]
        print('Blending model %s training started. This will take time...' %model_name)

        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=self.holdout_pct)
        if y.ndim <= 1:
            self.target_len = 1
        else:
            self.target_len = y.shape[1]

        #holdout_predictions = np.zeros((X_holdout.shape[0], self.target_len*len(self.base_models)))
        
        for i, model in enumerate(self.base_models_):
            print('  %s model training and prediction...' %str(model).split("(")[0])
            start_time = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_holdout)
            print('    Time taken = %0.0f seconds' %(time.time()-start_time))
            if i == 0:
                holdout_predictions = y_pred
            else:
                holdout_predictions = np.column_stack([holdout_predictions, y_pred])
        
        if self.use_features:
            if holdout_predictions.ndim < 2:
                self.meta_model_.fit(np.hstack((X_holdout, holdout_predictions.reshape(-1,1))), y_holdout)
            else:
                self.meta_model_.fit(np.hstack((X_holdout, holdout_predictions)), y_holdout)
        else:
            self.meta_model_.fit(holdout_predictions, y_holdout)

        return self

    def predict(self, X):
        #### This can handle multi_label predictions now ###
        
        if self.target_len == 1:
            meta_features = np.column_stack([
                model.predict(X) for model in self.base_models_])
        else:
            max_len = self.target_len
            for each_m, model in enumerate(self.base_models_):
                if each_m == 0:
                    stump_pred = model.predict(X)
                    pred = stump_pred[:]
                else:
                    addl_pred = model.predict(X)
                    stump_pred = np.column_stack([stump_pred, addl_pred])
                    for each_i in range(max_len):
                        next_i = int(each_i+self.target_len)                        
                        pred[:,each_i] = np.column_stack([stump_pred[:,each_i],stump_pred[:,next_i]]).mean(axis=1)
            meta_features = pred[:]
        
        if self.use_features:
            if meta_features.ndim < 2:
                return self.meta_model_.predict(np.hstack((X, meta_features.reshape(-1,1))))
            else:
                return self.meta_model_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_model_.predict(meta_features)
######################################################################################
def find_rare_class(classes, verbose=0):
    ######### Print the % count of each class in a Target variable  #####
    """
    Works on Multi Class too. Prints class percentages count of target variable.
    It returns the name of the Rare class (the one with the minimum class member count).
    This can also be helpful in using it as pos_label in Binary and Multi Class problems.
    """
    counts = OrderedDict(Counter(classes))
    total = sum(counts.values())
    if verbose >= 1:
        print(' Class  -> Counts -> Percent')
        for cls in counts.keys():
            print("%6s: % 7d  ->  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))
    if type(pd.Series(counts).idxmin())==str:
        return pd.Series(counts).idxmin()
    else:
        return int(pd.Series(counts).idxmin())
################################################################################
def stacking_models_list(X_train, y_train, modeltype='Regression', verbose=0):
    """
    Quickly build Stacks of multiple model results
    Input must be a clean data set (only numeric variables, no categorical or string variables).
    """
    import lightgbm as lgb
    
    X_train = copy.deepcopy(X_train)
    y_train = copy.deepcopy(y_train)
    start_time = time.time()
    seed = 99
    if len(X_train) <= 100000 or X_train.shape[1] < 50:
        NUMS = 100
        FOLDS = 5
    else:
        NUMS = 200
        FOLDS = 10
    ## create Stacking models
    estimators = []
    #### This is where you don't fit the model but just do cross_val_predict ####
    if modeltype == 'Regression':
        if y_train.ndim >= 2:
            if y_train.shape[1] > 1:
                stump = lgb.LGBMRegressor(n_estimators=50, random_state=99)
                model1 = MultiOutputRegressor(stump, n_jobs=-1)
                estimators.append(('Multi Output Regressor',model1))
                estimators_list = [(tuples[0],tuples[1]) for tuples in estimators]
                estimator_names = [tuples[0] for tuples in estimators]
                print('List of models chosen for stacking: %s' %estimators_list)
                return dict(estimators_list)
        ######    Bagging models if Bagging is chosen ####
        model3 = LinearRegression(n_jobs=-1)
        estimators.append(('Linear Model',model3))
        ####   Tree models if Linear chosen #####
        model5 = DecisionTreeRegressor(random_state=seed,min_samples_leaf=2)
        estimators.append(('Decision Trees',model5))
        ####   Linear Models if Boosting is chosen #####
        model6 = ExtraTreeRegressor(random_state=seed)
        estimators.append(('Extra Tree Regressor',model6))

        #model7 = RandomForestRegressor(n_estimators=50,random_state=seed, n_jobs=-1)
        model7 = Ridge(alpha=0.5)
        estimators.append(('Ridge',model7))
    else:
        ### This is for classification problems ########
        if y_train.ndim >= 2:
            if y_train.shape[1] > 1:
                stump = lgb.LGBMClassifier(n_estimators=50, random_state=99)
                model1 = MultiOutputClassifier(stump, n_jobs=-1)
                estimators.append(('Multi Output Classifier',model1))
                estimators_list = [(tuples[0],tuples[1]) for tuples in estimators]
                estimator_names = [tuples[0] for tuples in estimators]
                print('List of models chosen for stacking: %s' %estimators_list)
                return dict(estimators_list)
        ### Leave this as it is - don't change it #######
        n_classes = len(Counter(y_train))
        if n_classes > 2:
            model3 = LogisticRegression(max_iter=5000, multi_class='ovr')
        else:
            model3 = LogisticRegression(max_iter=5000)
        estimators.append(('Logistic Regression', model3))
        ####   Linear Models if Boosting is chosen #####
        model4 = LinearDiscriminantAnalysis()
        estimators.append(('Linear Discriminant',model4))

        model5 = LGBMClassifier()
        estimators.append(('LightGBM',model5))

        ######    Naive Bayes models if Bagging is chosen ####
        model7 = DecisionTreeClassifier(min_samples_leaf=2)
        estimators.append(('Decision Tree',model7))
    
    #### Create a new list here ####################

    estimators_list = [(tuples[0],tuples[1]) for tuples in estimators]
    estimator_names = [tuples[0] for tuples in estimators]
    print('List of models chosen for stacking: %s' %estimators_list)
    return dict(estimators_list)
#########################################################
import copy
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import class_weight
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import pdb

def analyze_problem_type_array(y_train, verbose=0) :  
    y_train = copy.deepcopy(y_train)
    cat_limit = 30 ### this determines the number of categories to name integers as classification ##
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    try:
        ndim = y_train.shape[1]
    except:
        ndim = 1
    #### Use this for finding multi-label in numpy arrays ###
    if ndim <= 1:
        multi_label = False
    else:
        multi_label = True
    ####  This is where you detect what kind of problem it is #################
    if not multi_label:
        if  y_train.dtype in ['int64', 'int32','int16']:
            if len(np.unique(y_train)) <= 2:
                model_class = 'Binary_Classification'
            elif len(np.unique(y_train)) > 2 and len(np.unique(y_train)) <= cat_limit:
                model_class = 'Multi_Classification'
            else:
                model_class = 'Regression'
        elif  y_train.dtype in ['float16','float32','float64']:
            if len(np.unique(y_train)) <= 2:
                model_class = 'Binary_Classification'
            elif len(np.unique(y_train)) > 2 and len(np.unique(y_train)) <= float_limit:
                model_class = 'Multi_Classification'
            else:
                model_class = 'Regression'
        else:
            if len(np.unique(y_train)) <= 2:
                model_class = 'Binary_Classification'
            else:
                model_class = 'Multi_Classification'
    else:
        for i in range(y_train.ndim):
            ### if target is a list, then we should test dtypes a different way ###
            if y_train.dtype in ['int64', 'int32','int16']:
                if len(np.unique(y_train[:,0])) <= 2:
                    model_class = 'Binary_Classification'
                elif len(np.unique(y_train[:,0])) > 2 and len(np.unique(y_train[:,0])) <= cat_limit:
                    model_class = 'Multi_Classification'
                else:
                    model_class = 'Regression'
            elif  y_train.dtype in ['float16','float32','float64']:
                if len(np.unique(y_train[:,0])) <= 2:
                    model_class = 'Binary_Classification'
                elif len(np.unique(y_train[:,0])) > 2 and len(np.unique(y_train[:,0])) <= float_limit:
                    model_class = 'Multi_Classification'
                else:
                    model_class = 'Regression'
            else:
                if len(np.unique(y_train[:,0])) <= 2:
                    model_class = 'Binary_Classification'
                else:
                    model_class = 'Multi_Classification'
    ########### print this for the start of next step ###########
    if multi_label:
        print('''    %s %s problem ''' %('Multi_Label', model_class))
    else:
        print('''    %s %s problem ''' %('Single_Label', model_class))
    return model_class, multi_label
###############################################################################
def train_evaluate_adasyn(X_train, y_train, X_test, y_test, final_estimator,
                    n_neighbors, sampling_strategy, class_weights_dict):
    # ADASYN resampling
    #adasyn = ADASYN(n_neighbors=n_neighbors, sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = X_train, y_train

    # Simplified base estimators for the stacking classifier
    base_estimators = [
        ('rf', RandomForestClassifier(class_weight=None, 
                                      n_estimators=100,
                                      random_state=42)),
        ('log_reg', LogisticRegression(random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42))
            ]

    # Final estimator with class weights
    if final_estimator is None:
        #final_estimator = XGBClassifier(learning_rate=0.2, n_estimators=100, random_state=99)
        final_estimator = RandomForestClassifier(class_weight=class_weights_dict, 
                                                 n_estimators=100,
                                                 random_state=42)

    # Creating the Stacking Classifier with simplified base estimators
    stacking_classifier = StackingClassifier(estimators=base_estimators, 
                                             final_estimator=final_estimator, cv=5)

    # Fitting the classifier on the resampled training data
    stacking_classifier.fit(X_resampled, y_resampled)

    # Predicting on the test set
    y_pred_resampled = stacking_classifier.predict(X_test)

    # Evaluating the classifier
    accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
    f1_score_resampled = f1_score(y_test, y_pred_resampled, average='macro')
    #print(f"ADASYN Parameters - n_neighbors: {n_neighbors}, sampling_strategy: {sampling_strategy}")
    print('F1 score macro = ', f1_score_resampled)
    print(classification_report(y_test, y_pred_resampled))
    return stacking_classifier, f1_score_resampled

#######################################################################################
class StackingClassifier_Multi(BaseEstimator, ClassifierMixin):
    def __init__(self, final_estimator=None):
        print('initialized')
        # Compute class weights
        self.final_model = None
        self.final_estimator = final_estimator

    def fit(self, X, y):
        class_weights_dict = get_class_distribution(y)
        modeltype, multi_label = analyze_problem_type_array(y)
        
        # Function to train and evaluate the model with ADASYN resampling
        if modeltype == 'Regression':
            X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                              test_size=0.10, random_state=1,)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                              test_size=0.10, 
                                                            stratify=y,
                                                            random_state=42)
        print('    Train data: ', X_train.shape, ', Validation data: ', X_val.shape)

        # Parameters to try for ADASYN
        n_neighbors = 5
        ### do not use dictionary for multi-class since it doesn't work
        ### do not try it because I have tried multiple things and they don't work
        sampling_strategy = 'auto'
        # Iterating over different combinations of ADASYN parameters
        best_neighbors = 3
        best_sampling_strategy = 'auto'
        f1_score_final = 0
        model_final = None
        print('Model results on Validation data:')
        model_temp, f1_score_temp = train_evaluate_adasyn(X_train, y_train, X_val, y_val,
                                final_estimator=self.final_estimator,
                                n_neighbors=n_neighbors, sampling_strategy=sampling_strategy, 
                                class_weights_dict=class_weights_dict)
        f1_score_final = copy.deepcopy(f1_score_temp)
        model_final = copy.deepcopy(model_temp)
            
        ### training the final model on full X and y before sending it out
        X_resampled, y_resampled = X, y
        model_final.fit(X_resampled, y_resampled)
        self.final_model = model_final
        return self
    
    def predict(self, X):
        return self.final_model.predict(X)
    
    def predict_proba(self, X):
        return self.final_model.predict_proba(X)
##########################################################################
