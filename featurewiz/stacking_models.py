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
from imblearn.over_sampling import ADASYN
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
    adasyn = ADASYN(n_neighbors=n_neighbors, sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

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
    print(f"ADASYN Parameters - n_neighbors: {n_neighbors}, sampling_strategy: {sampling_strategy}")
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
        n_neighbors_options = [5, 10]
        ### do not use dictionary for multi-class since it doesn't work
        ### do not try it because I have tried multiple things and they don't work
        sampling_strategy = 'auto'
        # Iterating over different combinations of ADASYN parameters
        best_neighbors = 3
        best_sampling_strategy = 'auto'
        f1_score_final = 0
        model_final = None
        print('Model results on Validation data:')
        try:
            for n_neighbors in n_neighbors_options:
                if np.unique(y_val).min() > n_neighbors:
                    n_neighbors = np.int(np.unique(y_val).min()-1)
                model_temp, f1_score_temp = train_evaluate_adasyn(X_train, y_train, X_val, y_val,
                                        final_estimator=self.final_estimator,
                                        n_neighbors=n_neighbors, sampling_strategy=sampling_strategy, 
                                        class_weights_dict=class_weights_dict)
                if f1_score_temp > f1_score_final:
                    best_neighbors = n_neighbors
                    f1_score_final = copy.deepcopy(f1_score_temp)
                    model_final = copy.deepcopy(model_temp)
        except:
            ### if it fails for any reason, just try auto and the smallest size of n_neighbors
            model_temp, f1_score_temp = train_evaluate_adasyn(X_train, y_train, X_val, y_val,
                                        final_estimator=self.final_estimator,
                                        n_neighbors=3, sampling_strategy='auto', 
                                        class_weights_dict=class_weights_dict)
            f1_score_final = copy.deepcopy(f1_score_temp)
            model_final = copy.deepcopy(model_temp)
            
        print('best neighbors for ADASYN selected = ', best_neighbors)
        ### training the final model on full X and y before sending it out
        adasyn = ADASYN(n_neighbors=best_neighbors, 
            sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        model_final.fit(X_resampled, y_resampled)
        self.final_model = model_final
        return self
    
    def predict(self, X):
        return self.final_model.predict(X)
    
    def predict_proba(self, X):
        return self.final_model.predict_proba(X)
##########################################################################
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class VariationalAutoEncoder(BaseEstimator, TransformerMixin):
    """
    Variational Autoencoder (VAE) for feature extraction in multi-class classification problems.

    This transformer applies a VAE to the input data, which is beneficial for capturing
    the underlying probability distribution of features. It's useful in scenarios like
    imbalanced datasets, complex feature interactions, or when data augmentation is required.

    Recommendation: Try VAE with MinMax Scaling which is good. 
            Even better try VAE_ADD with MinMax scaling which might improve performance.

    Key Enhancements:
    ##################
    Flexible architecture: This works for both Regression and Classification problems.
    Encoder with SELU Activation: The encoder uses the SELU activation 
        functions for its dense layers, providing self-normalizing properties 
        that can be particularly beneficial in deep neural networks
    Adjusting Encoder and Decoder: Use similar structures and activations, 
        ensuring they are suitable for the type of data ie. tabular data.
    Training Process: Adapt the training procedure to optimize both the 
        reconstruction and latent aspects of the model.
    Generative Capabilities: Include methods to generate new data samples 
        from the learned latent space.
    Sampling Layer: Integrated for the probabilistic encoding of the latent space.
    Encoder and Decoder: Both parts are constructed to suit tabular data, with 
        a structure that's more appropriate for non-image data.
    Loss Function: The loss function now includes the Kullback-Leibler divergence term, 
        important for the VAE's generative properties.
    Latent Space and Reconstruction: The transform method returns the latent 
        representation of the data, and a generate method is added to create new 
        data samples from the latent space.
    Dynamic Network Depth: The number of hidden layers in both the encoder and 
        decoder is adjusted based on the input dimension.
    Layer Size: The neuron count for each layer is dynamically set based 
        on the input dimension.
    Loss Function Adjustment: The loss function calculation is adjusted to ensure proper 
        scaling of the reconstruction and KL divergence losses.

    Parameters
    ----------
    intermediate_dim : int, default=64
        The dimension of the intermediate (hidden) layer in the encoder and decoder networks.

    latent_dim : int, default=4
        The dimension of the latent space (bottleneck layer).

    epochs : int, default=50
        The number of epochs for training the VAE.

    batch_size : int, default=128
        The batch size used during the training of the VAE.

    learning_rate : float, default=0.001
        The learning rate for the optimizer in the training process.

    Attributes
    ----------
    vae : keras.Model
        The complete Variational Autoencoder model.

    encoder : keras.Model
        The encoder part of the VAE, used for feature extraction.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This method trains the VAE model.

    transform(X)
        Apply the VAE to reduce the dimensionality of the data, returning the encoded features.

    Examples
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> vae = VariationalAutoEncoder()
    >>> vae.fit(X_train_scaled)
    >>> encoded_X_train = vae.transform(X_train_scaled)

    Notes:
    Here are the recommended values for ae_options dictionary for VAE:
    vae_dicto = {
    'intermediate_dim': 32,
    'latent_dim': 4,
    'epochs': 100, 
    'batch_size': 32,
    'learning_rate': 0.001
         }

    """
    try:
        from tensorflow.keras import layers
    except:
        print('tensorflow>= 2.5 not installed in machine. Please install and try again. ')

    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        def call(self, inputs):
            import tensorflow as tf
            def set_seed(seed=42):
                np.random.seed(seed)
                random.seed(seed)
                tf.random.set_seed(seed)
            set_seed(42)
            from tensorflow.keras import backend as K
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def __init__(self, latent_dim=2, intermediate_dim=64, epochs=50, 
                    batch_size=32, learning_rate=0.001):
        try:
            import tensorflow as tf
            def set_seed(seed=42):
                np.random.seed(seed)
                random.seed(seed)
                tf.random.set_seed(seed)
            set_seed(42)
            from tensorflow import keras
        except:
            print('tensorflow>= 2.5 not installed in machine. Please install and try again. ')
        self.latent_dim = latent_dim
        self.original_latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.original_intermediate_dim = intermediate_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.original_batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = None
        self.tasktype = 'Classification'
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10,
                        verbose=1, mode='min', baseline=None, restore_best_weights=False)
        # Learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                            patience=10, min_lr=0.0001)

        self.callbacks = [es, lr_scheduler]

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers, Model, backend as K
        from tensorflow.keras.layers import Dense, Input, LeakyReLU, BatchNormalization, Activation, Dropout, Add, Lambda
        from tensorflow.keras.losses import mse

        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)
        # Manually specify the activation function of the last layer
        # Adjust based on your model's specific configuration
        if self.task_type == 'Regression':
            last_layer_activation = 'linear'
        else:
            last_layer_activation = 'sigmoid'  # or 'linear', as appropriate

        # Dynamically adjust the depth and width based on input dimension
        hidden_layers = max(2, min(6, self.input_dim // 20))  # Between 2 and 6 hidden layers
        layer_size = max(32, min(256, self.input_dim * 2))  # Layer size between 32 and 256 neurons

        # Encoder
        original_inputs = tf.keras.Input(shape=(self.input_dim,), name='encoder_input')
        x = original_inputs
        for _ in range(hidden_layers):
            x = layers.Dense(layer_size, activation='selu')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Dropout(0.1)(x)  # Assuming a dropout rate of 10%
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = self.Sampling()([z_mean, z_log_var])
        encoder = Model(inputs=original_inputs, outputs=[z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,), name='z_sampling')
        x = latent_inputs
        for _ in range(hidden_layers):
            x = layers.Dense(layer_size, activation='relu')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Dropout(0.1)(x)

        outputs = layers.Dense(self.input_dim, activation=last_layer_activation)(x)
        decoder = Model(inputs=latent_inputs, outputs=outputs, name='decoder')

        # VAE Model
        outputs = decoder(encoder(original_inputs)[2])
        vae = Model(inputs=original_inputs, outputs=outputs, name='vae')

        # Adjust the reconstruction loss depending on the activation function of the last layer
        if last_layer_activation == 'sigmoid':
            reconstruction_loss = mse(K.flatten(original_inputs), K.flatten(outputs))
        else:
            reconstruction_loss = mse(original_inputs, outputs)
        reconstruction_loss *= self.input_dim
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        epsilon = 1e-7  # Small epsilon for numerical stability
        vae_loss = K.mean(reconstruction_loss + kl_loss + epsilon)
        vae.add_loss(vae_loss)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

        return vae, encoder, decoder

    def fit(self, X, y=None):
        self.input_dim = X.shape[1]
        if not y is None:
            self.task_type = analyze_problem_type_array(y)
        # Adjust dimensions if they exceed the number of features
        n_features = X.shape[1]
        self.intermediate_dim = min(self.original_intermediate_dim, int(3*n_features/4))
        self.latent_dim = max(self.original_latent_dim, int(n_features/4))

        # Adjust batch size based on the sample size of X
        n_samples = X.shape[0]
        self.batch_size = min(self.original_batch_size, int(n_samples/10))

        self.vae, self.encoder, self.decoder = self._build_model()
        print('Using Variational Auto Encoder to extract features...')

        self.vae.fit(X, X, epochs=self.epochs, batch_size=self.batch_size,
            callbacks=self.callbacks, shuffle=True, validation_split=0.20,
            verbose=1)
        return self

    def transform(self, X, y=None):
        return self.encoder.predict(X)[0]

    def generate(self, num_samples):
        z_sample = np.random.normal(size=(num_samples, self.latent_dim))
        return self.decoder.predict(z_sample)

#########################################################################################
import numpy as np
import pandas as pd

class GAN:
    """
    A Generative Adversarial Network (GAN) for generating synthetic data.

    This GAN implementation consists of a generator and a discriminator that are trained in tandem. The generator learns to produce data that resembles a given dataset, while the discriminator learns to distinguish between real and generated data.

    Parameters:
    ----------
    input_dim : int (default=10)
        The dimension of the input vector to the generator, typically the dimension of the noise vector.

    embedding_dim : int (default=50)
        The dimension of the embeddings in the hidden layers of the generator and discriminator.

    output_dim : int (default same as X's number of features )
        The dimension of the output vector from the generator, which should match the dimension of the real data.

    epochs : int, optional (default=200)
        The number of epochs to train the GAN.

    batch_size : int, optional (default=32)
        The size of the batches used during training.

    Methods
    -------
    fit(X, y=None)
        Train the GAN on the given data. The method alternately trains the discriminator and the generator.

    generate_data(num_samples)
        Generate synthetic data using the trained generator.

    Attributes
    ----------
    generator : keras.Model
        The generator component of the GAN.

    discriminator : keras.Model
        The discriminator component of the GAN.

    Examples
    --------
    >>> gan = GAN(input_dim=100, embedding_dim=50, output_dim=10, epochs=200, batch_size=32)
    >>> gan.fit(X_train)
    >>> synthetic_data = gan.generate_data(num_samples=1000)

    Notes
    -----
    GANs are particularly useful for data augmentation, domain adaptation, and as 
    a component in more complex generative models. They require careful tuning of 
    parameters and architecture for stable and meaningful output.

    Highlights of the model:
    ### This can be used only for Classification problems. It does not work for Regression problems.
    ### Early Stopping: The implementation of early stopping in GANs can be a bit tricky, 
    ### as it typically involves monitoring a validation metric. GANs don't use validation 
    ### data in the same way as traditional supervised learning models. 
    ### You might need to devise a custom criterion for early stopping based on 
    ### the generator's or discriminator's performance.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
    except:
        print('tensorflow>= 2.5 not installed in machine. Please install and try again. ')

    def __init__(self, input_dim, embedding_dim, output_dim, epochs=200, batch_size=32):
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import Model
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Initialize optimizers
        self.gen_optimizer = Adam()
        self.disc_optimizer = Adam()

    class Generator(Model):
        def __init__(self, input_dim, embedding_dim, output_dim):
            super().__init__()
            from tensorflow.keras.layers import Dense, Input, Activation, Dropout
            import tensorflow as tf
            def set_seed(seed=42):
                np.random.seed(seed)
                random.seed(seed)
                tf.random.set_seed(seed)
            set_seed(42)
            
            self.dense1 = Dense(embedding_dim, input_shape=(input_dim,))
            self.relu1 = Activation('relu')
            self.dense2 = Dense(embedding_dim * 2)
            self.relu2 = Activation('relu')
            self.dense3 = Dense(output_dim)
            self.sigmoid = Activation('sigmoid')

        def call(self, x):
            from tensorflow.keras.layers import Dense, Input, Activation, Dropout
            import tensorflow as tf
            def set_seed(seed=42):
                np.random.seed(seed)
                random.seed(seed)
                tf.random.set_seed(seed)
            set_seed(42)

            x = self.dense1(x)
            x = self.relu1(x)
            x = self.dense2(x)
            x = self.relu2(x)
            x = self.dense3(x)
            return self.sigmoid(x)

    class Discriminator(Model):
        def __init__(self, input_dim):
            super().__init__()
            from tensorflow.keras.layers import Dense, Input, Activation, Dropout
            import tensorflow as tf
            def set_seed(seed=42):
                np.random.seed(seed)
                random.seed(seed)
                tf.random.set_seed(seed)
            set_seed(42)

            self.dense1 = Dense(input_dim, input_shape=(input_dim,))
            self.leaky_relu = Activation(tf.nn.leaky_relu)
            self.dropout = Dropout(0.3)
            self.dense2 = Dense(1)
            self.sigmoid = Activation('sigmoid')

        def call(self, x):
            x = self.dense1(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
            x = self.dense2(x)
            return self.sigmoid(x)

    def _build_generator(self):
        return self.Generator(self.input_dim, self.embedding_dim, self.output_dim)

    def _build_discriminator(self):
        return self.Discriminator(self.output_dim)

    def fit(self, X, y):
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from sklearn.base import BaseEstimator
        from tensorflow.keras.losses import BinaryCrossentropy, mse

        # Define the loss function
        loss_fn = BinaryCrossentropy()

        # Initialize a variable to keep track of the best discriminator loss
        best_disc_loss = float('inf')
        patience = 10  # You can adjust this value as needed

        for epoch in range(self.epochs):
            # Shuffle the dataset
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]

            for i in range(0, len(X), self.batch_size):
                real_data = X[i:i + self.batch_size]
                noise = tf.random.normal([len(real_data), self.input_dim])

                # Train Discriminator
                with tf.GradientTape() as disc_tape:
                    fake_data = self.generator(noise, training=True)
                    real_output = self.discriminator(real_data, training=True)
                    fake_output = self.discriminator(fake_data, training=True)

                    real_loss = loss_fn(tf.ones_like(real_output), real_output)
                    fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
                    disc_loss = real_loss + fake_loss

                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                # Train Generator
                with tf.GradientTape() as gen_tape:
                    generated_data = self.generator(noise, training=True)
                    gen_data_discriminated = self.discriminator(generated_data, training=True)
                    gen_loss = loss_fn(tf.ones_like(gen_data_discriminated), gen_data_discriminated)

                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

            if epoch % 10 == 0:
                ### print every 10 epochs ###
                print(f"Epoch {epoch+1}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")

            # Check if the current discriminator loss is better than the best so far
            if disc_loss < best_disc_loss:
                best_disc_loss = disc_loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1  # Increment patience counter

            # If the discriminator loss hasn't improved for 'patience' epochs, stop training
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch+1} because discriminator loss has stopped improving.")
                break

    def generate_data(self, num_samples):
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)

        noise = tf.random.normal([num_samples, self.input_dim])
        return self.generator(noise).numpy()

###########################################################################
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y

class GANAugmenter(BaseEstimator, TransformerMixin):
    """
    A scikit-learn-style transformer for augmenting tabular datasets using Generative Adversarial Networks (GANs).

    This transformer trains a separate GAN for each class in the dataset to generate synthetic data, which is then used to augment the dataset. It is particularly useful for handling imbalanced datasets by generating more samples for underrepresented classes.

    Parameters:
    ----------
    gan_model : GAN class (default=None)
        A GAN class that has fit and generate_data methods. You can leave it as None.
        This class will be created and instantiated for each class in the dataset automatically.

    input_dim : int
        Refers to the dimension of the input noise vector to the generator. This is a 
        hyperparameter that can be tuned. A larger input dimension can provide the 
        generator with more capacity to capture complex data distributions, but it 
        also increases the model's complexity and training time.

    embedding_dim : int
        The dimension of the embeddings in the hidden layers of the GAN.

    epochs : int
        The number of epochs to train each GAN.

    num_synthetic_samples : int
        The number of synthetic samples to generate for each class.

    Methods
    -------
    fit(X, y)
        Fit the transformer to the data by training a separate GAN for each class found in y.

    transform(X, y)
        Augment the data by generating synthetic data using the trained GANs and combining it with the original data.

    Attributes
    ----------
    gans : dict
        A dictionary storing the trained GANs for each class.

    Examples
    --------
    >>> gan_model = GAN
    >>> gan_augmenter = GANAugmenter(gan_model, embedding_dim=100, epochs=200, num_synthetic_samples=1000)
    >>> gan_augmenter.fit(X_train, y_train)
    >>> X_train_augmented, y_train_augmented = gan_augmenter.transform(X_train, y_train)

    Here are recommended values for ae_options for GANAugmenter:
    gan_dicto = {
        'gan_model':None,
        'input_dim': 10,
        'embedding_dim': 100, 
        'epochs': 100, 
        'num_synthetic_samples': 400,
             }

    Notes
    -----
    The GANAugmenter can be used only for Classifications. It does not work for Regression problems.

    The GANAugmenter is useful in scenarios where certain classes in a dataset are underrepresented. 
    By generating additional synthetic samples, it can help create a more balanced dataset, 
    which can be beneficial for training machine learning models.

    The GANAugmenter class initializes a pre-made GAN model where the input and output 
    dimensions are the same. This design choice is made under the assumption that the 
    GAN is being used for data augmentation, where the goal is typically to generate 
    synthetic data that has the same shape and structure as the original input data.    

    In the context of data augmentation with GANs: Would Different Dimensions Improve Performance?
    If you're referring to the input noise dimension, varying its size could potentially 
    impact the GAN's ability to learn complex data distributions. It's a hyperparameter 
    that can be experimented with. However, this doesn't directly translate to "better" 
    performance; it's more about finding the right capacity for the model to capture 
    the necessary level of detail in the data.

    The output dimension, however, should typically match the dimensionality of your 
    real data. Changing the output dimension would mean the generated data no longer 
    aligns with the feature space of your original dataset, which defeats the purpose 
    of augmentation for tasks like classification or regression on the same feature set.    

    """
    
    def __init__(self, gan_model=None, input_dim = None, 
                embedding_dim=100, epochs=25, num_synthetic_samples=1000):
        """
        A transformer that trains GANs for each class to generate synthetic data.

        Parameters:
        gan_model: A GAN model class with fit and generate_data methods.
        embedding_dim: The embedding dimension for the GAN.
        epochs: The number of training epochs for each GAN.
        num_synthetic_samples: Number of synthetic samples to generate for each class.
        """
        try:
            import tensorflow as tf
            def set_seed(seed=42):
                np.random.seed(seed)
                random.seed(seed)
                tf.random.set_seed(seed)
            set_seed(42)
            from tensorflow.keras.optimizers import Adam
        except:
            print('tensorflow>= 2.5 not installed in machine. Please install and try again. ')

        if gan_model is None:
            self.gan_model = GAN
        else:
            self.gan_model = gan_model
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.input_dim = input_dim
        self.num_synthetic_samples = num_synthetic_samples
        self.gans = {}

    def fit(self, X, y):
        """
        Fit a separate GAN for each class in y.

        Parameters:
        X: Feature matrix
        y: Target vector
        """
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)
        import math
        def rlog(x, base):
          return math.log(x) / math.log(base)

        X, y = check_X_y(X, y)

        # Fit a GAN for each class
        for class_label in np.unique(y):
            X_class = X[y == class_label]
            if self.input_dim is None:
                ### if input dimension is not givem use the same size of X's features
                self.input_dim = int(rlog(X_class.shape[1], 4))*5

            ### Don't change this! this needs to be same as X's features
            output_dim = X_class.shape[1]

            gan = self.gan_model(self.input_dim, self.embedding_dim, output_dim, self.epochs)
            gan.fit(X_class, y)

            self.gans[class_label] = gan

        print('Input dimension given as %s for GAN. Try different input_dim values to tune GAN if needed.' %self.input_dim)
        return self

    def transform(self, X, y):
        """
        Generate synthetic data using the trained GANs and combine it with X.

        Parameters:
        X: Feature matrix to be augmented

        Returns:
        combined_data: The augmented feature matrix
        combined_labels: The labels for the augmented feature matrix
        """
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)

        synthetic_data_list = []
        synthetic_labels_list = []

        for class_label, gan in self.gans.items():
            synthetic_data = gan.generate_data(self.num_synthetic_samples)
            synthetic_data_list.append(synthetic_data)
            synthetic_labels = np.full(self.num_synthetic_samples, class_label)
            synthetic_labels_list.append(synthetic_labels)

        all_synthetic_data = np.vstack(synthetic_data_list)
        all_synthetic_labels = np.concatenate(synthetic_labels_list)

        combined_data = np.vstack([X, all_synthetic_data])
        combined_labels = np.concatenate([y, all_synthetic_labels])

        return combined_data, combined_labels
#############################################################################
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import copy
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
######################################################################################

class DenoisingAutoEncoder(BaseEstimator, TransformerMixin):
    """
    A Denoising Autoencoder for tabular data. It is designed to learn a representation 
    that is robust to noise, which can be useful for feature extraction and data denoising.

    DENOISING Autoencoders were first introduced in a research paper by: Pascal Vincent et al. 
        "Extracting and Composing Robust Features with Denoising Autoencoders"
        Appearing in Proceedings of the 25th International Conference on Machine Learning, 
        Helsinki, Finland, 2008. Copyright 2008 by the author(s)/owner(s).

    This transformer adds noise to the input data and then trains an autoencoder to 
    reconstruct the original data, thereby learning robust features. It can automatically 
    select between a simple and a complex architecture based on the size of the dataset, 
    or this selection can be overridden by user input.

    Key highlights of this model:
    #############################
    Flexible architecture: This Denoising AE Works for all types of problems: regression, 
        binary classification and multi-class classification.
    Noise Introduction: Added methods to introduce noise ('gaussian' or 'dropout') to the input
         data, making the model suitable for denoising tasks.
    Flexibility in Architecture: The architecture is kept relatively simple but can be 
        expanded based on the complexity of the tabular data.
    Binary Crossentropy Loss: Suitable for a range of tabular data, especially when
         normalized between 0 and 1.
    Dynamic Network Depth: The number of hidden layers is determined based on the input dimension, 
        allowing for a deeper network for high-dimensional data.
    Layer Size: The size of each layer is also dynamically set based on the input dimension, 
        ensuring the network has sufficient capacity to model complex data relationships.
    Range Limits: Both the number of layers and the size of each layer are constrained within 
        reasonable ranges to avoid overly large models, especially for very high-dimensional data.

    Parameters:
    ----------
    encoding_dim : int
        The size of the encoded representation.

    noise_type : str, optional (default='gaussian')
        Type of noise to use for denoising. Options: 'gaussian', 'dropout'.

    noise_factor : float, optional (default=0.1)
        The level or intensity of noise to add. For 'gaussian', it's the standard deviation.
        For 'dropout', it's the dropout rate.

    learning_rate : float, optional (default=0.001)
        The learning rate for the optimizer.

    epochs : int, optional (default=50)
        The number of epochs to train the autoencoder.

    batch_size : int, optional (default=32)
        The batch size used during training.

    Attributes
    ----------
    autoencoder : keras.Model
        The complete autoencoder model.

    encoder : keras.Model
        The encoder part of the autoencoder model.

    Methods
    -------
    fit(X, y=None)
        Fit the autoencoder model to the data.

    transform(X)
        Apply the dimensionality reduction learned by the autoencoder.

    Examples
    -------------
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('feature_extractor', DenoisingAutoEncoder()),
        ('classifier', LogisticRegression())
        ])
    pipeline.fit(X_train)
    y_preds = pipeline.predict(X_test)
    y_probas = pipeline.predict_proba(X_test)
    --------------
    Notes:
        Here are the recommended values for ae_options dictionary for DAE:
        dae_dicto = {
        'noise_factor': 0.2,
        'encoding_dim': 10,
        'epochs': 100, 
        'batch_size': 32,        
         }

    """
    def __init__(self, encoding_dim=8, noise_type='gaussian', noise_factor=0.2,
                 learning_rate=0.001, epochs=1, batch_size=32):
        #### Try out the imports and see if they work here ###
        try:
            import tensorflow as tf
            def set_seed(seed=42):
                np.random.seed(seed)
                random.seed(seed)
                tf.random.set_seed(seed)
            set_seed(42)
            from tensorflow import keras
            from tensorflow.keras.models import Model
            from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
        except:
            print('tensorflow>= 2.5 not installed in machine. Please install and try again. ')

        self.encoding_dim = encoding_dim
        self.noise_type = noise_type
        self.noise_factor = noise_factor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        reduce_lr = ReduceLROnPlateau(monitor='val_mse', factor=0.90, patience=5, min_lr=0.0001)
        early_stopping = EarlyStopping(monitor='val_mse', patience=5, restore_best_weights=True)
        callbacks = [early_stopping]
        self.callbacks = callbacks
        self.input_dim = None
        self.autoencoder = None
        self.encoder = None

    def _add_noise(self, X):
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_factor, X.shape)
            #  By ensuring that data stays in [0, 1] range we assume that we use MinMax scaler
            ### This would apply to both Regression and Classification tasks since we are 
            ###   scaling only X which will lie within the 0 to 1 range.
            return np.clip(X + noise, 0, 1)  
        elif self.noise_type == 'dropout':
            # Not using Dropout layer here since this is a manual injection before training
            dropout_mask = np.random.binomial(1, 1 - self.noise_factor, X.shape)
            return X * dropout_mask
        else:
            raise ValueError("Invalid noise_type. Expected 'gaussian' or 'dropout'.")

    def _build_model(self):
        from tensorflow.keras.layers import Dense, Input, LeakyReLU, BatchNormalization, Activation, Dropout, Add, Lambda
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        inputs = Input(shape=(self.input_dim,))
        x = inputs
        min_layers = max(3, self.input_dim // 20)

        ### set the basic size of neurons here ##
        if self.input_dim <= 5:
            base_size = min(512, self.input_dim*20)  # Increased the potential size for more complexity
        else:
            base_size = min(512, self.input_dim*10)  # Increased the potential size for more complexity

        # Encoder - DO NOT MODIFY THIS SINCE IT GIVES A PERFECTLY BALANCED NETWORK
        layer_num = min(6, min_layers)
        for ee in range(layer_num-1):
            x = Dense(max(16, base_size // (ee+2))) (x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)  # Using LeakyReLU
            x = Dropout(self.noise_factor)(x)

        # Bottleneck
        encoded = Dense(self.encoding_dim, activation='relu')(x)

        # Decoder - DO NOT MODIFY THIS SINCE IT GIVES A PERFECTLY BALANCED NETWORK
        x = encoded
        x = Dense(self.encoding_dim, activation='relu') (x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)  # Using LeakyReLU
        x = Dropout(self.noise_factor)(x)
        for dd in range(layer_num-1):
            x = Dense(max(16, base_size // (layer_num-dd) )) (x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)  # Using LeakyReLU
            x = Dropout(self.noise_factor)(x)

        # Output layer
        x = Dense(self.input_dim, activation='linear')(x)  # Linear activation for regression and Classification

        # Add residual connections if the dimensions allow for it
        if self.input_dim == max(32, min(256, self.input_dim * 2)):
            x = Add()([x, inputs])

        # Create autoencoder model
        autoencoder = Model(inputs, x)
        loss_function = 'mse'
        metrics = 'mse'

        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                        loss=loss_function,
                        metrics=[metrics])

        # Create encoder model
        encoder = Model(inputs, encoded)

        return autoencoder, encoder


    def fit(self, X, y=None):
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)

        self.input_dim = X.shape[1]
        X_noisy = self._add_noise(X)
        self.autoencoder, self.encoder = self._build_model()
        self.autoencoder.fit(X_noisy, X, epochs=self.epochs, batch_size=self.batch_size,
                             shuffle=True, callbacks=self.callbacks, validation_split=0.2)
        return self

    def transform(self, X, y=None):
        return self.encoder.predict(X)

def dae_hyperparam_selection(dae, X_train, y_train):
    if X_train.max().max() > 1.0:
        print('    defining a pipeline with MinMaxScaler and DAE')
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('feature_extractor', dae),
        ])
    else:
        print('    defining a pipeline with DAE')
        pipeline = Pipeline([
            ('feature_extractor', dae),
        ])

    #### This is for DEAFeatureExtractor #####
    param_grid = {
        'feature_extractor__batch_size': [16, 32, 64],
        'feature_extractor__encoding_dim': [5, 10, 15],
        #'feature_extractor__noise_type': ['gaussian', 'dropout'],
        #'feature_extractor__noise_factor': [0.1, 0.2],
        'feature_extractor__epochs': [3],
        #'feature_extractor__learning_rate': [0.001, 0.01],    
    }

    # Setup grid search
    grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', 
                               cv=3, n_jobs=-1,verbose=1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    return grid_search

##########################################################################
