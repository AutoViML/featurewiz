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
    if y_train.ndim <= 1:
        multi_label = False
    else:
        multi_label = True
    ####  This is where you detect what kind of problem it is #################
    if not multi_label:
        if  y_train.dtype in ['int64', 'int32','int16']:
            if len(np.unique(y_train)) <= 2:
                model_class = 'Binary_Classification'
            elif len(y_train.unique()) > 2 and len(y_train.unique()) <= cat_limit:
                model_class = 'Multi_Classification'
            else:
                model_class = 'Regression'
        elif  y_train.dtype in ['float16','float32','float64']:
            if len(y_train.unique()) <= 2:
                model_class = 'Binary_Classification'
            elif len(y_train.unique()) > 2 and len(y_train.unique()) <= float_limit:
                model_class = 'Multi_Classification'
            else:
                model_class = 'Regression'
        else:
            if len(y_train.unique()) <= 2:
                model_class = 'Binary_Classification'
            else:
                model_class = 'Multi_Classification'
    else:
        for i in range(y_train.ndim):
            ### if target is a list, then we should test dtypes a different way ###
            if y_train.dtypes.all() in ['int64', 'int32','int16']:
                if len(np.unique(y_train.iloc[:,0])) <= 2:
                    model_class = 'Binary_Classification'
                elif len(np.unique(y_train.iloc[:,0])) > 2 and len(np.unique(y_train.iloc[:,0])) <= cat_limit:
                    model_class = 'Multi_Classification'
                else:
                    model_class = 'Regression'
            elif  y_train.dtypes.all() in ['float16','float32','float64']:
                if len(np.unique(y_train.iloc[:,0])) <= 2:
                    model_class = 'Binary_Classification'
                elif len(np.unique(y_train.iloc[:,0])) > 2 and len(np.unique(y_train.iloc[:,0])) <= float_limit:
                    model_class = 'Multi_Classification'
                else:
                    model_class = 'Regression'
            else:
                if len(np.unique(y_train.iloc[:,0])) <= 2:
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

class StackingClassifier_Multi(BaseEstimator, ClassifierMixin):
    
    def __init__(self, final_estimator=None):
        print('initialized')
        # Compute class weights
        self.final_model = None
        self.final_estimator = final_estimator

    def fit(self, X, y):
        class_weights_dict = get_class_distribution(y_train)
        modeltype, multi_label = analyze_problem_type_array(y_train)
        
        # Function to train and evaluate the model with ADASYN resampling
        if modeltype == 'Regression':
            X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                              test_size=0.10, random_state=1,)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                              test_size=0.10, 
                                                            stratify=y_train,
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
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DenoisingAutoEncoder(BaseEstimator, TransformerMixin):
    """
    A denoising autoencoder transformer for feature extraction from tabular datasets. 
    
    This implementation is based on a research paper by: 
    Pascal Vincent et al. "Extracting and Composing Robust Features with Denoising Autoencoders"
    Appearing in Proceedings of the 25th International Conference on Machine Learning, Helsinki, Finland, 
    2008.Copyright 2008 by the author(s)/owner(s).

    This transformer adds noise to the input data and then trains an autoencoder to 
    reconstruct the original data, thereby learning robust features. It can automatically 
    select between a simple and a complex architecture based on the size of the dataset, 
    or this selection can be overridden by user input.

    Recommendation: Best is DAE with MinMax Scaling. But DAE_ADD also gives similar but slightly less performance.

    Parameters
    ----------
    encoding_dim : int, default=50
        The size of the encoding layer. This determines the dimensionality of the output 
        features from the transformer.

    noise_factor : float, default=0.1
        The factor by which noise is added to the input data. Noise is generated from a 
        normal distribution and scaled by this factor.

    learning_rate : float, default=0.001
        The learning rate for the Adam optimizer used in training the autoencoder.

    epochs : int, default=100
        The number of epochs to train the autoencoder.

    batch_size : int, default=16
        The batch size used during the training of the autoencoder.

    callbacks : list of keras.callbacks.Callback, optional
        Callbacks to apply during training of the autoencoder.

    simple_architecture : bool or None, default=None
        If set to True, the transformer always uses a simple architecture regardless of the dataset size.
        If set to False, it always uses a more complex architecture. If set to None, the architecture
        is chosen automatically based on the dataset size (simple for less than 10,000 samples).

    Attributes
    ----------
    autoencoder : keras.Model
        The complete autoencoder model.

    encoder : keras.Model
        The encoder part of the autoencoder model, used for feature extraction.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This method trains the autoencoder model.

    transform(X)
        Apply the dimensionality reduction learned by the autoencoder, returning the encoded features.

    Examples
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> dae = DenoisingAutoEncoder()
    >>> dae.fit(X_train_scaled, y_train)
    >>> encoded_X_train = dae.transform(X_train_scaled)
    >>> encoded_X_test = dae.transform(X_test_scaled)

    Notes:
    Here are the recommende values for ae_options dictionary for DAE:
    dae_dicto = {
        'noise_factor': 0.2,
        'encoding_dim': 10,
        'epochs': 100, 
        'batch_size': 32,
        'simple_architecture': None
         }

    """

    def __init__(self, encoding_dim=50, noise_factor=0.1, 
                 learning_rate=0.001, epochs=100, batch_size=16, 
                 callbacks=None, simple_architecture=None):
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
        self.encoding_dim = encoding_dim
        self.noise_factor = noise_factor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        if callbacks is None:
            es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5,
                            verbose=1, mode='min', baseline=None, restore_best_weights=False)
            self.callbacks = [es]
        else:
            self.callbacks = callbacks
        self.simple_architecture = simple_architecture
        self.autoencoder = None
        self.encoder = None

    def _build_autoencoder(self, input_dim):
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.models import Model
        
        input_layer = Input(shape=(input_dim,))
        if self.simple_architecture or (self.simple_architecture is None and input_dim < 10000):
            print('Performing Denoising Auto Encoder transform using Simple architecture...')
            encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='sigmoid')(encoded)
        else:
            print('Performing Denoising Auto Encoder transform using Complex architecture...')
            encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
            encoded = Dense(self.encoding_dim // 2, activation='relu')(encoded)
            decoded = Dense(self.encoding_dim, activation='relu')(encoded)
            decoded = Dense(input_dim, activation='sigmoid')(decoded)

        return Model(input_layer, decoded), Model(input_layer, encoded)

    def fit(self, X, y=None):
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)

        # Add noise to the training data
        X_noisy = X + self.noise_factor * np.random.normal(size=X.shape)

        # Build the autoencoder
        self.autoencoder, self.encoder = self._build_autoencoder(X.shape[1])
        
        # Compile and train the autoencoder
        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                                 loss='mean_squared_error')
        self.autoencoder.fit(X_noisy, X, batch_size=self.batch_size, epochs=self.epochs, 
                             callbacks=self.callbacks, shuffle=True, validation_split=0.20)

        return self

    def transform(self, X, y=None):
        # Extract the features from the input data
        encoded_X = self.encoder.predict(X)
        return encoded_X
#########################################################################
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

    def __init__(self, intermediate_dim=64, latent_dim=4, epochs=300, batch_size=64, learning_rate=0.001):
        self.original_intermediate_dim = intermediate_dim
        self.intermediate_dim = intermediate_dim
        self.original_latent_dim = latent_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.original_batch_size = batch_size
        self.learning_rate = learning_rate
        self.vae = None
        self.encoder = None
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
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5,
                        verbose=1, mode='min', baseline=None, restore_best_weights=False)
        # Learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                            patience=5, min_lr=0.0001)

        self.callbacks = [es, lr_scheduler]

    def _sampling(self, args):
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)
        from tensorflow.keras import backend as K
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def _build_vae(self, input_shape):
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)
        from tensorflow.keras.layers import Input, Dense, Lambda
        from tensorflow.keras.models import Model
        from tensorflow.keras.losses import mse
        from tensorflow.keras import backend as K

        # Manually specify the activation function of the last layer
        # Adjust based on your model's specific configuration
        last_layer_activation = 'sigmoid'  # or 'linear', as appropriate

        # Encoder
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        
        # Latent space
        z = Lambda(self._sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # Instantiate the encoder
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(input_shape[0], activation=last_layer_activation)(x)

        # Instantiate the decoder
        decoder = Model(latent_inputs, outputs, name='decoder')

        # VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae')

        # Adjust the reconstruction loss depending on the activation function of the last layer
        if last_layer_activation == 'sigmoid':
            reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        else:
            reconstruction_loss = mse(inputs, outputs)
        
        reconstruction_loss *= input_shape[0]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        epsilon = 1e-7  # Small epsilon for numerical stability
        vae_loss = K.mean(reconstruction_loss + kl_loss + epsilon)

        vae.add_loss(vae_loss)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)) 
        return vae, encoder


    def fit(self, X, y=None):
        # Adjust dimensions if they exceed the number of features
        n_features = X.shape[1]
        self.intermediate_dim = min(self.original_intermediate_dim, int(3*n_features/4))
        self.latent_dim = min(self.original_latent_dim, int(n_features/2))

        # Adjust batch size based on the sample size of X
        n_samples = X.shape[0]
        self.batch_size = min(self.original_batch_size, int(n_samples/10))

        self.vae, self.encoder = self._build_vae((n_features,))
        print('Using Variational Auto Encoder to extract features...')
        self.vae.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, 
            callbacks=self.callbacks, shuffle=True, validation_split=0.20,
            verbose=1)
        return self

    def transform(self, X, y=None):
        return self.encoder.predict(X)[0]
###############################################################################
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
        from tensorflow.keras.losses import BinaryCrossentropy

        # Define the loss function
        loss_fn = BinaryCrossentropy()

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
                embedding_dim=100, epochs=200, num_synthetic_samples=1000):
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
