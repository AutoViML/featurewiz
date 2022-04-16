import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
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

from sklearn.model_selection import train_test_split
import pathlib
from scipy import stats
from scipy.stats import norm, skew
import time
import copy
from collections import Counter
from collections import defaultdict
from collections import OrderedDict

from sklearn.linear_model import Lasso
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
    #### Modifed to create a Stacking Classifier that can handle multi-label targets
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
            stump = lgb.LGBMClassifier(n_estimators=50, random_state=99)
            self.meta_model = MultiOutputClassifier(stump)
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
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                if y.ndim < 2:
                    out_of_fold_predictions[holdout_index, i] = y_pred
                else:
                    next_i = int(i+self.target_len)
                    out_of_fold_predictions[holdout_index,i:next_i] = y_pred
        
        if self.use_features:
            self.meta_model_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_model_.fit(out_of_fold_predictions, y)
            
        print('    Time taken = %0.0f seconds' %(time.time()-start_time))
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
    #### Modifed to create a Stacking Regressor that can handle multi-label targets
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
            stump = lgb.LGBMRegressor(n_estimators=50, random_state=99)
            self.meta_model = MultiOutputRegressor(stump)
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
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                if y.ndim < 2:
                    out_of_fold_predictions[holdout_index, i] = y_pred
                else:
                    next_i = int(i+self.target_len)
                    out_of_fold_predictions[holdout_index,i:next_i] = y_pred
        
        if self.use_features:
            self.meta_model_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_model_.fit(out_of_fold_predictions, y)
            
        print('    Time taken = %0.0f seconds' %(time.time()-start_time))
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
    #### Modifed to create a Blending Regressor that can handle multi-label targets
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

    def fit(self, X, y):
        import lightgbm as lgb
        models_dict = stacking_models_list(X_train=X, y_train=y, modeltype='Regression', verbose=1)
        self.base_models = list(models_dict.values())
        self.base_models_ = [clone(x) for x in self.base_models]
        if y.ndim >= 2:
            stump = lgb.LGBMRegressor(n_estimators=50, random_state=99)
            self.meta_model = MultiOutputRegressor(stump)
        else:
            self.meta_model = lgb.LGBMRegressor(n_estimators=50, random_state=99)
        self.meta_model_ = clone(self.meta_model)

        start_time = time.time()
        model_name = str(self.meta_model).split("(")[0]
        print('Stacking model %s training started. This will take time...' %model_name)

        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=self.holdout_pct)

        holdout_predictions = np.zeros((X_holdout.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models_):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_holdout)
            if y_train.ndim < 2:
                holdout_predictions[:, i] = y_pred
            else:
                holdout_predictions = y_pred[:]
        if self.use_features:
            self.meta_model_.fit(np.hstack((X_holdout, holdout_predictions)), y_holdout)
        else:
            self.meta_model_.fit(holdout_predictions, y_holdout)

        print('    Time taken = %0.0f seconds' %(time.time()-start_time))
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models_
        ])
        if self.use_features:
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
            stump = RandomForestRegressor(random_state=seed, n_estimators=100)
            model1 = MultiOutputRegressor(stump)
            estimators.append(('Multi Output Regressor',model1))
        else:
            ######    Bagging models if Bagging is chosen ####
            model3 = KNeighborsRegressor(n_jobs=-1)
            estimators.append(('KNN',model3))
            model4 = LinearSVR()
            estimators.append(('Linear_SVR',model4))
            ####   Tree models if Linear chosen #####
            model5 = DecisionTreeRegressor(random_state=seed,min_samples_leaf=2)
            estimators.append(('Decision Trees',model5))
            ####   Linear Models if Boosting is chosen #####
            model6 = LassoCV(alphas=np.logspace(-5,-1,20), cv=5,random_state=seed)
            estimators.append(('LassoCV Regularization',model6))
            model7 = RandomForestRegressor(n_estimators=50,random_state=seed)
            estimators.append(('Random Forest',model7))
    else:
        if y_train.ndim >= 2:
            stump = RandomForestClassifier(random_state=seed, n_estimators=100)
            model1 = MultiOutputClassifier(stump)
            estimators.append(('Multi Output Classifier',model1))
        else:
            n_classes = len(Counter(y_train))
            if n_classes > 2:
                model3 = LogisticRegression(max_iter=5000, multi_class='ovr')
                estimators.append(('Logistic Regression',model3))
            else:
                model3 = KNeighborsClassifier(n_jobs=-1)
                estimators.append(('KNN',model3))
                
            ####   Linear Models if Boosting is chosen #####
            if n_classes > 2:
                model4 = RandomForestClassifier(n_estimators=100, random_state=99)
                estimators.append(('RandomForestClassifier',model4))
            else:
                model4 = LinearDiscriminantAnalysis()
                estimators.append(('Linear Discriminant',model4))
            
            ####   Tree models if Linear chosen #####
            model6 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50)
            estimators.append(('Adaboost Classifier',model6))
            
            ######    Naive Bayes models if Bagging is chosen ####
            if n_classes <= 2:
                try:
                    model7 = GaussianNB()
                except:
                    model7 = DecisionTreeClassifier(min_samples_leaf=2)
            else:
                try:
                    model7 = MultinomialNB()
                except:
                    model7 = DecisionTreeClassifier(min_samples_leaf=2)
            estimators.append(('Naive Bayes',model7))
        
    #stacks = np.c_[results1,results2,results3]
    estimators_list = [(tuples[0],tuples[1]) for tuples in estimators]
    estimator_names = [tuples[0] for tuples in estimators]
    print('List of models chosen for stacking: %s' %estimators_list)
    return dict(estimators_list)
#########################################################