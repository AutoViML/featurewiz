############          Credit for Blending Regressor        ############
####  Greatly indebted to Gilbert Tanner who created Blending Regressor
####   https://gilberttanner.com/blog/introduction-to-ensemble-learning
####  I have modifed his code to create a Stacking Classifier #########
#######################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error,auc
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings('ignore')
def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))
##################################################
### Define the input models here #######
###################################################
class Stacking_Classifier(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    ############          Credit for Blending Regressor        ############
    ####  Greatly indebted to Gilbert Tanner who created Blending Regressor
    ####   https://gilberttanner.com/blog/introduction-to-ensemble-learning
    ####  I have modifed his code to create a Stacking Classifier #########
    #######################################################################
    """
    def __init__(self):
        n_folds = 3
        logit = LogisticRegression(C=1.0, random_state = 1, max_iter=5000)
        DT = DecisionTreeClassifier(max_depth=10, random_state = 3)
        GBoost = LinearSVC(random_state=99)
        model_rf = RandomForestClassifier(max_depth=10,n_estimators=100,
                            random_state=99)
        xgbc = AdaBoostClassifier(random_state=0)
        gpc = MLPClassifier(hidden_layer_sizes=50, random_state=0)
        base_models =  (logit, model_rf, DT, GBoost, xgbc, gpc)
        meta_model = DT
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    #Do the predictions of all base models on the test data and use the averaged predictions as
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            stats.mode(np.column_stack([model.predict(X) for model in base_models]), axis=1)[0]
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
###################################################################
from sklearn.model_selection import train_test_split
import pathlib
from scipy import stats
from scipy.stats import norm, skew

from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import xgboost as xgb

class Blending_Regressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    ############          Credit for Blending Regressor        ############
    ####  Greatly indebted to Gilbert Tanner who created Blending Regressor
    ####   https://gilberttanner.com/blog/introduction-to-ensemble-learning
    ####  I have modifed his code to create a Stacking Classifier #########
    #######################################################################
    """
    def __init__(self, holdout_pct=0.2, use_features_in_secondary=False):
        # create models
        lasso_model = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
        rf = RandomForestRegressor()
        gbr = GradientBoostingRegressor()
        xgb_model = xgb.XGBRegressor()
        base_models = [gbr, rf, xgb_model, lasso_model]
        meta_model = lasso_model
        self.base_models = base_models
        self.meta_model = meta_model
        self.holdout_pct = holdout_pct
        self.use_features_in_secondary = use_features_in_secondary

    def fit(self, X, y):
        self.base_models_ = [clone(x) for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=self.holdout_pct)

        holdout_predictions = np.zeros((X_holdout.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models_):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_holdout)
            holdout_predictions[:, i] = y_pred
        if self.use_features_in_secondary:
            self.meta_model_.fit(np.hstack((X_holdout, holdout_predictions)), y_holdout)
        else:
            self.meta_model_.fit(holdout_predictions, y_holdout)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models_
        ])
        if self.use_features_in_secondary:
            return self.meta_model_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_model_.predict(meta_features)
######################################################################################
