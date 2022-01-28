import pandas as pd
import numpy as np
np.random.seed(99)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgbm
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
import csv
import re
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_log_error, mean_squared_error,balanced_accuracy_score
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
import scipy as sp
import time
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter, defaultdict
import pdb
from tqdm.notebook import tqdm
from pathlib import Path

#sklearn data_preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#sklearn categorical encoding
import category_encoders as ce

#sklearn modelling
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

# boosting library
import xgboost as xgb
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import copy
#################################################################################
#### Regression or Classification type problem
def analyze_problem_type(train, target, verbose=0) :
    target = copy.deepcopy(target)
    train = copy.deepcopy(train)
    if isinstance(train, pd.Series):
        train = pd.DataFrame(train)
    cat_limit = 30 ### this determines the number of categories to name integers as classification ##
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    if isinstance(target, str):
        target = [target]
    if len(target) == 1:
        targ = target[0]
        model_label = 'Single_Label'
    else:
        targ = target[0]
        model_label = 'Multi_Label'
    ####  This is where you detect what kind of problem it is #################
    if  train[targ].dtype in ['int64', 'int32','int16']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= cat_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    elif  train[targ].dtype in ['float16','float32','float64']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= float_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    else:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        else:
            model_class = 'Multi_Classification'
    ########### print this for the start of next step ###########
    if verbose <= 1:
        print('''#### %s %s Feature Selection Started ####''' %(
                                model_label,model_class))
    return model_class
#####################################################################################
from sklearn.base import TransformerMixin, BaseEstimator
from collections import defaultdict
class My_LabelEncoder(TransformerMixin):
    """
    ################################################################################################
    ######  This Label Encoder class works just like sklearn's Label Encoder!  #####################
    #####  You can label encode any column in a data frame using this new class. But unlike sklearn,
    the beauty of this function is that it can take care of NaN's and unknown (future) values.
    It uses the same fit() and fit_transform() methods of sklearn's LabelEncoder class.
    ################################################################################################
    ##################### This is the best working version - don't mess with it! ###################
    Usage:
          MLB = My_LabelEncoder()
          train[column] = MLB.fit_transform(train[column])
          test[column] = MLB.transform(test[column])
    """
    def __init__(self):
        self.transformer = defaultdict(str)
        self.inverse_transformer = defaultdict(str)

    def fit(self,testx):
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            return testx
        outs = np.unique(testx.factorize()[0])
        ins = testx.value_counts(dropna=False).index
        #if -1 in outs:
        #   it already has nan if -1 is in outs. No need to add it.
        #    ins.insert(0,np.nan)
        self.transformer = dict(zip(ins,outs.tolist()))
        self.inverse_transformer = dict(zip(outs.tolist(),ins))
        return self

    def transform(self, testx):
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            return testx
        ### now convert the input to transformer dictionary values
        ins = np.unique(testx.factorize()[1]).tolist()
        missing = [x for x in ins if x not in self.transformer.keys()]
        if len(missing) > 0:
            for each_missing in missing:
                max_val = np.max(list(self.transformer.values())) + 1
                self.transformer[each_missing] = max_val
                self.inverse_transformer[max_val] = each_missing
        outs = testx.map(self.transformer).values
        testk = testx.map(self.transformer)
        if testx.dtype not in [np.int16, np.int32, np.int64, float, bool, object]:
            if testx.isnull().sum().sum() > 0:
                fillval = self.transformer[np.nan]
                testk = testk.cat.add_categories([fillval])
                testk = testk.fillna(fillval)
                testk = testk.astype(int)
                return testk
            else:
                testk = testk.astype(int)
                return testk
        else:
            return outs

    def inverse_transform(self, testx):
        ### now convert the input to transformer dictionary values
        if isinstance(testx, pd.Series):
            outs = testx.map(self.inverse_transformer).values
        elif isinstance(testx, np.ndarray):
            outs = pd.Series(testx).map(self.inverse_transformer).values
        else:
            outs = testx[:]
        return outs
#################################################################################
from sklearn.impute import SimpleImputer
def data_transform(X_train, Y_train, X_test="", Y_test="", modeltype='Classification',
            multi_label=False, enc_method='label', scaler = StandardScaler()):
    ##### First make sure that the originals are not modified ##########
    X_train_encoded = copy.deepcopy(X_train)
    X_test_encoded = copy.deepcopy(X_test)
    ##### Use My_Label_Encoder to transform label targets if needed #####
    if multi_label:
        if modeltype != 'Regression':
            targets = Y_train.columns
            Y_train_encoded = copy.deepcopy(Y_train)
            for each_target in targets:
                mlb = My_LabelEncoder()
                Y_train_encoded[each_target] = mlb.fit_transform(Y_train[each_target])
                if not isinstance(Y_test, str):
                    Y_test_encoded= mlb.transform(Y_test)
                else:
                    Y_test_encoded = copy.deepcopy(Y_test)
        else:
            Y_train_encoded = copy.deepcopy(Y_train)
            Y_test_encoded = copy.deepcopy(Y_test)
    else:
        if modeltype != 'Regression':
            mlb = My_LabelEncoder()
            Y_train_encoded= mlb.fit_transform(Y_train)
            if not isinstance(Y_test, str):
                Y_test_encoded= mlb.transform(Y_test)
            else:
                Y_test_encoded = copy.deepcopy(Y_test)
        else:
            Y_train_encoded = copy.deepcopy(Y_train)
            Y_test_encoded = copy.deepcopy(Y_test)
    
    #### This is where we find out if test data is given ####
    ####### Set up feature to encode  ####################
    feature_to_encode = X_train.columns[X_train.dtypes == 'O'].tolist()
    #print('features to label encode: %s' %feature_to_encode)
    #### Do label encoding now #################
    if enc_method == 'label':
        for feat in feature_to_encode:
            # Initia the encoder model
            lbEncoder = My_LabelEncoder()
            # fit the train data
            lbEncoder.fit(X_train[feat])
            # transform training set
            X_train_encoded[feat] = lbEncoder.transform(X_train[feat])
            # transform test set
            if not isinstance(X_test_encoded, str):
                X_test_encoded[feat] = lbEncoder.transform(X_test[feat])
    elif enc_method == 'glmm':
        # Initialize the encoder model
        GLMMEncoder = ce.glmm.GLMMEncoder(verbose=0 ,binomial_target=False)
        # fit the train data
        GLMMEncoder.fit(X_train[feature_to_encode],Y_train_encoded)
        # transform training set  ####
        X_train_encoded[feature_to_encode] = GLMMEncoder.transform(X_train[feature_to_encode])
        # transform test set
        if not isinstance(X_test_encoded, str):
            X_test_encoded[feature_to_encode] = GLMMEncoder.transform(X_test[feature_to_encode])
    else:
        print('No encoding transform performed')

    ### make sure there are no missing values ###
    try:
        imputer = SimpleImputer(strategy='constant', fill_value=0, verbose=0, add_indicator=True)
        imputer.fit_transform(X_train_encoded)
        if not isinstance(X_test_encoded, str):
            imputer.transform(X_test_encoded)
    except:
        X_train_encoded = X_train_encoded.fillna(0)
        if not isinstance(X_test_encoded, str):
            X_test_encoded = X_test_encoded.fillna(0)

    # fit the scaler to the entire train and transform the test set
    scaler.fit(X_train_encoded)
    # transform training set
    X_train_scaled = pd.DataFrame(scaler.transform(X_train_encoded), 
        columns=X_train_encoded.columns, index=X_train_encoded.index)
    # transform test set
    if not isinstance(X_test_encoded, str):
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), 
            columns=X_test_encoded.columns, index=X_test_encoded.index)
    return X_train_scaled, Y_train_encoded, X_test_scaled, Y_test_encoded
##################################################################################
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
import csv
import re
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_log_error, mean_squared_error,balanced_accuracy_score
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
import scipy as sp
import time
##################################################################################
import lightgbm as lgbm
def lightgbm_model_fit(random_search_flag, x_train, y_train, x_test, y_test, modeltype,
                         multi_label, log_y, model=""):
    start_time = time.time()
    if multi_label:
        rand_params = {
                }
    else:
        rand_params = {
            'learning_rate': sp.stats.uniform(scale=1),
            'num_leaves': sp.stats.randint(20, 100),
           'n_estimators': sp.stats.randint(100,500),
            "max_depth": sp.stats.randint(3, 15),
                }

    if modeltype == 'Regression':
        lgb = lgbm.LGBMRegressor()
        objective = 'regression' 
        metric = 'rmse'
        is_unbalance = False
        class_weight = None
        score_name = 'Score'
    else:
        if modeltype =='Binary_Classification':
            lgb = lgbm.LGBMClassifier()
            objective = 'binary'
            metric = 'auc'
            is_unbalance = True
            class_weight = None
            score_name = 'ROC AUC'
            num_class = 1
        else:
            lgb = lgbm.LGBMClassifier()
            objective = 'multiclass'
            #objective = 'multiclassova'
            metric = 'multi_logloss'
            is_unbalance = True
            class_weight = 'balanced'
            score_name = 'Multiclass Logloss'
            if multi_label:
                if isinstance(y_train, np.ndarray):
                    num_class = np.unique(y_train).max() + 1
                else:
                    num_class = y_train.nunique().max() 
            else:
                if isinstance(y_train, np.ndarray):
                    num_class = np.unique(y_train).max() + 1
                else:
                    num_class = y_train.nunique()

    early_stopping_params={"early_stopping_rounds":10,
                "eval_metric" : metric, 
                "eval_set" : [[x_test, y_test]],
               }
    if modeltype == 'Regression':
        ## there is no num_class in regression for LGBM model ##
        lgbm_params = {'learning_rate': 0.001,
                       'objective': objective,
                       'metric': metric,
                       'boosting_type': 'gbdt',
                       'max_depth': 8,
                       'subsample': 0.2,
                       'colsample_bytree': 0.3,
                       'reg_alpha': 0.54,
                       'reg_lambda': 0.4,
                       'min_split_gain': 0.7,
                       'min_child_weight': 26,
                        'num_leaves': 32,
                       'save_binary': True,
                       'seed': 1337, 'feature_fraction_seed': 1337,
                       'bagging_seed': 1337, 'drop_seed': 1337, 
                       'data_random_seed': 1337,
                       'verbose': -1, 
                        'n_estimators': 400,
                    }
    else:
        lgbm_params = {'learning_rate': 0.001,
                       'objective': objective,
                       'metric': metric,
                       'boosting_type': 'gbdt',
                       'max_depth': 8,
                       'subsample': 0.2,
                       'colsample_bytree': 0.3,
                       'reg_alpha': 0.54,
                       'reg_lambda': 0.4,
                       'min_split_gain': 0.7,
                       'min_child_weight': 26,
                        'num_leaves': 32,
                       'save_binary': True,
                       'seed': 1337, 'feature_fraction_seed': 1337,
                       'bagging_seed': 1337, 'drop_seed': 1337, 
                       'data_random_seed': 1337,
                       'verbose': -1, 
                       'num_class': num_class,
                       'is_unbalance': is_unbalance,
                       'class_weight': class_weight,
                        'n_estimators': 400,
                }

    lgb.set_params(**lgbm_params)
    if multi_label:
        if modeltype == 'Regression':
            lgb = MultiOutputRegressor(lgb)
        else:
            lgb = MultiOutputClassifier(lgb)
    ########   Now let's perform randomized search to find best hyper parameters ######
    if random_search_flag:
        model = RandomizedSearchCV(lgb,
                   param_distributions = rand_params,
                   n_iter = 10,
                   return_train_score = True,
                   random_state = 99,
                   n_jobs=-1,
                   cv = 3,
                   verbose = False)
        ##### This is where we search for hyper params for model #######
        if multi_label:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, **early_stopping_params)
        print('Time taken for Hyper Param tuning of LGBM (in minutes) = %0.1f' %(
                                        (time.time()-start_time)/60))
        cv_results = pd.DataFrame(model.cv_results_)
        print('Mean cross-validated test %s = %0.04f' %(score_name, cv_results['mean_test_score'].mean()))
    else:
        try:
            model.fit(x_train, y_train,  verbose=-1)
        except:
            print('lightgbm model is crashing. Please check your inputs and try again...')
    return model
##############################################################################################
def complex_XGBoost_model(X_train, y_train, X_test, log_y=False, GPU_flag=False,
                                scaler = '', enc_method='label', n_splits=5, verbose=-1):
    """
    This model is called complex because it handle multi-label, mulit-class datasets which XGBoost ordinarily cant.
    Just send in X_train, y_train and what you want to predict, X_test
    It will automatically split X_train into multiple folds (10) and train and predict each time on X_test.
    It will then use average (or use mode) to combine the results and give you a y_test.
    It will automatically detect modeltype as "Regression" or 'Classification'
    It will also add MultiOutputClassifier and MultiOutputRegressor to multi_label problems.
    The underlying estimators in all cases is XGB. So you get the best of both worlds.

    Inputs:
    ------------
    X_train: pandas dataframe only: do not send in numpy arrays. This is the X_train of your dataset.
    y_train: pandas Series or DataFrame only: do not send in numpy arrays. This is the y_train of your dataset.
    X_test: pandas dataframe only: do not send in numpy arrays. This is the X_test of your dataset.
    log_y: default = False: If True, it means use the log of the target variable "y" to train and test.
    GPU_flag: if your machine has a GPU set this flag and it will use XGBoost GPU to speed up processing.
    scaler : default is StandardScaler(). But you can send in MinMaxScaler() as input to change it or any other scaler.
    enc_method: default is 'label' encoding. But you can choose 'glmm' as an alternative. But those are the only two.
    verbose: default = 0. Choosing 1 will give you lot more output.

    Outputs:
    ------------
    y_preds: Predicted values for your X_XGB_test dataframe.
        It has been averaged after repeatedly predicting on X_XGB_test. So likely to be better than one model.
    """
    X_XGB = copy.deepcopy(X_train)
    Y_XGB = copy.deepcopy(y_train)
    X_XGB_test = copy.deepcopy(X_test)
    ####################################
    start_time = time.time()
    top_num = 10
    if isinstance(Y_XGB, pd.Series):
        targets = [Y_XGB.name]
    else:
        targets = Y_XGB.columns.tolist()
    if len(targets) == 1:
        multi_label = False
        if isinstance(Y_XGB, pd.DataFrame):
            Y_XGB = pd.Series(Y_XGB.values.ravel(),name=targets[0], index=Y_XGB.index)
    else:
        multi_label = True
    modeltype = analyze_problem_type(Y_XGB, targets)
    columns =  X_XGB.columns
    ##### Now continue with scaler pre-processing ###########
    if isinstance(scaler, str):
        if not scaler == '':
            scaler = scaler.lower()
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    #########     G P U     P R O C E S S I N G      B E G I N S    ############
    ###### This is where we set the CPU and GPU parameters for XGBoost
    if GPU_flag:
        GPU_exists = check_if_GPU_exists()
    else:
        GPU_exists = False
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    cpu_params['tree_method'] = 'hist'
    cpu_params['gpu_id'] = 0
    cpu_params['updater'] = 'grow_colmaker'
    cpu_params['predictor'] = 'cpu_predictor'
    if GPU_exists:
        param['tree_method'] = 'gpu_hist'
        param['gpu_id'] = 0
        param['updater'] = 'grow_gpu_hist' #'prune'
        param['predictor'] = 'gpu_predictor'
        print('    Hyper Param Tuning XGBoost with GPU parameters. This will take time. Please be patient...')
    else:
        param = copy.deepcopy(cpu_params)
        print('    Hyper Param Tuning XGBoost with CPU parameters. This will take time. Please be patient...')
    #################################################################################
    if modeltype == 'Regression':
        if log_y:
            Y_XGB.loc[Y_XGB==0] = 1e-15  ### just set something that is zero to a very small number

    #########  Now set the number of rows we need to tune hyper params ###
    scoreFunction = { "precision": "precision_weighted","recall": "recall_weighted"}
    random_search_flag =  True

    if multi_label:
        #### We need a small validation data set for hyper-param tuning #########################
        hyper_frac = 0.2
        #### now select a random sample from X_XGB ##
        if modeltype == 'Regression':
            X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                                random_state=999)
        else:
            X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                                random_state=999, stratify = Y_XGB)
        
        #### First convert test data into numeric using train data ###
        X_train, Y_train, X_valid, Y_valid = data_transform(X_train, Y_train, X_valid, Y_valid,
                                    modeltype, multi_label, scaler=scaler, enc_method=enc_method)

        ######  This step is needed for making sure y is transformed to log_y ####################
        if modeltype == 'Regression' and log_y:
                Y_train = np.log(Y_train)

        ######  Time to hyper-param tune model using randomizedsearchcv and partial train data #########
        num_boost_round = xgbm_model_fit(random_search_flag, X_train, Y_train, X_valid, Y_valid, modeltype,
                             multi_label, log_y, num_boost_round=100)

    else:
        ######  This step is needed for making sure y is transformed to log_y ######
        if modeltype == 'Regression' and log_y:
            ######  Time to hyper-param tune model using randomizedsearchcv and full train data #########
            num_boost_round = xgbm_model_fit(random_search_flag, X_XGB, np.log(Y_XGB), "", "", modeltype,
                                 multi_label, log_y, num_boost_round=100)
        else:
            ######  Time to hyper-param tune model using randomizedsearchcv and full train data #########
            num_boost_round = xgbm_model_fit(random_search_flag, X_XGB, Y_XGB, "", "", modeltype,
                                 multi_label, log_y, num_boost_round=100)

    #### First convert test data into numeric using train data ###############################
    if not isinstance(X_XGB_test, str):
        x_train, y_train, x_test, _ = data_transform(X_XGB, Y_XGB, X_XGB_test, "",
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

    ######  Time to train the hyper-tuned model on full train data ##########################
    random_search_flag = False
    model = xgbm_model_fit(random_search_flag, x_train, y_train, x_test, "", modeltype,
                                multi_label, log_y, num_boost_round=num_boost_round)
    
    #############  Time to get feature importances based on full train data   ################
    if multi_label:
        for i,target_name in enumerate(targets):
            each_model = model.estimators_[i]
            imp_feats = dict(zip(x_train.columns, each_model.feature_importances_))
            importances = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].values
            important_features = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
            print('Top 10 features for {}: {}'.format(target_name, important_features))
    else: 
        imp_feats = model.get_score(fmap='', importance_type='gain')
        importances = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].values
        important_features = pd.Series(imp_feats).sort_values(ascending=False)[:top_num].index.tolist()
        print('Top 10 features:\n%s' %important_features[:top_num])
        #######  order this in the same order in which they were collected ######
        feature_importances = pd.DataFrame(importances,
                                           index = important_features,
                                            columns=['importance'])
    
    ######  Time to consolidate the predictions on test data ################################
    if not multi_label and not isinstance(X_XGB_test, str):
        x_test = xgb.DMatrix(x_test)
    if isinstance(X_XGB_test, str):
        print('No predictions since X_XGB_test is empty string. Returning...')
        return {}
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            if log_y:
                pred_xgbs = np.exp(model.predict(x_test))
            else:
                pred_xgbs = model.predict(x_test)
            #### if there is no test data just return empty strings ###
        else:
            pred_xgbs = []
    else:
        if multi_label:
            pred_xgbs = model.predict(x_test)
            pred_probas = model.predict_proba(x_test)
        else:
            pred_probas = model.predict(x_test)
            if modeltype =='Multi_Classification':
                pred_xgbs = pred_probas.argmax(axis=1)
            else:
                pred_xgbs = (pred_probas>0.5).astype(int)
    ##### once the entire model is trained on full train data ##################
    print('    Time taken for training XGBoost on entire train data (in minutes) = %0.1f' %(
             (time.time()-start_time)/60))
    if multi_label:
        for i,target_name in enumerate(targets):
            each_model = model.estimators_[i]
            xgb.plot_importance(each_model, title='XGBoost model feature importances for %s' %target_name)
    else:
        xgb.plot_importance(model, title='XGBoost final model feature importances')
    print('Returning the following:')
    print('    Model = %s' %model)
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            print('    final predictions', pred_xgbs[:10])
        return (pred_xgbs, model)
    else:
        if not isinstance(X_XGB_test, str):
            print('    final predictions (may need to be transformed to original labels)', pred_xgbs[:10])
            print('    predicted probabilities', pred_probas[:1])
        return (pred_xgbs, pred_probas, model)
##############################################################################################
import xgboost as xgb
def xgbm_model_fit(random_search_flag, x_train, y_train, x_test, y_test, modeltype,
                         multi_label, log_y, num_boost_round=100):
    start_time = time.time()
    if multi_label:
        rand_params = {
                }
    else:
        rand_params = {
            'learning_rate': sp.stats.uniform(scale=1),
            'gamma': sp.stats.randint(0, 100),
            'n_estimators': sp.stats.randint(100,500),
            "max_depth": sp.stats.randint(3, 15),
                }

    if modeltype == 'Regression':
        objective = 'reg:squarederror' 
        eval_metric = 'rmse'
        shuffle = False
        stratified = False
        num_class = 0
        score_name = 'Score'
    else:
        if modeltype =='Binary_Classification':
            objective='binary:logistic'
            eval_metric = 'error' ## dont foolishly change to auc or aucpr since it doesnt work in finding feature imps later
            shuffle = True
            stratified = True
            num_class = 1
            score_name = 'Error Rate'
        else:
            objective = 'multi:softprob'
            eval_metric = 'merror'  ## dont foolishly change to auc or aucpr since it doesnt work in finding feature imps later
            shuffle = True
            stratified = True
            if multi_label:
                num_class = y_train.nunique().max() 
            else:
                if isinstance(y_train, np.ndarray):
                    num_class = np.unique(y_train).max() + 1
                elif isinstance(y_train, pd.Series):
                    num_class = y_train.nunique()
                else:
                    num_class = y_train.nunique().max() 
            score_name = 'Multiclass Error Rate'

    
    final_params = {
          'booster' :'gbtree',
          'colsample_bytree': 0.5,
          'alpha': 0.015,
          'gamma': 4,
          'learning_rate': 0.01,
          'max_depth': 8,
          'min_child_weight': 2,
          'reg_lambda': 0.5,
          'subsample': 0.7,
          'random_state': 99,
          'objective': objective,
          'eval_metric': eval_metric,
          'verbosity': 0,
          'n_jobs': -1,
          #grow_policy='lossguide',
          'num_class': num_class,
          'silent': True
            }
    #######  This is where we split into single and multi label ############
    if multi_label:
        ######   This is for Multi_Label problems ############
        if modeltype == 'Regression':
            clf = XGBRegressor(n_jobs=-1, random_state=999, max_depth=6)
            clf.set_params(**final_params)
            model = MultiOutputRegressor(clf)
        else:
            clf = XGBClassifier(n_jobs=-1, random_state=999, max_depth=6)
            clf.set_params(**final_params)
            model = MultiOutputClassifier(clf)
        if random_search_flag:
            model = RandomizedSearchCV(model,
                       param_distributions = rand_params,
                       n_iter = 10,
                       return_train_score = True,
                       random_state = 99,
                       n_jobs=-1,
                       cv = 3,
                       verbose = False)        
            model.fit(x_train, y_train)
            print('Time taken for Hyper Param tuning of multi_label XGBoost (in minutes) = %0.1f' %(
                                            (time.time()-start_time)/60))
            cv_results = pd.DataFrame(model.cv_results_)
            print('Mean cross-validated test %s = %0.04f' %(score_name, cv_results['mean_test_score'].mean()))
            ### In this case, there is no boost rounds so just return the default num_boost_round
            return num_boost_round
        else:
            try:
                model.fit(x_train, y_train)
            except:
                print('Multi_label XGBoost model is crashing during training. Please check your inputs and try again...')
            return model
    else:
        
        #### This is for Single Label Problems #############
        dtrain = xgb.DMatrix(x_train, label=y_train)
        ########   Now let's perform randomized search to find best hyper parameters ######
        if random_search_flag:
            cv_results = xgb.cv(final_params, dtrain, num_boost_round=400, nfold=5, 
                stratified=stratified, metrics=eval_metric, early_stopping_rounds=10, seed=999, shuffle=shuffle)
            # Update best eval_metric
            best_eval = 'test-'+eval_metric+'-mean'
            mean_mae = cv_results[best_eval].min()
            boost_rounds = cv_results[best_eval].argmin()
            print("Cross-validated %s = %0.3f in num rounds = %s" %(score_name, mean_mae, boost_rounds))
            print('Time taken for Hyper Param tuning of XGBoost (in minutes) = %0.1f' %(
                                                (time.time()-start_time)/60))
            return boost_rounds
        else:
            try:
                model = xgb.train(
                    final_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    verbose_eval=False,
                )
            except:
                print('XGBoost model is crashing. Please check your inputs and try again...')
            return model
####################################################################################
def xgboost_model_fit(model, x_train, y_train, x_test, y_test, modeltype, log_y, params, 
                    cpu_params, early_stopping_params={}):
    early_stopping = 10
    start_time = time.time()
    if str(model).split("(")[0] == 'RandomizedSearchCV':

        model.fit(x_train, y_train, **early_stopping_params)
        print('Time taken for Hyper Param tuning of XGB (in minutes) = %0.1f' %(
                                        (time.time()-start_time)/60))
    else:
        try:
            if modeltype == 'Regression':
                if log_y:
                    model.fit(x_train, np.log(y_train), early_stopping_rounds=early_stopping, eval_metric=['rmse'],
                            eval_set=[(x_test, np.log(y_test))], verbose=0)
                else:
                    model.fit(x_train, y_train, early_stopping_rounds=early_stopping, eval_metric=['rmse'],
                            eval_set=[(x_test, y_test)], verbose=0)
            else:
                if modeltype == 'Binary_Classification':
                    objective='binary:logistic'
                    eval_metric = 'error'
                else:
                    objective='multi:softprob'
                    eval_metric = 'merror'
                model.fit(x_train, y_train, early_stopping_rounds=early_stopping, eval_metric = eval_metric,
                                eval_set=[(x_test, y_test)], verbose=0)
        except:
            print('GPU is present but not turned on. Please restart after that. Currently using CPU...')
            if str(model).split("(")[0] == 'RandomizedSearchCV':
                xgb = model.estimator_
                xgb.set_params(**cpu_params)
                model = RandomizedSearchCV(xgb,
                                                   param_distributions = params,
                                                   n_iter = 10,
                                                   n_jobs=-1,
                                                   cv=5)
                model.fit(x_train, y_train, **early_stopping_params)
                return model
            else:
                model = model.set_params(**cpu_params)
            if modeltype == 'Regression':
                if log_y:
                    model.fit(x_train, np.log(y_train), early_stopping_rounds=6, eval_metric=['rmse'],
                            eval_set=[(x_test, np.log(y_test))], verbose=0)
                else:
                    model.fit(x_train, y_train, early_stopping_rounds=6, eval_metric=['rmse'],
                            eval_set=[(x_test, y_test)], verbose=0)
            else:
                model.fit(x_train, y_train, early_stopping_rounds=6,eval_metric=eval_metric,
                                eval_set=[(x_test, y_test)], verbose=0)
    return model
#################################################################################
def simple_XGBoost_model(X_train, y_train, X_test, log_y=False, GPU_flag=False,
                                scaler = '', enc_method='label', n_splits=5, verbose=0):
    """
    Easy to use XGBoost model. Just send in X_train, y_train and what you want to predict, X_test
    It will automatically split X_train into multiple folds (10) and train and predict each time on X_test.
    It will then use average (or use mode) to combine the results and give you a y_test.
    You just need to give the modeltype as "Regression" or 'Classification'

    Inputs:
    ------------
    X_train: pandas dataframe only: do not send in numpy arrays. This is the X_train of your dataset.
    y_train: pandas Series or DataFrame only: do not send in numpy arrays. This is the y_train of your dataset.
    X_test: pandas dataframe only: do not send in numpy arrays. This is the X_test of your dataset.
    modeltype: can only be 'Regression' or 'Classification'
    log_y: default = False: If True, it means use the log of the target variable "y" to train and test.
    GPU_flag: if your machine has a GPU set this flag and it will use XGBoost GPU to speed up processing.
    scaler : default is StandardScaler(). But you can send in MinMaxScaler() as input to change it or any other scaler.
    enc_method: default is 'label' encoding. But you can choose 'glmm' as an alternative. But those are the only two.
    verbose: default = 0. Choosing 1 will give you lot more output.

    Outputs:
    ------------
    y_preds: Predicted values for your X_XGB_test dataframe.
        It has been averaged after repeatedly predicting on X_XGB_test. So likely to be better than one model.
    """
    X_XGB = copy.deepcopy(X_train)
    Y_XGB = copy.deepcopy(y_train)
    X_XGB_test = copy.deepcopy(X_test)
    start_time = time.time()
    if isinstance(Y_XGB, pd.Series):
        targets = [Y_XGB.name]
    else:
        targets = Y_XGB.columns.tolist()
    Y_XGB_index = Y_XGB.index
    if len(targets) == 1:
        multi_label = False
        if isinstance(Y_XGB, pd.DataFrame):
            Y_XGB = pd.Series(Y_XGB.values.ravel(),name=targets[0], index=Y_XGB.index)
    else:
        multi_label = True
        print('Multi_label is not supported in simple_XGBoost_model. Try the complex_XGBoost_model...Returning')
        return {}
    ##### Start your analysis of the data ############
    modeltype = analyze_problem_type(Y_XGB, targets)
    columns =  X_XGB.columns
    ##### Now continue with scaler pre-processing ###########
    if isinstance(scaler, str):
        if not scaler == '':
            scaler = scaler.lower()
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    #########     G P U     P R O C E S S I N G      B E G I N S    ############
    ###### This is where we set the CPU and GPU parameters for XGBoost
    if GPU_flag:
        GPU_exists = check_if_GPU_exists()
    else:
        GPU_exists = False
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    cpu_params['tree_method'] = 'hist'
    cpu_params['gpu_id'] = 0
    cpu_params['updater'] = 'grow_colmaker'
    cpu_params['predictor'] = 'cpu_predictor'
    if GPU_exists:
        param['tree_method'] = 'gpu_hist'
        param['gpu_id'] = 0
        param['updater'] = 'grow_gpu_hist' #'prune'
        param['predictor'] = 'gpu_predictor'
        print('    Hyper Param Tuning XGBoost with GPU parameters. This will take time. Please be patient...')
    else:
        param = copy.deepcopy(cpu_params)
        print('    Hyper Param Tuning XGBoost with CPU parameters. This will take time. Please be patient...')
    #################################################################################
    if modeltype == 'Regression':
        if log_y:
            Y_XGB.loc[Y_XGB==0] = 1e-15  ### just set something that is zero to a very small number
        xgb = XGBRegressor(
                          booster = 'gbtree',
                          colsample_bytree=0.5,
                          alpha=0.015,
                          gamma=4,
                          learning_rate=0.01,
                          max_depth=8,
                          min_child_weight=2,
                          n_estimators=1000,
                          reg_lambda=0.5,
          	              #reg_alpha=8,
                          subsample=0.7,
                          random_state=99,
                          objective='reg:squarederror',
          	              eval_metric='rmse',
                          verbosity = 0,
                          n_jobs=-1,
                          #grow_policy='lossguide',
                          silent = True)
        objective='reg:squarederror'
        eval_metric = 'rmse'
        score_name = 'RMSE'
    else:
        if multi_label:
            num_class = Y_XGB.nunique().max() 
        else:
            if isinstance(Y_XGB, np.ndarray):
                num_class = np.unique(Y_XGB).max() + 1
            else:
                num_class = Y_XGB.nunique()
        if num_class == 2:
            num_class = 1
        if num_class <= 2:
            objective='binary:logistic'
            eval_metric = 'error'
            score_name = 'Error Rate'
        else:
            objective='multi:softprob'
            eval_metric = 'merror'
            score_name = 'Multiclass Error Rate'
        xgb = XGBClassifier(
                         booster = 'gbtree',
                         colsample_bytree=0.5,
                         alpha=0.015,
                         gamma=4,
                         learning_rate=0.01,
                         max_depth=8,
                         min_child_weight=2,
                         n_estimators=1000,
                         reg_lambda=0.5,
                         objective=objective,
                         subsample=0.7,
                         random_state=99,
                         n_jobs=-1,
                         #grow_policy='lossguide',
                         num_class = num_class,
                         verbosity = 0,
                         silent = True)

    #testing for GPU
    model = xgb.set_params(**param)
    hyper_frac = 0.2
    #### now select a random sample from X_XGB and Y_XGB ################
    if modeltype == 'Regression':
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac,
                                        random_state=99)
    else:
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac,
                                        random_state=99, stratify=Y_XGB)

    scoreFunction = { "precision": "precision_weighted","recall": "recall_weighted"}
    params = {
        'learning_rate': sp.stats.uniform(scale=1),
        'gamma': sp.stats.randint(0, 32),
       'n_estimators': sp.stats.randint(100,500),
        "max_depth": sp.stats.randint(3, 15),
            }
    early_stopping_params={"early_stopping_rounds":5,
                "eval_metric" : eval_metric, 
                "eval_set" : [[X_valid, Y_valid]]
               }

    model = RandomizedSearchCV(xgb.set_params(**param),
                                       param_distributions = params,
                                       n_iter = 10,
                                       return_train_score = True,
                                       random_state = 99,
                                       n_jobs=-1,
                                       #cv = 3,
                                        verbose = False)

    X_train, Y_train, X_valid, Y_valid = data_transform(X_train, Y_train, X_valid, Y_valid,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)
    
    gbm_model = xgboost_model_fit(model, X_train, Y_train, X_valid, Y_valid, modeltype,
                         log_y, params, cpu_params, early_stopping_params)
    #############################################################################
    ls=[]
    if modeltype == 'Regression':
        fold = KFold(n_splits=n_splits)
    else:
        fold = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=99)
    scores=[]
    if not isinstance(X_XGB_test, str):
        pred_xgbs = np.zeros(len(X_XGB_test))
        pred_probas = np.zeros(len(X_XGB_test))
    else:
        pred_xgbs = []
        pred_probas = []
    #### First convert test data into numeric using train data ###
    if not isinstance(X_XGB_test, str):
        X_XGB_train_enc, Y_XGB, X_XGB_test_enc, _ = data_transform(X_XGB, Y_XGB, X_XGB_test,"",
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)
    else:
        X_XGB_train_enc, Y_XGB, X_XGB_test_enc, _ = data_transform(X_XGB, Y_XGB, "","",
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)
    #### now run all the folds each one by one ##################################
    start_time = time.time()
    for folds, (train_index, test_index) in tqdm(enumerate(fold.split(X_XGB,Y_XGB))):
        
        x_train, x_valid = X_XGB.iloc[train_index], X_XGB.iloc[test_index]
        ### you need to keep y_valid as-is in the same original state as it was given ####
        if isinstance(Y_XGB, np.ndarray):
            Y_XGB = pd.Series(Y_XGB,name=targets[0], index=Y_XGB_index)
        ### y_valid here will be transformed into log_y to ensure training and validation ####
        
        if modeltype == 'Regression':
            if log_y:
                y_train, y_valid = np.log(Y_XGB.iloc[train_index]), np.log(Y_XGB.iloc[test_index])
            else:
                y_train, y_valid = Y_XGB.iloc[train_index], Y_XGB.iloc[test_index]
        else:
            
            y_train, y_valid = Y_XGB.iloc[train_index], Y_XGB.iloc[test_index]

        ##  scale the x_train and x_valid values - use all columns -
        x_train, y_train, x_valid, y_valid = data_transform(x_train, y_train, x_valid, y_valid,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

        model = gbm_model.best_estimator_
        model = xgboost_model_fit(model, x_train, y_train, x_valid, y_valid, modeltype,
                                log_y, params, cpu_params)

        #### now make predictions on validation data and compare it to y_valid which is in original state ##
        if modeltype == 'Regression':
            if log_y:
                preds = np.exp(model.predict(x_valid))
            else:
                preds = model.predict(x_valid)
        else:
            preds = model.predict(x_valid)

        feature_importances = pd.DataFrame(model.feature_importances_,
                                           index = X_XGB.columns,
                                            columns=['importance'])
        sum_all=feature_importances.values
        ls.append(sum_all)
        ######  Time to consolidate the predictions on test data #########
        if modeltype == 'Regression':
            if not isinstance(X_XGB_test, str):
                if log_y:
                    pred_xgb=np.exp(model.predict(X_XGB_test_enc[columns]))
                else:
                    pred_xgb=model.predict(X_XGB_test_enc[columns])
                pred_xgbs = np.vstack([pred_xgbs, pred_xgb])
                pred_xgbs = pred_xgbs.mean(axis=0)
            #### preds here is for only one fold and we are comparing it to original y_valid ####
            score = np.sqrt(mean_squared_error(y_valid, preds))
            print('%s score in fold %d = %s' %(score_name, folds+1, score))
        else:
            if not isinstance(X_XGB_test, str):
                pred_xgb=model.predict(X_XGB_test_enc[columns])
                pred_proba = model.predict_proba(X_XGB_test_enc[columns])
                if folds == 0:
                    pred_xgbs = copy.deepcopy(pred_xgb)
                    pred_probas = copy.deepcopy(pred_proba)
                else:
                    pred_xgbs = np.vstack([pred_xgbs, pred_xgb])
                    pred_xgbs = stats.mode(pred_xgbs, axis=0)[0][0]
                    pred_probas = np.mean( np.array([ pred_probas, pred_proba ]), axis=0 )
            #### preds here is for only one fold and we are comparing it to original y_valid ####
            score = balanced_accuracy_score(y_valid, preds)
            print('%s score in fold %d = %0.1f%%' %(score_name, folds+1, score*100))
        scores.append(score)
    print('    Time taken for Cross Validation of XGBoost (in minutes) = %0.1f' %(
             (time.time()-start_time)/60))
    print("\nCross-validated Average scores are: ", np.sum(scores)/len(scores))
    ##### Train on full train data set and predict #################################
    print('Training model on full train dataset...')
    start_time1 = time.time()
    model = gbm_model.best_estimator_
    model.fit(X_XGB_train_enc, Y_XGB)
    print('    Time taken for training XGBoost (in minutes) = %0.1f' %(
             (time.time()-start_time1)/60))
    if verbose:
        plot_importances_XGB(train_set=X_XGB, labels=Y_XGB, ls=ls, y_preds=pred_xgbs,
                            modeltype=modeltype, top_num='all')
    print('Returning the following:')
    print('    Model = %s' %model)
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            print('    final predictions', pred_xgbs[:10])
        return (pred_xgbs, model)
    else:
        if not isinstance(X_XGB_test, str):
            print('    final predictions (may need to be transformed to original labels)', pred_xgbs[:10])
            print('    predicted probabilities', pred_probas[:1])
        return (pred_xgbs, pred_probas, model)
##################################################################################
def complex_LightGBM_model(X_train, y_train, X_test, log_y=False, GPU_flag=False,
                scaler = '', enc_method='label', n_splits=5, verbose=-1):
    """
    This model is called complex because it handle multi-label, mulit-class datasets which LGBM ordinarily cant.
    Just send in X_train, y_train and what you want to predict, X_test
    It will automatically split X_train into multiple folds (10) and train and predict each time on X_test.
    It will then use average (or use mode) to combine the results and give you a y_test.
    It will automatically detect modeltype as "Regression" or 'Classification'
    It will also add MultiOutputClassifier and MultiOutputRegressor to multi_label problems.
    The underlying estimators in all cases is LGBM. So you get the best of both worlds.

    Inputs:
    ------------
    X_train: pandas dataframe only: do not send in numpy arrays. This is the X_train of your dataset.
    y_train: pandas Series or DataFrame only: do not send in numpy arrays. This is the y_train of your dataset.
    X_test: pandas dataframe only: do not send in numpy arrays. This is the X_test of your dataset.
    log_y: default = False: If True, it means use the log of the target variable "y" to train and test.
    GPU_flag: if your machine has a GPU set this flag and it will use XGBoost GPU to speed up processing.
    scaler : default is StandardScaler(). But you can send in MinMaxScaler() as input to change it or any other scaler.
    enc_method: default is 'label' encoding. But you can choose 'glmm' as an alternative. But those are the only two.
    verbose: default = 0. Choosing 1 will give you lot more output.

    Outputs:
    ------------
    y_preds: Predicted values for your X_XGB_test dataframe.
        It has been averaged after repeatedly predicting on X_XGB_test. So likely to be better than one model.
    """
    X_XGB = copy.deepcopy(X_train)
    Y_XGB = copy.deepcopy(y_train)
    X_XGB_test = copy.deepcopy(X_test)
    ####################################
    start_time = time.time()
    if isinstance(Y_XGB, pd.Series):
        targets = [Y_XGB.name]
    else:
        targets = Y_XGB.columns.tolist()
    if len(targets) == 1:
        multi_label = False
        if isinstance(Y_XGB, pd.DataFrame):
            Y_XGB = pd.Series(Y_XGB.values.ravel(),name=targets[0], index=Y_XGB.index)
    else:
        multi_label = True
    modeltype = analyze_problem_type(Y_XGB, targets)
    columns =  X_XGB.columns
    #### In some cases, there are special chars in column names. Remove them. ###
    if np.array([':' in x for x in columns]).any():
        sel_preds = columns[np.array([':' in x for x in columns])].tolist()
        print('removing special char : in %s since LightGBM does not like it...' %sel_preds)
        columns = ["_".join(x.split(":")) for x in columns]
        X_XGB.columns = columns
        if not isinstance(X_XGB_test, str):
            X_XGB_test.columns = columns
    ##### Now continue with scaler pre-processing ###########
    if isinstance(scaler, str):
        if not scaler == '':
            scaler = scaler.lower()
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    #########     G P U     P R O C E S S I N G      B E G I N S    ############
    ###### This is where we set the CPU and GPU parameters for XGBoost
    if GPU_flag:
        GPU_exists = check_if_GPU_exists()
    else:
        GPU_exists = False
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    cpu_params['tree_method'] = 'hist'
    cpu_params['gpu_id'] = 0
    cpu_params['updater'] = 'grow_colmaker'
    cpu_params['predictor'] = 'cpu_predictor'
    if GPU_exists:
        param['tree_method'] = 'gpu_hist'
        param['gpu_id'] = 0
        param['updater'] = 'grow_gpu_hist' #'prune'
        param['predictor'] = 'gpu_predictor'
        print('    Hyper Param Tuning LightGBM with GPU parameters. This will take time. Please be patient...')
    else:
        param = copy.deepcopy(cpu_params)
        print('    Hyper Param Tuning LightGBM with CPU parameters. This will take time. Please be patient...')
    #################################################################################
    if modeltype == 'Regression':
        if log_y:
            Y_XGB.loc[Y_XGB==0] = 1e-15  ### just set something that is zero to a very small number

    #########  Now set the number of rows we need to tune hyper params ###
    scoreFunction = { "precision": "precision_weighted","recall": "recall_weighted"}

    #### We need a small validation data set for hyper-param tuning #############
    hyper_frac = 0.2
    #### now select a random sample from X_XGB ##
    if modeltype == 'Regression':
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                            random_state=999)
    else:
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                            random_state=999, stratify = Y_XGB)

    #### First convert test data into numeric using train data ###
    X_train, Y_train, X_valid, Y_valid = data_transform(X_train, Y_train, X_valid, Y_valid,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)
    ######  This step is needed for making sure y is transformed to log_y ######
    if modeltype == 'Regression' and log_y:
            Y_train = np.log(Y_train)
    random_search_flag =  True

    ######  Time to hyper-param tune model using randomizedsearchcv #########
    gbm_model = lightgbm_model_fit(random_search_flag, X_train, Y_train, X_valid, Y_valid, modeltype,
                         multi_label, log_y, model="")
    model = gbm_model.best_estimator_

    #### First convert test data into numeric using train data ###
    if not isinstance(X_XGB_test, str):
        x_train, y_train, x_test, _ = data_transform(X_XGB, Y_XGB, X_XGB_test, "",
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

    ######  Time to train the hyper-tuned model on full train data #########
    random_search_flag = False
    model = lightgbm_model_fit(random_search_flag, x_train, y_train, x_test, "", modeltype,
                            multi_label, log_y, model=model)
    #############  Time to get feature importances based on full train data   ################
    if multi_label:
        for i,target_name in enumerate(targets):
            print('Top 10 features for {}: {}'.format(target_name,pd.DataFrame(model.estimators_[i].feature_importances_,
                index=model.estimators_[i].feature_name_,
                columns=['importance']).sort_values('importance', ascending=False).index.tolist()[:10]))
    else: 
        print('Top 10 features:\n', pd.DataFrame(model.feature_importances_,index=model.feature_name_,
            columns=['importance']).sort_values('importance', ascending=False).index.tolist()[:10])
    ######  Time to consolidate the predictions on test data #########
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            if log_y:
                pred_xgbs = np.exp(model.predict(x_test))
            else:
                pred_xgbs = model.predict(x_test)
            #### if there is no test data just return empty strings ###
        else:
            pred_xgbs = []
    else:
        if not isinstance(X_XGB_test, str):
            if not multi_label:
                pred_xgbs = model.predict(x_test)
                pred_probas = model.predict_proba(x_test)
            else:
                ### This is how you have to process if it is multi_label ##
                pred_probas = model.predict_proba(x_test)
                predsy = [np.argmax(line,axis=1) for line in pred_probas]
                pred_xgbs = np.array(predsy)
        else:
            pred_xgbs = []
            pred_probas = []
    ##### once the entire model is trained on full train data ##################
    print('    Time taken for training Light GBM on entire train data (in minutes) = %0.1f' %(
             (time.time()-start_time)/60))
    if multi_label:
        for i,target_name in enumerate(targets):
            lgbm.plot_importance(model.estimators_[i], title='LGBM model feature importances for %s' %target_name)
    else:
        lgbm.plot_importance(model, title='LGBM final model feature importances')
    print('Returning the following:')
    print('    Model = %s' %model)
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            print('    final predictions', pred_xgbs[:10])
        return (pred_xgbs, model)
    else:
        if not isinstance(X_XGB_test, str):
            print('    final predictions (may need to be transformed to original labels)', pred_xgbs[:10])
            print('    predicted probabilities', pred_probas[:1])
        return (pred_xgbs, pred_probas, model)
##################################################################################
def simple_LightGBM_model(X_train, y_train, X_test, log_y=False, GPU_flag=False,
                                scaler = '', enc_method='label', n_splits=5, verbose=-1):
    """
    Easy to use XGBoost model. Just send in X_train, y_train and what you want to predict, X_test
    It will automatically split X_train into multiple folds (10) and train and predict each time on X_test.
    It will then use average (or use mode) to combine the results and give you a y_test.
    You just need to give the modeltype as "Regression" or 'Classification'

    Inputs:
    ------------
    X_train: pandas dataframe only: do not send in numpy arrays. This is the X_train of your dataset.
    y_train: pandas Series or DataFrame only: do not send in numpy arrays. This is the y_train of your dataset.
    X_test: pandas dataframe only: do not send in numpy arrays. This is the X_test of your dataset.
    modeltype: can only be 'Regression' or 'Classification'
    log_y: default = False: If True, it means use the log of the target variable "y" to train and test.
    GPU_flag: if your machine has a GPU set this flag and it will use XGBoost GPU to speed up processing.
    scaler : default is StandardScaler(). But you can send in MinMaxScaler() as input to change it or any other scaler.
    enc_method: default is 'label' encoding. But you can choose 'glmm' as an alternative. But those are the only two.
    verbose: default = 0. Choosing 1 will give you lot more output.

    Outputs:
    ------------
    y_preds: Predicted values for your X_XGB_test dataframe.
        It has been averaged after repeatedly predicting on X_XGB_test. So likely to be better than one model.
    """
    X_XGB = copy.deepcopy(X_train)
    Y_XGB = copy.deepcopy(y_train)
    X_XGB_test = copy.deepcopy(X_test)
    #######################################
    start_time = time.time()
    if isinstance(Y_XGB, pd.Series):
        targets = [Y_XGB.name]
    else:
        targets = Y_XGB.columns.tolist()
    if len(targets) == 1:
        multi_label = False
        if isinstance(Y_XGB, pd.DataFrame):
            Y_XGB = pd.Series(Y_XGB.values.ravel(),name=targets[0], index=Y_XGB.index)
    else:
        multi_label = True
        print('Multi_label is not supported in simple_LightGBM_model. Try the complex_LightGBM_model...Returning')
        return {}
    ##### Start your analysis of the data ############
    modeltype = analyze_problem_type(Y_XGB, targets)
    columns =  X_XGB.columns
    #### In some cases, there are special chars in column names. Remove them. ###
    if np.array([':' in x for x in columns]).any():
        sel_preds = columns[np.array([':' in x for x in columns])].tolist()
        print('removing special char : in %s since LightGBM does not like it...' %sel_preds)
        columns = ["_".join(x.split(":")) for x in columns]
        X_XGB.columns = columns
        if not isinstance(X_XGB_test, str):
            X_XGB_test.columns = columns
    ##### Now continue with scaler pre-processing ###########
    if isinstance(scaler, str):
        if not scaler == '':
            scaler = scaler.lower()
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    #########     G P U     P R O C E S S I N G      B E G I N S    ############
    ###### This is where we set the CPU and GPU parameters for XGBoost
    if GPU_flag:
        GPU_exists = check_if_GPU_exists()
    else:
        GPU_exists = False
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    cpu_params['tree_method'] = 'hist'
    cpu_params['gpu_id'] = 0
    cpu_params['updater'] = 'grow_colmaker'
    cpu_params['predictor'] = 'cpu_predictor'
    if GPU_exists:
        param['tree_method'] = 'gpu_hist'
        param['gpu_id'] = 0
        param['updater'] = 'grow_gpu_hist' #'prune'
        param['predictor'] = 'gpu_predictor'
        print('    Hyper Param Tuning LightGBM with GPU parameters. This will take time. Please be patient...')
    else:
        param = copy.deepcopy(cpu_params)
        print('    Hyper Param Tuning LightGBM with CPU parameters. This will take time. Please be patient...')
    #################################################################################
    if modeltype == 'Regression':
        if log_y:
            Y_XGB.loc[Y_XGB==0] = 1e-15  ### just set something that is zero to a very small number

    #testing for GPU
    hyper_frac = 0.2
    #### now select a random sample from X_XGB and Y_XGB ################
    if modeltype == 'Regression':
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac,
                                        random_state=99)
    else:
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac,
                                        random_state=99, stratify=Y_XGB)

    scoreFunction = { "precision": "precision_weighted","recall": "recall_weighted"}

    X_train, Y_train, X_valid, Y_valid = data_transform(X_train, Y_train, X_valid, Y_valid,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)
    if modeltype == 'Regression':
        if log_y:
            Y_train, Y_valid = np.log(Y_train), np.log(Y_valid)
    random_search_flag =  True
    gbm_model = lightgbm_model_fit(random_search_flag, X_train, Y_train, X_valid, Y_valid, modeltype,
                         multi_label, log_y, model="")
    model = gbm_model.best_estimator_
    random_search_flag = False
    #############################################################################
    ls=[]
    if modeltype == 'Regression':
        fold = KFold(n_splits=n_splits)
    else:
        fold = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=99)
    scores=[]
    if not isinstance(X_XGB_test, str):
        pred_xgbs = np.zeros(len(X_XGB_test))
        pred_probas = np.zeros(len(X_XGB_test))
    else:
        pred_xgbs = []
        pred_probas = []
    #### First convert test data into numeric using train data ###
    if not isinstance(X_XGB_test, str):
        _,_, X_XGB_test_enc,_ = data_transform(X_XGB, Y_XGB, X_XGB_test, "",
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)
    #### now run all the folds each one by one ##################################
    start_time = time.time()
    for folds, (train_index, test_index) in tqdm(enumerate(fold.split(X_XGB,Y_XGB))):
        x_train, x_test = X_XGB.iloc[train_index], X_XGB.iloc[test_index]
        ### you need to keep y_test as-is in the same original state as it was given ####
        y_test = Y_XGB.iloc[test_index]
        ### y_valid here will be transformed into log_y to ensure training and validation ####
        if modeltype == 'Regression':
            if log_y:
                y_train, y_valid = np.log(Y_XGB.iloc[train_index]), np.log(Y_XGB.iloc[test_index])
            else:
                y_train, y_valid = Y_XGB.iloc[train_index], Y_XGB.iloc[test_index]
        else:
            y_train, y_valid = Y_XGB.iloc[train_index], Y_XGB.iloc[test_index]

        ##  scale the x_train and x_test values - use all columns -
        x_train, y_train, x_test, _ = data_transform(x_train, y_train, x_test, y_test,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

        model = gbm_model.best_estimator_
        model = lightgbm_model_fit(random_search_flag, x_train, y_train, x_test, y_valid, modeltype,
                                multi_label, log_y, model=model)

        #### now make predictions on validation data and compare it to y_test which is in original state ##
        if modeltype == 'Regression':
            if log_y:
                preds = np.exp(model.predict(x_test))
            else:
                preds = model.predict(x_test)
        else:
            preds = model.predict(x_test)

        feature_importances = pd.DataFrame(model.feature_importances_,
                                           index = X_XGB.columns,
                                            columns=['importance'])
        sum_all=feature_importances.values
        ls.append(sum_all)
        ######  Time to consolidate the predictions on test data #########
        if modeltype == 'Regression':
            if not isinstance(X_XGB_test, str):
                if log_y:
                    pred_xgb=np.exp(model.predict(X_XGB_test_enc[columns]))
                else:
                    pred_xgb=model.predict(X_XGB_test_enc[columns])
                pred_xgbs = np.vstack([pred_xgbs, pred_xgb])
                pred_xgbs = pred_xgbs.mean(axis=0)
            #### preds here is for only one fold and we are comparing it to original y_test ####
            score = np.sqrt(mean_squared_error(y_test, preds))
            print('RMSE score in fold %d = %s' %(folds+1, score))
        else:
            if not isinstance(X_XGB_test, str):
                pred_xgb=model.predict(X_XGB_test_enc[columns])
                pred_proba = model.predict_proba(X_XGB_test_enc[columns])
                if folds == 0:
                    pred_xgbs = copy.deepcopy(pred_xgb)
                    pred_probas = copy.deepcopy(pred_proba)
                else:
                    pred_xgbs = np.vstack([pred_xgbs, pred_xgb])
                    pred_xgbs = stats.mode(pred_xgbs, axis=0)[0][0]
                    pred_probas = np.mean( np.array([ pred_probas, pred_proba ]), axis=0 )
            #### preds here is for only one fold and we are comparing it to original y_test ####
            score = balanced_accuracy_score(y_test, preds)
            print('AUC score in fold %d = %0.1f%%' %(folds+1, score*100))
        scores.append(score)
    print('    Time taken for training Light GBM (in minutes) = %0.1f' %(
             (time.time()-start_time)/60))
    if verbose:
        plot_importances_XGB(train_set=X_XGB, labels=Y_XGB, ls=ls, y_preds=pred_xgbs,
                            modeltype=modeltype, top_num='all')
    print("\nCross-validated average AUC scores are: ", np.sum(scores)/len(scores))
    print('Returning the following:')
    print('    Model = %s' %model)
    if modeltype == 'Regression':
        print('    final predictions', pred_xgbs[:10])
        return (pred_xgbs, model)
    else:
        print('    final predictions (may need to be transformed to original labels)', pred_xgbs[:10])
        print('    predicted probabilities', pred_probas[:1])
        return (pred_xgbs, pred_probas, model)
########################################################################################
def plot_importances_XGB(train_set, labels, ls, y_preds, modeltype, top_num='all'):
    add_items=0
    for item in ls:
        add_items +=item
    if isinstance(top_num, str):
        df_cv=pd.DataFrame(add_items/len(ls),index=train_set.columns,
            columns=["importance"]).sort_values('importance', ascending=False)
        df_cv=df_cv.reset_index()
        feat_imp = pd.Series(df_cv.importance.values, 
            index=df_cv.drop(["importance"], axis=1)).sort_values(axis='index',ascending=False)
    else:
        ## this limits the number of items to the top_num items
        df_cv=pd.DataFrame(add_items/len(ls),index=train_set.columns[:top_num],
            columns=["importance"]).sort_values('importance', ascending=False)
        df_cv=df_cv.reset_index()
        feat_imp = pd.Series(df_cv.importance.values, 
            index=df_cv.drop(["importance"], axis=1)).sort_values(axis='index',ascending=False)[:top_num]
    ##### Now plot the feature importances #################        
    feat_imp2=feat_imp[feat_imp>0.00005]
    imp_columns=[]
    for item in pd.DataFrame(feat_imp2).reset_index()["index"].tolist():
        fcols=re.sub("[(),]","",str(item))
        try:
            columns= int(re.sub("['']","",fcols))
            imp_columns.append(columns)
        except:
            columns= re.sub("['']","",fcols)
            imp_columns.append(columns)
    # X_UPDATED=X_GB[imp_columns]
    len(imp_columns)
    fig = plt.figure(figsize=(15,8))
    ax1=plt.subplot(2, 2, 1)
    feat_imp.nlargest(5).plot(kind='barh', ax=ax1, title='Feature importances of model on test data')
    if modeltype == 'Regression':
        ax2=plt.subplot(2, 2, 2)
        pd.Series(y_preds).plot(ax=ax2, color='b', title='Model predictions on test data');
    else:
        ax2=plt.subplot(2, 2, 2)
        pd.Series(y_preds).hist(ax=ax2, color='b', label='Model predictions histogram on test data');
##################################################################################