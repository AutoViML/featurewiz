import numpy as np
np.random.seed(99)
import random
random.seed(42)
import pandas as pd
################################################################################
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
#################################################################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
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
from pathlib import Path

#sklearn data_preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#sklearn categorical encoding
import category_encoders as ce
from .my_encoders import My_LabelEncoder
from .stacking_models import get_class_distribution

#sklearn modelling
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

# boosting library
import xgboost as xgb
import matplotlib.pyplot as plt

import copy
#################################################################################
#### Regression or Classification type problem
#####################################################################################
from sklearn.impute import SimpleImputer
def data_transform(X_train, Y_train, X_test="", Y_test="", modeltype='Classification',
            multi_label=False, enc_method='label', scaler = StandardScaler()):
    ##### Use My_Label_Encoder to transform label targets if needed #####
    if multi_label:
        if modeltype != 'Regression':
            targets = Y_train.columns
            Y_train_encoded = copy.deepcopy(Y_train)
            for each_target in targets:
                if Y_train[each_target].dtype not in ['int64', 'int32','int16','int8', 'float16','float32','float64','float']:
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
            Y_train_encoded = copy.deepcopy(Y_train)
            Y_test_encoded = copy.deepcopy(Y_test)
    else:
        if modeltype != 'Regression':
            if Y_train.dtype not in ['int64', 'int32','int16','int8', 'float16','float32','float64','float']:
                mlb = My_LabelEncoder()
                Y_train_encoded= mlb.fit_transform(Y_train)
                if not isinstance(Y_test, str):
                    Y_test_encoded= mlb.transform(Y_test)
                else:
                    Y_test_encoded = copy.deepcopy(Y_test)
            else:
                Y_train_encoded = copy.deepcopy(Y_train)
                Y_test_encoded = copy.deepcopy(Y_test)
        else:
            Y_train_encoded = copy.deepcopy(Y_train)
            Y_test_encoded = copy.deepcopy(Y_test)
    
    
    #### This is where we find datetime vars and convert them to strings ####
    datetime_feats = X_train.select_dtypes(include='datetime').columns.tolist()
    ### if there are datetime values, convert them into features here ###
    from .featurewiz import FE_create_time_series_features
    for date_col in datetime_feats:
        fillnum = X_train[date_col].mode()[0]
        X_train[date_col].fillna(fillnum,inplace=True)
        X_train, ts_adds = FE_create_time_series_features(X_train, date_col)
        if not isinstance(X_test, str):
            X_test[date_col].fillna(fillnum,inplace=True)
            X_test, _ = FE_create_time_series_features(X_test, date_col, ts_adds)
        print('        Adding time series features from %s to data...' %date_col)
    ####### Set up feature to encode  ####################
    ##### First make sure that the originals are not modified ##########
    X_train_encoded = copy.deepcopy(X_train)
    X_test_encoded = copy.deepcopy(X_test)
    feature_to_encode = X_train.select_dtypes(include='object').columns.tolist(
                    )+X_train.select_dtypes(include='category').columns.tolist()
    #### Do label encoding now #################
    if enc_method == 'label':
        for feat in feature_to_encode:
            # Initia the encoder model
            lbEncoder = My_LabelEncoder()
            fillnum = X_train[feat].mode()[0]
            X_train[feat].fillna(fillnum,inplace=True)
            # fit the train data
            lbEncoder.fit(X_train[feat])
            # transform training set
            X_train_encoded[feat] = lbEncoder.transform(X_train[feat])
            # transform test set
            if not isinstance(X_test_encoded, str):
                X_test[feat].fillna(fillnum,inplace=True)
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
    if isinstance(X_train_encoded, np.ndarray):
        X_train_scaled = pd.DataFrame(scaler.transform(X_train_encoded))
    else:
        X_train_scaled = pd.DataFrame(scaler.transform(X_train_encoded), 
            columns=X_train_encoded.columns, index=X_train_encoded.index)

    # transform test set
    if not isinstance(X_test_encoded, str):
        if isinstance(X_test_encoded, np.ndarray):
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), 
                )
        else:
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), 
                columns=X_test_encoded.columns, index=X_test_encoded.index)
    else:
        X_test_scaled = ""

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
def lightgbm_model_fit(random_search_flag, x_train, y_train, x_test, y_test, modeltype,
                         multi_label, log_y, model=""):
    try:
        import lightgbm as lgbm
    except:
        print('Please pip install lightgbm before trying this function. Returning...')
        return None
    #####################################################################
    start_time = time.time()
    if multi_label:
        ######   This is for Multi_Label problems ############
        rand_params = {'estimator__learning_rate':[0.1, 0.5, 0.01, 0.05],
          'estimator__n_estimators':[50, 100, 150, 200, 250],
          #'estimator__gamma':[0, 2, 4, 8, 16, 32], ## there is no gamma in LGBM models ##
          'estimator__max_depth':[3, 5, 8, 12],
          'estimator__class_weight':[None, 'balanced']
          }
    else:
        rand_params = {
            'learning_rate': sp.stats.uniform(scale=1),
            'num_leaves': sp.stats.randint(20, 100),
           'n_estimators': sp.stats.randint(100,500),
            "max_depth": sp.stats.randint(3, 15),
            'class_weight':[None, 'balanced']
                }
    gpu_exists = check_if_GPU_exists()
    if modeltype == 'Regression':
        if gpu_exists:
            lgb = lgbm.LGBMRegressor(device="gpu")
        else:
            lgb = lgbm.LGBMRegressor()
        objective = 'regression' 
        metric = 'rmse'
        is_unbalance = False
        class_weight = None
        score_name = 'Score'
    else:
        if modeltype =='Binary_Classification':
            if gpu_exists:
                lgb = lgbm.LGBMClassifier(device="gpu")
            else:
                lgb = lgbm.LGBMClassifier()
            objective = 'binary'
            metric = 'auc'
            is_unbalance = True
            class_weight = None
            score_name = 'ROC AUC'
            num_class = 1
        else:
            if gpu_exists:
                lgb = lgbm.LGBMClassifier(device="gpu")
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
        lgbm_params = {
                       'objective': objective,
                       'metric': metric,
                       'boosting_type': 'gbdt',
                       'save_binary': True,
                       'seed': 1337, 'feature_fraction_seed': 1337,
                       'bagging_seed': 1337, 'drop_seed': 1337, 
                       'data_random_seed': 1337,
                       'verbose': -1, 
                        'n_estimators': 400,
                    }
    else:
        if multi_label:
            ### If it is multi_label, having fewer params help avoid errors ##
            ### Also LGBM doesn't work well when there are binary and multiclass mixed in multi-labels ##
            lgbm_params = {
                           'boosting_type': 'gbdt',
                           'seed': 1337, 'feature_fraction_seed': 1337,
                           'bagging_seed': 1337, 'drop_seed': 1337, 
                           'data_random_seed': 1337,
                           'verbose': -1, 
                            'n_estimators': 400,
                    }
        else:
            lgbm_params = {
                           'objective': objective,
                           'metric': metric,
                           'boosting_type': 'gbdt',
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
    ### Don't change the next line. It has to be lgb to refer to the model!!
    lgb.set_params(**lgbm_params)
    if multi_label:
        if modeltype == 'Regression':
            lgb = MultiOutputRegressor(lgb)
        else:
            lgb = MultiOutputClassifier(lgb)
        if random_search_flag:
            if modeltype == 'Regression':
                scoring = 'neg_mean_squared_error'
            else:
                scoring = 'precision'
            model = RandomizedSearchCV(lgb,
                       param_distributions = rand_params,
                       n_iter = 15,
                       return_train_score = True,
                       random_state = 99,
                       n_jobs=-1,
                       cv = 3,
                       refit=True,
                       scoring = scoring,
                       verbose = False)        
            model.fit(x_train, y_train)
            print('Time taken for Hyper Param tuning of multi_label LightGBM (in minutes) = %0.1f' %(
                                            (time.time()-start_time)/60))
            cv_results = pd.DataFrame(model.cv_results_)
            if modeltype == 'Regression':
                print('Mean cross-validated train %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_train_score'].mean()))))
                print('Mean cross-validated test %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_test_score'].mean()))))
            else:
                print('Mean cross-validated train %s = %0.04f' %(score_name, cv_results['mean_train_score'].mean()))
                print('Mean cross-validated test %s = %0.04f' %(score_name, cv_results['mean_test_score'].mean()))
            ### In this case, there is no boost rounds so just return the default num_boost_round
            return model.best_estimator_
        else:
            try:
                model.fit(x_train, y_train)
            except:
                print('Multi_label LightGBM model is crashing during training. Please check your inputs and try again...')
            return model
    else:
        ########   Single Label problems ############
        if random_search_flag:
            if modeltype == 'Regression':
                scoring = 'neg_mean_squared_error'
            else:
                scoring = 'precision'
            model = RandomizedSearchCV(lgb,
                       param_distributions = rand_params,
                       n_iter = 10,
                       return_train_score = True,
                       random_state = 99,
                       n_jobs=-1,
                       cv = 3,
                       refit=True,
                       scoring = scoring,
                       verbose = False)
            ##### This is where we search for hyper params for model #######
            if multi_label:
                model.fit(x_train, y_train)
            else:
                model.fit(x_train, y_train, **early_stopping_params)
            print('Time taken for Hyper Param tuning of LGBM (in minutes) = %0.1f' %(
                                            (time.time()-start_time)/60))
            cv_results = pd.DataFrame(model.cv_results_)
            if modeltype == 'Regression':
                print('Mean cross-validated train %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_train_score'].mean()))))
                print('Mean cross-validated test %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_test_score'].mean()))))
            else:
                print('Mean cross-validated train %s = %0.04f' %(score_name, cv_results['mean_train_score'].mean()))
                print('Mean cross-validated test %s = %0.04f' %(score_name, cv_results['mean_test_score'].mean()))
        else:
            try:
                model.fit(x_train, y_train,  verbose=-1)
            except:
                print('lightgbm model is crashing. Please check your inputs and try again...')
        return model
##############################################################################################
import os
def check_if_GPU_exists(verbose=0):
    try:
        os.environ['NVIDIA_VISIBLE_DEVICES']
        if verbose:
            print('GPU active on this device')
        return True
    except:
        if verbose:
            print('No GPU active on this device')
        return False
#############################################################################################
def complex_XGBoost_model(X_train, y_train, X_test, log_y=False, GPU_flag=False,
                                scaler = '', enc_method='label', n_splits=5, verbose=0):
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
    scaler : default is empty string which means to use StandardScaler.
            But you can explicity send in "minmax' to select MinMaxScaler().
            Alternatively, you can send in a scaler object that you define here: MaxAbsScaler(), etc.
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
    num_boost_round = 400
    if isinstance(Y_XGB, pd.Series):
        targets = [Y_XGB.name]
    elif isinstance(Y_XGB, np.ndarray):
        print('   y input is an numpy array and hence convert into a series or dataframe and re-try.')
        return
    else:
        targets = Y_XGB.columns.tolist()
    if len(targets) == 1:
        multi_label = False
        if isinstance(Y_XGB, pd.DataFrame):
            Y_XGB = pd.Series(Y_XGB.values.ravel(),name=targets[0], index=Y_XGB.index)
    else:
        multi_label = True
    modeltype, _ = analyze_problem_type(Y_XGB, targets)
    ### XGBoost #####
    if modeltype == 'Binary_Classification':
        print('# XGBoost is a good choice since it is best for binary classification problems.')
    elif modeltype == 'Multi_Classification':
        print('# XGBoost is a poor choice for this problem since LightGBM is better for multi-class')
    else:
        print('# Simple XGBoost is better than Complex_XGBoost for Regression problems')

    columns =  X_XGB.columns
    ###################################################################################
    #########     S C A L E R     P R O C E S S I N G      B E G I N S    ############
    ###################################################################################
    if isinstance(scaler, str):
        if not scaler == '':
            scaler = scaler.lower()
            if scaler == 'standard':
                scaler = StandardScaler()
            elif scaler == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()
    else:
        pass
    #################################################################################
    if modeltype == 'Regression':
        if log_y:
            Y_XGB.loc[Y_XGB==0] = 1e-15  ### just set something that is zero to a very small number

    #########  Now set the number of rows we need to tune hyper params ###
    scoreFunction = { "precision": "precision_weighted","recall": "recall_weighted"}
    random_search_flag =  True

      #### We need a small validation data set for hyper-param tuning #########################
    hyper_frac = 0.2
    #### now select a random sample from X_XGB ##
    if modeltype == 'Regression':
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                            random_state=999)
    else:
        try:
            X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                            random_state=999, stratify = Y_XGB)
        except:
            ## In some small cases there are too few samples to stratify hence just split them as is 
            X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                                random_state=999)        
    ######  This step is needed for making sure y is transformed to log_y ####################
    if modeltype == 'Regression' and log_y:
            Y_train = np.log(Y_train)
            Y_valid = np.log(Y_valid)
    
    #### First convert test data into numeric using train data ###
    X_train, Y_train, X_valid, Y_valid = data_transform(X_train, Y_train, X_valid, Y_valid,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

    
    ######  Time to hyper-param tune model using randomizedsearchcv and partial train data #########
    num_boost_round = xgbm_model_fit(random_search_flag, X_train, Y_train, X_valid, Y_valid, modeltype,
                         multi_label, log_y, num_boost_round=num_boost_round)

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
            xgb.plot_importance(each_model, importance_type='gain', max_num_features=top_num,
                title='XGBoost model feature importances for %s' %target_name)
    else:
        xgb.plot_importance(model, importance_type='gain', max_num_features=top_num,
                            title='XGBoost final model feature importances')
    print('Returning the following:')
    print('    Model = %s' %model)
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            print('    final predictions', pred_xgbs[:10])
        return (pred_xgbs, model)
    else:
        if not isinstance(X_XGB_test, str):
            print('    final predictions (may need to be transformed to original labels)', pred_xgbs[:10])
            if isinstance(pred_probas, list):
                print('    predicted probabilities shape = [(%s, %s)...]' 
                    %(pred_probas[0].shape[0],pred_probas[0].shape[1]))
            else:
                print('    predicted probabilities', pred_probas[:4])
        return (pred_xgbs, pred_probas, model)
##############################################################################################
import xgboost as xgb
def xgbm_model_fit(random_search_flag, x_train, y_train, x_test, y_test, modeltype,
                         multi_label, log_y, num_boost_round=100):
    start_time = time.time()
    if multi_label and not random_search_flag:
        model = num_boost_round
    else:
        rand_params = {
            'learning_rate': sp.stats.uniform(scale=1),
            'gamma': sp.stats.randint(0, 32),
            'n_estimators': sp.stats.randint(100,500),
            "max_depth": sp.stats.randint(3, 15),
            'class_weight':[None, 'balanced'],
                }
    #####   Set the params for GPU and CPU here ###
    tree_method = 'hist'
    if check_if_GPU_exists():
        tree_method = 'gpu_hist'
    ######   This is where we set the default parameters ###########
    if modeltype == 'Regression':
        objective = 'reg:squarederror' 
        eval_metric = 'rmse'
        shuffle = False
        stratified = False
        num_class = 0
        score_name = 'Score'
        scale_pos_weight = 1
    else:
        if modeltype =='Binary_Classification':
            objective='binary:logistic'
            eval_metric = 'auc' ## dont change this. AUC works well.
            shuffle = True
            stratified = True
            num_class = 1
            score_name = 'AUC'
            scale_pos_weight = get_scale_pos_weight(y_train)
        else:
            objective = 'multi:softprob'
            eval_metric = 'auc'  ## dont change this. AUC works well for now.
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
            score_name = 'Multiclass AUC'
            scale_pos_weight = 1  ### use sample_weights in multi-class settings ##
    ######################################################
    final_params = {
          'booster' :'gbtree',
          'random_state': 99,
          'objective': objective,
          'eval_metric': eval_metric,
          'tree_method': tree_method,
          'verbosity': 0,
          'n_jobs': -1,
          'scale_pos_weight':scale_pos_weight,
          'num_class': num_class,
          'silent': True
            }
    #######  This is where we split into single and multi label ############
    if multi_label:
        ######   This is for Multi_Label problems ############
        rand_params = {'estimator__learning_rate':[0.1, 0.5, 0.01, 0.05],
          'estimator__n_estimators':[50, 100, 150, 200, 250],
          'estimator__gamma':[0, 2, 4, 8, 16, 32],
          'estimator__max_depth':[3, 5, 8, 12],
          'estimator__class_weight':[None, 'balanced']
          }
        if random_search_flag:
            if modeltype == 'Regression':
                clf = XGBRegressor(n_jobs=-1, random_state=999, max_depth=6)
                clf.set_params(**final_params)
                model = MultiOutputRegressor(clf, n_jobs=-1)
            else:
                clf = XGBClassifier(n_jobs=-1, random_state=999, max_depth=6)
                clf.set_params(**final_params)
                model = MultiOutputClassifier(clf, n_jobs=-1)
            if modeltype == 'Regression':
                scoring = 'neg_mean_squared_error'
            else:
                scoring = 'precision'
            model = RandomizedSearchCV(model,
                       param_distributions = rand_params,
                       n_iter = 15,
                       return_train_score = True,
                       random_state = 99,
                       n_jobs=-1,
                       cv = 3,
                       refit=True,
                       scoring = scoring,
                       verbose = False)        
            model.fit(x_train, y_train)
            print('Time taken for Hyper Param tuning of multi_label XGBoost (in minutes) = %0.1f' %(
                                            (time.time()-start_time)/60))
            cv_results = pd.DataFrame(model.cv_results_)
            if modeltype == 'Regression':
                print('Mean cross-validated train %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_train_score'].mean()))))
                print('Mean cross-validated test %s = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_test_score'].mean()))))
            else:
                print('Mean cross-validated train %s = %0.04f' %(score_name, cv_results['mean_train_score'].mean()))
                print('Mean cross-validated test %s = %0.04f' %(score_name, cv_results['mean_test_score'].mean()))
            ### In this case, there is no boost rounds so just return the default num_boost_round
            return model.best_estimator_
        else:
            try:
                model.fit(x_train, y_train)
            except:
                print('Multi_label XGBoost model is crashing during training. Please check your inputs and try again...')
            return model
    else:
        #### This is for Single Label Problems #############
        if modeltype == 'Multi_Classification':
            wt_array = get_sample_weight_array(y_train)
            dtrain = xgb.DMatrix(x_train, label=y_train, weight=wt_array)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        ########   Now let's perform randomized search to find best hyper parameters ######
        if random_search_flag:
            cv_results = xgb.cv(final_params, dtrain, num_boost_round=400, nfold=5, 
                stratified=stratified, metrics=eval_metric, early_stopping_rounds=10, seed=999, shuffle=shuffle)
            # Update best eval_metric
            best_eval = 'test-'+eval_metric+'-mean'
            if modeltype == 'Regression':
                mean_mae = cv_results[best_eval].min()
                boost_rounds = cv_results[best_eval].argmin()
            else:
                mean_mae = cv_results[best_eval].max()
                boost_rounds = cv_results[best_eval].argmax()                
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
# Calculate class weight
from sklearn.utils.class_weight import compute_class_weight
import copy
from collections import Counter
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
        print('       Class  -> Counts -> Percent')
        sorted_keys = sorted(counts.keys())
        for cls in sorted_keys:
            print("%12s: % 7d  ->  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))
    if type(pd.Series(counts).idxmin())==str:
        return pd.Series(counts).idxmin()
    else:
        return int(pd.Series(counts).idxmin())
###################################################################################
def get_sample_weight_array(y_train):
    y_train = copy.deepcopy(y_train)    
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    elif isinstance(y_train, pd.Series):
        pass
    elif isinstance(y_train, pd.DataFrame):
        ### if it is a dataframe, return only if it s one column dataframe ##
        y_train = y_train.iloc[:,0]
    else:
        ### if you cannot detect the type or if it is a multi-column dataframe, ignore it
        return None
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    if len(class_weights[(class_weights < 1)]) > 0:
        ### if the weights are less than 1, then divide them until the lowest weight is 1.
        class_weights = class_weights/min(class_weights)
    else:
        class_weights = (class_weights)
    ### even after you change weights if they are all below 1.5 do this ##
    #if (class_weights<=1.5).all():
    #    class_weights = np.around(class_weights+0.49)
    class_weights = class_weights.astype(int)    
    wt = dict(zip(classes, class_weights))

    ### Map class weights to corresponding target class values
    ### You have to make sure class labels have range (0, n_classes-1)
    wt_array = y_train.map(wt)
    #set(zip(y_train, wt_array))

    # Convert wt series to wt array
    wt_array = wt_array.values
    return wt_array
###############################################################################
from collections import OrderedDict
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import copy
def get_class_weights(y_input):    
    ### get_class_weights has lower ROC_AUC but higher F1 scores than get_class_distribution
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
    classes = np.unique(y_input)
    xp = Counter(y_input)    
    if len(class_weights[(class_weights < 1)]) > 0:
        ### if the weights are less than 1, then divide them until the lowest weight is 1.
        class_weights = class_weights/min(class_weights)
    else:
        class_weights = (class_weights)
    ### This is the best version that returns correct weights ###   
    class_weights = class_weights.astype(int)
    class_weights[(class_weights<1)]=1
    class_weights_dict_corrected = dict(zip(classes,class_weights))
    return class_weights_dict_corrected
##################################################################################
from collections import OrderedDict
def get_scale_pos_weight(y_input):
    class_weighted_rows = get_class_weights(y_input)
    rare_class = find_rare_class(y_input)
    rare_class_weight = class_weighted_rows[rare_class]
    print('    For class %s, weight = %s' %(rare_class, rare_class_weight))
    return rare_class_weight
############################################################################################
def xgboost_model_fit(model, x_train, y_train, x_test, y_test, modeltype, log_y, params, 
                    cpu_params, early_stopping_params={}):
    early_stopping = 10
    start_time = time.time()
    if str(model).split("(")[0] == 'RandomizedSearchCV':
        #### we need to set the xgboost version fixed at 1.5 otherwise error!
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
                    eval_metric = 'auc'
                else:
                    objective='multi:softprob'
                    eval_metric = 'auc'
                model.fit(x_train, y_train, early_stopping_rounds=early_stopping, eval_metric = eval_metric,
                                eval_set=[(x_test, y_test)], verbose=0)
        except:
            print('GPU is present but not turned on. Please restart after that. Currently using CPU...')
            if str(model).split("(")[0] == 'RandomizedSearchCV':
                xgb = model.estimator_
                xgb.set_params(**cpu_params)
                if modeltype == 'Regression':
                    scoring = 'neg_mean_squared_error'
                else:
                    scoring = 'precision'
                model = RandomizedSearchCV(xgb,
                                           param_distributions = params,
                                           n_iter = 15,
                                           n_jobs=-1,
                                           cv = 3,
                                           scoring=scoring,
                                           refit=True,
                                           )
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
    try:
        import lightgbm as lgbm
        from tqdm import tqdm
    except:
        print('Please pip install xgboost tqdm before trying this function. Returning...')
        return None
    X_XGB = copy.deepcopy(X_train)
    Y_XGB = copy.deepcopy(y_train)
    X_XGB_test = copy.deepcopy(X_test)
    start_time = time.time()
    if isinstance(Y_XGB, pd.Series):
        targets = [Y_XGB.name]
    elif isinstance(Y_XGB, np.ndarray):
        print('   y input is an numpy array and hence convert into a series or dataframe and re-try.')
        return
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
    modeltype, _ = analyze_problem_type(Y_XGB, targets)
    ### XGBoost #####
    if modeltype == 'Binary_Classification':
        print('# XGBoost is a good choice since it is better for binary classification problems.')
    elif modeltype == 'Multi_Classification':
        print('# Avoid XGBoost for this problem since LightGBM is better for multi-class than XGBoost.')
    else:
        print('# Simple XGBoost is a good choice compared to complex_XGBoost_model for Regression problems.')

    columns =  X_XGB.columns
    ###################################################################################
    #########     S C A L E R     P R O C E S S I N G      B E G I N S    ############
    ###################################################################################
    if isinstance(scaler, str):
        if not scaler == '':
            scaler = scaler.lower()
            if scaler == 'standard':
                scaler = StandardScaler()
            elif scaler == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()
    else:
        pass
    #########     G P U     P R O C E S S I N G      B E G I N S    ############
    ###### This is where we set the CPU and GPU parameters for XGBoost
    if GPU_flag:
        GPU_exists = check_if_GPU_exists(verbose)
    else:
        GPU_exists = False
    #####   Set the Scoring Parameters here based on each model and preferences of user ###
    cpu_params = {}
    param = {}
    tree_method = 'hist'
    if GPU_exists:
        tree_method = 'gpu_hist'
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
                          tree_method=tree_method,
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
            eval_metric = 'auc'
            score_name = 'ROC AUC'
        else:
            objective='multi:softprob'
            eval_metric = 'auc'
            score_name = 'Multiclass ROC AUC'
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
                         tree_method=tree_method,
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

    if modeltype == 'Regression':
        scoring = 'neg_mean_squared_error'
    else:
        scoring = 'precision'
    model = RandomizedSearchCV(xgb.set_params(**param),
                                       param_distributions = params,
                                       n_iter = 15,
                                       return_train_score = True,
                                       random_state = 99,
                                       n_jobs=-1,
                                       cv = 3,
                                       refit=True,
                                       scoring=scoring,
                                       verbose = False)

    X_train, Y_train, X_valid, Y_valid = data_transform(X_train, Y_train, X_valid, Y_valid,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)

    #### Don't move this. It has to be done after you transform Y_valid to numeric ########
    early_stopping_params={"early_stopping_rounds":5,
                "eval_metric" : eval_metric, 
                "eval_set" : [[X_valid, Y_valid]]
               }
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
    if not isinstance(X_XGB_test, str):
        pred_xgbs = model.predict(X_XGB_test_enc)
        if modeltype != 'Regression':
            pred_probas = model.predict_proba(X_XGB_test_enc)
        else:
            pred_probas = np.array([])
    else:
        pred_xgbs = np.array([])
        pred_probas = np.array([])
    print('    Time taken for training XGBoost (in minutes) = %0.1f' %((time.time()-start_time1)/60))
    if verbose:
        plot_importances_XGB(train_set=X_XGB, labels=Y_XGB, ls=ls, y_preds=pred_xgbs,
                            modeltype=modeltype, top_num='all')
    print('Returning the following:')
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            print('    final predictions', pred_xgbs[:10])
        else:
            print('    no X_test given. Returning empty array.')
        print('    Model = %s' %model)
        return (pred_xgbs, model)
    else:
        if not isinstance(X_XGB_test, str):
            print('    final predictions (may need to be transformed to original labels)', pred_xgbs[:10])
            if isinstance(pred_probas, list):
                print('    predicted probabilities shape = [(%s, %s)...]' 
                    %(pred_probas[0].shape[0],pred_probas[0].shape[1]))
            else:
                print('    predicted probabilities', pred_probas[:4])
        else:
            print('    no X_test given. Returning empty array.')
        print('    Model = %s' %model)
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
    try:
        import lightgbm as lgbm
    except:
        print('Please pip install lightgbm before trying this function. Returning...')
        return None
    X_XGB = copy.deepcopy(X_train)
    Y_XGB = copy.deepcopy(y_train)
    X_XGB_test = copy.deepcopy(X_test)
    ####################################
    start_time = time.time()
    top_num = 10
    if isinstance(Y_XGB, pd.Series):
        targets = [Y_XGB.name]
    elif isinstance(Y_XGB, pd.DataFrame):
        targets = Y_XGB.columns.tolist()
    elif isinstance(Y_XGB, np.ndarray):
        print('   y input is an numpy array and hence convert into a series or dataframe and re-try.')
        return
    else:
        print('Dont use complex LightGBM models for single label problems. Try simple_LightGBM_model instead.')
        return
    if len(targets) == 1:
        multi_label = False
        if isinstance(Y_XGB, pd.DataFrame):
            Y_XGB = pd.Series(Y_XGB.values.ravel(),name=targets[0], index=Y_XGB.index)
    else:
        multi_label = True
    modeltype, _ = analyze_problem_type(Y_XGB, targets)
    ### LightGBM #####
    if modeltype == 'Binary_Classification':
        print('# LightGBM is not a good choice since XGBoost is better for binary classification problems.')
    elif modeltype == 'Multi_Classification':
        print('# LightGBM is a good choice since it is better for multi-class than XGBoost.')
    else:
        print('# LightGBM is a poor choice compared to XGBoost for Regression problems.')

    columns =  X_XGB.columns
    #### In some cases, there are special chars in column names. Remove them. ###
    if np.array([':' in x for x in columns]).any():
        sel_preds = columns[np.array([':' in x for x in columns])].tolist()
        print('removing special char : in %s since LightGBM does not like it...' %sel_preds)
        columns = ["_".join(x.split(":")) for x in columns]
        X_XGB.columns = columns
        if not isinstance(X_XGB_test, str):
            X_XGB_test.columns = columns
    ###################################################################################
    #########     S C A L E R     P R O C E S S I N G      B E G I N S    ############
    ###################################################################################
    if isinstance(scaler, str):
        if not scaler == '':
            scaler = scaler.lower()
            if scaler == 'standard':
                scaler = StandardScaler()
            elif scaler == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()
    else:
        pass
    #############################################################################
    #########     G P U     P R O C E S S I N G      B E G I N S    #############
    #############################################################################
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
        try:
            X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                            random_state=999, stratify = Y_XGB)
        except:
            ## In some small cases, you cannot stratify since there are too few samples. So leave it as is ##
            X_train, X_valid, Y_train, Y_valid = train_test_split(X_XGB, Y_XGB, test_size=hyper_frac, 
                                random_state=999)

    #### First convert test data into numeric using train data ###
    X_train, Y_train, X_valid, Y_valid = data_transform(X_train, Y_train, X_valid, Y_valid,
                                modeltype, multi_label, scaler=scaler, enc_method=enc_method)
    ######  This step is needed for making sure y is transformed to log_y ######
    if modeltype == 'Regression' and log_y:
            Y_train = np.log(Y_train)
            Y_valid = np.log(Y_valid)

    random_search_flag =  True
    ######  Time to hyper-param tune model using randomizedsearchcv #########
    gbm_model = lightgbm_model_fit(random_search_flag, X_train, Y_train, X_valid, Y_valid, modeltype,
                         multi_label, log_y, model="")
    if multi_label:
        model = copy.deepcopy(gbm_model)
    else:
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
        print('Top 10 features:\n', pd.DataFrame(
        model.booster_.feature_importance(importance_type='gain'),index=columns,
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
                pred_xgbs = pred_xgbs.reshape(-1, len(predsy))
        else:
            pred_xgbs = []
            pred_probas = []
    ##### once the entire model is trained on full train data ##################
    print('    Time taken for training Light GBM on entire train data (in minutes) = %0.1f' %(
             (time.time()-start_time)/60))
    if multi_label:
        for i,target_name in enumerate(targets):
            lgbm.plot_importance(model.estimators_[i], importance_type='gain', max_num_features=top_num,
                title='LGBM model feature importances for %s' %target_name)
    else:
        lgbm.plot_importance(model, importance_type='gain', max_num_features=top_num,
                title='LGBM final model feature importances')
    print('Returning the following:')
    print('    Model = %s' %model)
    if modeltype == 'Regression':
        if not isinstance(X_XGB_test, str):
            print('    final predictions', pred_xgbs[:10])
        return (pred_xgbs, model)
    else:
        if not isinstance(X_XGB_test, str):
            print('    final predictions (may need to be transformed to original labels)', pred_xgbs[:10])
            if isinstance(pred_probas, list):
                print('    predicted probabilities shape = [(%s, %s)...]' 
                    %(pred_probas[0].shape[0],pred_probas[0].shape[1]))
            else:
                print('    predicted probabilities', pred_probas[:4])
        return (pred_xgbs, pred_probas, model)
#############################################################################################
import copy
import time
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
import pdb
def simple_LightGBM_model(X_train, y_train, X_test, log_y=False, 
        GPU_flag=False, scaler='', enc_method='label', n_splits=5, verbose=-1):
    """
    This is a simple lightGBM model that works only on single label problems.
    """
    try:
        import lightgbm as lgbm
        from tqdm import tqdm
    except:
        print('Please pip install lightgbm tqdm before trying this function. Returning...')
        return None
    X_index = X_train.index
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train, index=X_index)
    num_splits = 5
    num_repeats = 5
    if X_train.shape[0] <= 100000:
        n_estimators = 1000  # LightGBM is fast but mak sure it is small
        early_stopping_rounds = 100
    else:
        n_estimators = 500  # make sure it is small
        early_stopping_rounds = 10
    ###### Now you can set these defaults ##
    verbose = False
    SEED = 42
    X_XGB = copy.deepcopy(X_train)
    Y_XGB = copy.deepcopy(y_train)
    X_XGB_test = copy.deepcopy(X_test)
    ####################################
    start_time = time.time()
    top_num = 10
    num_boost_round = 400
    if isinstance(Y_XGB, pd.Series):
        targets = [Y_XGB.name]
    elif isinstance(Y_XGB, np.ndarray):
        print('   y input is an numpy array and hence convert into a series or dataframe and re-try.')
        return
    else:
        targets = Y_XGB.columns.tolist()
    if len(targets) == 1:
        multi_label = False
        if isinstance(Y_XGB, pd.DataFrame):
            Y_XGB = pd.Series(Y_XGB.values.ravel(),name=targets[0], index=Y_XGB.index)
    else:
        multi_label = True
    modeltype, multi_label = analyze_problem_type(Y_XGB, targets)
    print('* LightGBM model training started... *')
    if multi_label:
        model_label = 'Multi_Label'
        print('This is a %s problem. You must use complex_LightGBM_model for this dataset.' %model_label)
        return
    else:
        model_label = 'Single_Label'
    columns =  X_XGB.columns
    ##############################################
    rand_params = {
    #'learning_rate': sp.stats.uniform(scale=1),
    #'learning_rate': np.linspace(1e-8,1e-1),
    #'num_leaves': sp.stats.randint(2, 30),
   'n_estimators': sp.stats.randint(100,400),
    #"max_depth": sp.stats.randint(2, 7),
    #'subsample': sp.stats.uniform(scale=1),
    #'colsample_bytree': sp.stats.uniform(scale=1),
    #'class_weight':[None, 'balanced']
        }

    gpu_exists = check_if_GPU_exists(verbose=1)
    if modeltype == 'Regression':
        if gpu_exists:
            lgbmx = lgbm.LGBMRegressor(device="gpu")
        else:
            lgbmx = lgbm.LGBMRegressor()
        objective = 'regression' 
        metric = 'rmse'
        is_unbalance = False
        class_weight = None
        score_name = 'Score'
    else:
        if modeltype =='Binary_Classification':
            if gpu_exists:
                lgbmx = lgbm.LGBMClassifier(device="gpu")
            else:
                lgbmx = lgbm.LGBMClassifier()
            objective = 'binary'
            metric = 'auc'
            is_unbalance = True
            class_weight = None
            score_name = 'ROC AUC'
            num_class = 1
        else:
            if gpu_exists:
                lgbmx = lgbm.LGBMClassifier(device="gpu")
            else:
                lgbmx = lgbm.LGBMClassifier()
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

    if modeltype == 'Regression':
        ## there is no num_class in regression for LGBM model ##        
        lgbm_params = {
                       'objective': objective,
                       'metric': metric,
                       'boosting_type': 'gbdt',
                       'save_binary': True,
                       'seed': 1337, 'feature_fraction_seed': 1337,
                       'bagging_seed': 1337, 'drop_seed': 1337, 
                       'data_random_seed': 1337,
                       'verbose': -1, 
                        'n_estimators': n_estimators,
                    }
    else:
        lgbm_params = {
                       'objective': objective,
                       'metric': metric,
                       'boosting_type': 'gbdt',
                       'save_binary': True,
                       'seed': 1337, 'feature_fraction_seed': 1337,
                       'bagging_seed': 1337, 'drop_seed': 1337, 
                       'data_random_seed': 1337,
                       'verbose': -1, 
                       'num_class': num_class,
                       'is_unbalance': is_unbalance,
                       'class_weight': class_weight,
                        'n_estimators': n_estimators,
                }

    lgbm_copy = copy.deepcopy(lgbmx)
    lgb_importances = pd.DataFrame()
    lgb_oof = np.zeros(X_train.shape[0])
    lgb_pred = np.zeros(X_test.shape[0])

    start = time.time()
    if modeltype == 'Regression':
        scoring = 'neg_mean_squared_error'
        score_name = 'MSE'
    else:
        if modeltype =='Binary_Classification':
            scoring = 'roc_auc'
            score_name = 'ROC AUC'
        else:
            scoring = 'balanced_accuracy'
            score_name = 'balanced_accuracy'            
    print('Starting Hyper Param tuning of %s lightGBM model. This will take time...' %model_label)
    lgbmx.set_params(**lgbm_params)

    model = RandomizedSearchCV(lgbmx,
               param_distributions = rand_params,
               n_iter = 5,
               return_train_score = True,
               random_state = 99,
               n_jobs=-1,
               cv = 5,
               refit=True,
               scoring = scoring,
               verbose = False)        
    model.fit(X_train, y_train)
    print('Time taken for Hyper Param tuning of LightGBM model (in minutes) = %0.1f' %(
                                    (time.time()-start_time)/60))
    cv_results = pd.DataFrame(model.cv_results_)
    if modeltype == 'Regression':
        print('    Mean cross-validated train %s score = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_train_score'].mean()))))
        print('    Mean cross-validated test %s score = %0.04f' %(score_name, np.sqrt(abs(cv_results['mean_test_score'].mean()))))
    else:
        print('    Mean cross-validated train score %s = %0.04f' %(score_name, cv_results['mean_train_score'].mean()))
        print('    Mean cross-validated test score %s = %0.04f' %(score_name, cv_results['mean_test_score'].mean()))
    ### In this case, there is no boost rounds so just return the default num_boost_round
    best_params = model.best_params_
    print('    Hyper tuned params are: %s' %best_params)
    start_time = time.time()
    i = 0
    scores = []
    if modeltype == 'Regression':
        repeat_strats = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=SEED)
    else:
        repeat_strats = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=SEED)
    #### This is where we repeatedly make training and predictions ######################
    for fold, (trn_idx, val_idx) in enumerate(repeat_strats.split(X=X_train, y=y_train)):
        start_time = time.time()
        xx_train, yy_train = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        xx_valid, yy_valid = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        es = lgbm.early_stopping(
            stopping_rounds=early_stopping_rounds,
            first_metric_only=True,
            verbose=verbose,
        )
        
        #le = lgbm.log_evaluation(
        #    period=verbose,
        #    show_stdv=verbose
        #)
        ########## Now train the model ###########
        lgbm_copy.set_params(**lgbm_params)
        lgbm_copy.set_params(**best_params)
        model = copy.deepcopy(lgbm_copy)
        if i == 0:
            print('    Training hyper-tuned', model)
        i += 1
        
        model.fit(
            xx_train, 
            yy_train,
            eval_set=[(xx_valid, yy_valid)],
            eval_names=['train', 'valid'],
            eval_metric=metric,
            callbacks=[es],
        )
        
        fi_tmp = pd.DataFrame()
        fi_tmp['feature'] = xx_train.columns.tolist()
        fi_tmp['importance'] = model.feature_importances_
        fi_tmp['fold'] = fold
        fi_tmp['seed'] = SEED
        lgb_importances = lgb_importances.append(fi_tmp)

        lgb_oof[val_idx] = model.predict(xx_valid)
        lgb_pred += model.predict(X_test) / num_splits / num_repeats

        if modeltype == 'Regression':
            auc = np.sqrt(mean_squared_error(yy_valid, lgb_oof[val_idx]))
            scores.append(auc)
            print(f"    iteration {i}: RMSE: {auc:.2f}")
        else:
            if modeltype =='Binary_Classification':
                auc = roc_auc_score(yy_valid, lgb_oof[val_idx])
                scores.append(auc)
                print(f"    iteration {i}: ROC AUC: {auc:.2f}")
            else:
                auc = f1_score(yy_valid, lgb_oof[val_idx], average="macro")
                #auc = roc_auc_score(yy_valid, lgb_oof[val_idx], multi_class='ovr',average="macro")
                scores.append(auc)
                print(f"    iteration {i}: Macro F1 score: {auc:.2f}")
            
    elapsed = time.time() - start_time
    
    if modeltype == 'Regression':
        auc = np.sqrt(mean_squared_error(y_train, lgb_oof))
        print(f"Average Train RMSE: {auc:6f}, elapsed time: {elapsed:.0f} seconds")
    else:
        if modeltype =='Binary_Classification':
            auc = roc_auc_score(y_train, lgb_oof)
            print(f"Average Train AUC: {auc:6f}, elapsed time: {elapsed:.0f} seconds")
        else:
            auc = f1_score(y_train, lgb_oof, average="macro")
            print(f"Average Train Macro F1 score: {auc:6f}, elapsed time: {elapsed:.0f} seconds")

    #### Now change the probas to fit within 0 and 1 #######
    MM = MinMaxScaler()
    print('Fitting model on entire train dataset...')
    start_time = time.time()
    model = copy.deepcopy(lgbm_copy)
    model.fit(
        X_train, 
        y_train,
        )
    elapsed = time.time() - start_time
    print(f"    Training time: {elapsed:.0f} seconds")
    order = list(lgb_importances.groupby("feature").mean().sort_values("importance", ascending=False).index)
    plt.figure(figsize=(12, 4), tight_layout=True)
    sns.barplot(x="importance", y="feature", data=lgb_importances, order=order[:15])
    plt.title("{} feature importances".format("lgb"))
    plt.tight_layout()
    if modeltype == 'Regression':
        lgb_preds = lgb_pred.ravel()
        lgb_probas = np.array([])
    else:
        lgb_probas = MM.fit_transform(lgb_pred.reshape(-1, 1))
        lgb_preds = (lgb_probas>0.5).astype(int)
    print('Returning the following:')
    print('    final predictions sample', lgb_preds[:4])
    print('    predicted probabilities sample', lgb_probas[:4])
    print('    Model = %s' %model)
    return lgb_preds, lgb_probas, model
########################################################################################
def plot_importances_XGB(train_set, labels, ls, y_preds, modeltype, top_num='all'):
    add_items=0
    for item in ls:
        add_items +=item
    
    if isinstance(top_num, str):
        feat_imp=pd.DataFrame(add_items/len(ls),index=train_set.columns,
            columns=["importance"]).sort_values('importance', ascending=False)
        feat_imp2=feat_imp[feat_imp>0.00005]
        #df_cv=df_cv.reset_index()
        #### don't add [:top_num] at the end of this statement since it will error #######
        #feat_imp = pd.Series(df_cv.importance.values, 
        #    index=df_cv.drop(["importance"], axis=1)).sort_values(axis='index',ascending=False)
    else:
        ## this limits the number of items to the top_num items
        feat_imp=pd.DataFrame(add_items/len(ls),index=train_set.columns[:top_num],
            columns=["importance"]).sort_values('importance', ascending=False)
        feat_imp2=feat_imp[feat_imp>0.00005]
        #df_cv=df_cv.reset_index()
        #feat_imp = pd.Series(df_cv.importance.values, 
        #    index=df_cv.drop(["importance"], axis=1)).sort_values(axis='index',ascending=False)[:top_num]
    ##### Now plot the feature importances #################        
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
    if isinstance(top_num, str):
        feat_imp2[:].plot(kind='barh', ax=ax1, title='Feature importances of model on test data')
    else:
        feat_imp2[:top_num].plot(kind='barh', ax=ax1, title='Feature importances of model on test data')
    if modeltype == 'Regression':
        ax2=plt.subplot(2, 2, 2)
        pd.Series(y_preds).plot(ax=ax2, color='b', title='Model predictions on test data');
    else:
        ax2=plt.subplot(2, 2, 2)
        pd.Series(y_preds).hist(ax=ax2, color='b', label='Model predictions histogram on test data');
##################################################################################
def analyze_problem_type(y_train, target, verbose=0) :  
    y_train = copy.deepcopy(y_train)
    cat_limit = 30 ### this determines the number of categories to name integers as classification ##
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    if isinstance(target, str):
        multi_label = False
        string_target = True
    else:
        if len(target) == 1:
            multi_label = False
            string_target = False
        else:
            multi_label = True
            string_target = False

    ####  This is where you detect what kind of problem it is #################
    if string_target or type(y_train) == pd.Series:
        ## If target is a string then we should test for dtypes this way #####
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
        for i in range(y_train.shape[1]):
            ### if target is a list, then we should test dtypes a different way ###
            if y_train.dtypes.values.all() in ['int64', 'int32','int16']:
                if len(np.unique(y_train.iloc[:,0])) <= 2:
                    model_class = 'Binary_Classification'
                elif len(np.unique(y_train.iloc[:,0])) > 2 and len(np.unique(y_train.iloc[:,0])) <= cat_limit:
                    model_class = 'Multi_Classification'
                else:
                    model_class = 'Regression'
            elif  y_train.dtypes.values.all() in ['float16','float32','float64']:
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
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone

class IterativeBestClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom classifier for binary and multi-class classification problems
    that employs an iterative one-vs-rest approach. For each class in the dataset,
    a separate classifier is trained to distinguish that particular class from all other classes.
    This class is particularly effective when used with ensemble methods like BlaggingClassifier
    or gradient boosting algorithms such as XGBClassifier, LGBMClassifier, and CatBoostClassifier.

    Parameters:
    -----------
    base_estimator : estimator object (default=XGBClassifier(n_estimators=100, random_state=99))
        The base estimator from which the IterativeBestClassifier is built. This is the model
        that will be trained on each one-vs-rest classification task. It should be a classifier
        that supports the `fit` and `predict_proba` methods.

    Attributes:
    -----------
    classifiers : list of estimator objects
        The collection of one-vs-rest classifiers trained during the fitting process. Each classifier
        in this list is an instance of the `base_estimator` trained to distinguish one class from the rest.

    classes_ : array of shape (n_classes,)
        The classes labels.

    Methods:
    --------
    fit(X, y):
        Fit the IterativeBestClassifier to the training data. The method trains a separate classifier
        for each class in `y`, using `X` as the training data.

    predict(X):
        Perform classification on samples in `X`. For each sample, the method returns the class label
        that has the highest probability score across all one-vs-rest classifiers.

    predict_proba(X):
        Return probability estimates for all classes for each sample in `X`. The probability
        of each class is computed as the normalized output of the corresponding one-vs-rest classifier.

    Example:
    --------
    >>> from xgboost import XGBClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, n_classes=3)
    >>> clf = IterativeBestClassifier(base_estimator=XGBClassifier(n_estimators=100, random_state=99))
    >>> clf.fit(X, y)
    >>> print(clf.predict(X[:5]))

    Notes:
    ------
    The effectiveness of the IterativeBestClassifier depends heavily on the choice of `base_estimator`.
    It's recommended to experiment with different base estimators and their hyperparameters to achieve
    optimal performance.
    """
    def __init__(self, base_estimator=XGBClassifier(n_estimators=100,
                    random_state=99)):
        self.base_estimator = base_estimator
        self.classifiers = []
        self.classes_ = []

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.values
        if not isinstance(y, np.ndarray):
            y = y.values
        self.classifiers = []
        unique_classes = sorted(np.unique(y))
        self.classes_ = unique_classes

        for class_label in unique_classes:
            binary_y_train = np.where(y == class_label, 1, 0)  # 1 for the current class, 0 for the rest

            clf = clone(self.base_estimator)
            clf.fit(X, binary_y_train)

            self.classifiers.append(clf)

        return self

    def predict(self, X):
        # Initialize a DataFrame to store probability scores for each class
        prob_scores = pd.DataFrame( columns=self.classes_)

        for class_label, clf in zip(self.classes_, self.classifiers):
            # Store the probability of the positive class (class_label)
            prob_scores[class_label] = clf.predict_proba(X)[:, 1]

        # The final prediction is the class with the highest probability score
        final_predictions = prob_scores.idxmax(axis=1)

        return final_predictions

    def predict_proba(self, X):
        # Initialize a DataFrame to store probability scores for each class
        prob_scores = pd.DataFrame( columns=self.classes_)

        for class_label, clf in zip(self.classes_, self.classifiers):
            # Store the probability of the positive class (class_label)
            prob_scores[class_label] = clf.predict_proba(X)[:, 1]

        # Normalize the probabilities so they sum to 1 for each sample
        normalized_probs = prob_scores.div(prob_scores.sum(axis=1), axis=0)

        return normalized_probs.values
#########################################################################
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
class IterativeDoubleClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier designed for binary and multi-class classification problems,
    employing a unique approach by training two separate classifiers for each one-vs-rest
    binary classification task. The predictions from these two classifiers are then
    combined using specified weights to produce the final prediction for each class.

    This approach allows for a more nuanced decision boundary, potentially leading to
    improved performance on complex datasets. It is particularly effective when used
    with a combination of models like BlaggingClassifier and RandomForestClassifier
    for one classifier, and potentially simpler linear models like LinearDiscriminantAnalysis
    for the other, allowing for a blend of model complexities and perspectives.

    Parameters:
    -----------
    base_estimator1 : estimator object, optional (default=None)
        The first base estimator from which the classifier is built. If not provided,
        a default RandomForestClassifier is used.

    base_estimator2 : estimator object, optional (default=None)
        The second base estimator. If not provided, a default LinearDiscriminantAnalysis
        is used.

    weights : dict, optional (default={1: 0.5, 2: 0.5})
        A dictionary specifying the weights for combining the predictions from
        `base_estimator1` and `base_estimator2`. Default is equal weighting.

    Attributes:
    -----------
    classifiers1 : list of estimator objects
        The list of first set of one-vs-rest classifiers trained during the fitting process.

    classifiers2 : list of estimator objects
        The list of second set of one-vs-rest classifiers trained during the fitting process.

    classes_ : array of shape (n_classes,)
        The class labels.

    Methods:
    --------
    fit(X, y):
        Fit the IterativeDoubleClassifier to the training data by training both
        `base_estimator1` and `base_estimator2` for each class in a one-vs-rest fashion.

    predict(X):
        Predict class labels for samples in `X` by combining the predictions from both sets
        of classifiers based on the specified weights.

    predict_proba(X):
        Predict class probabilities for samples in `X` by combining the prediction probabilities
        from both sets of classifiers based on the specified weights, and normalizing them.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)
    >>> clf = IterativeDoubleClassifier()
    >>> clf.fit(X, y)
    >>> print(clf.predict(X[:5]))

    Note:
    -----
    The effectiveness of this classifier is contingent on the complementary nature of the two
    base estimators and the appropriateness of the weights assigned to their predictions.
    """
    def __init__(self, base_estimator1=None, 
        base_estimator2=None, weights=None):
        self.base_estimator1 = base_estimator1
        self.base_estimator2 = base_estimator2
        self.weights = weights if weights else {1: 0.5, 2: 0.5}  # Default to equal weighting if none provided

        self.classifiers1 = []
        self.classifiers2 = []
        self.classes_ = []

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.values
        if not isinstance(y, np.ndarray):
            y = y.values
        if self.base_estimator1 is None:
            class_weights_dict_corrected = get_class_weights(y)
            self.base_estimator1 = RandomForestClassifier(n_estimators=100,#class_weight=class_weights_dict_corrected,
                             random_state=42)
        if self.base_estimator2 is None:
            class_weights_dict_corrected = get_class_weights(y)
            #self.base_estimator2 = RandomForestClassifier(class_weight=class_weights_dict_corrected, random_state=42)
            self.base_estimator2 = LinearDiscriminantAnalysis()
        #### No start training the models ####
        self.classifiers1 = []
        self.classifiers2 = []
        unique_classes = sorted(np.unique(y))
        self.classes_ = unique_classes

        for class_label in unique_classes:
            binary_y_train = np.where(y == class_label, 1, 0)  # 1 for the current class, 0 for the rest

            clf1 = clone(self.base_estimator1)
            clf1.fit(X, binary_y_train)
            self.classifiers1.append(clf1)

            clf2 = clone(self.base_estimator2)
            clf2.fit(X, binary_y_train)
            self.classifiers2.append(clf2)

        return self

    def predict(self, X):
        combined_probs = self.predict_proba(X)
        # Convert combined_probs to a DataFrame with class labels as columns
        prob_df = pd.DataFrame(combined_probs,  columns=self.classes_)
        final_predictions = prob_df.idxmax(axis=1)
        return final_predictions

    def predict_proba(self, X):
        prob_scores = pd.DataFrame(  columns=self.classes_)

        for class_label, (clf1, clf2) in zip(self.classes_, zip(self.classifiers1, self.classifiers2)):
            prob1 = clf1.predict_proba(X)[:, 1]
            prob2 = clf2.predict_proba(X)[:, 1]
            weighted_prob = prob1 * self.weights[1] + prob2 * self.weights[2]
            prob_scores[class_label] = weighted_prob

        normalized_probs = prob_scores.div(prob_scores.sum(axis=1), axis=0)
        return normalized_probs.values

######################################################################################
class IterativeForwardClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, early_estimator=None, 
                 late_estimator=None, 
                 grid_params=None, threshold=None):
        self.early_estimator = early_estimator
        self.late_estimator = late_estimator
        self.threshold = threshold
        self.classifiers = []
        self.classes_ = []
        self.grid_params = grid_params

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.values
        if not isinstance(y, np.ndarray):
            y = y.values
        self.classifiers = []
        unique_classes = sorted(np.unique(y), key=lambda x: -np.sum(y == x))
        self.classes_ = unique_classes
        iteration = 1
        for class_label in unique_classes:
            binary_y_train = np.where(y == class_label, 1, 0)
            if self.early_estimator is None:
                class_weights_dict_corrected = get_class_weights(binary_y_train)
                self.early_estimator = RandomForestClassifier(class_weight=class_weights_dict_corrected, random_state=42)
                self.early_estimator = LinearDiscriminantAnalysis()
            if self.late_estimator is None:
                class_weights_dict_corrected = get_class_weights(binary_y_train)
                self.late_estimator = RandomForestClassifier(class_weight=class_weights_dict_corrected,
                                 random_state=42)
            current_class_size = np.sum(binary_y_train)
            print('Current class size = %s' %current_class_size)
            # Dynamic threshold: e.g., 50% of the current class size
            if self.threshold is None:
                dynamic_threshold = max(100, current_class_size * 0.5)  # Ensuring a minimum threshold
            else:
                dynamic_threshold =  copy.deepcopy(self.threshold)
            print('Current threshold = %s' %dynamic_threshold)
            print('Forward Iteration %s:' % iteration)
            if current_class_size <= dynamic_threshold:
                classifier_string = str(self.late_estimator).split("(")[0]
                print(classifier_string, "used as Late classifier")
                if self.grid_params:  # If hyperparameters are provided
                    if classifier_string in ['SVC', 'NuSVC', 'GaussianNB','LinearDiscriminantAnalysis']:
                        print('    Using RandomizedSearchCV')
                        ### don't use different kernels = just use one sigmoid or rbf
                        randomized_search = RandomizedSearchCV(self.late_estimator, 
                                            param_distributions=self.grid_params, 
                                            n_iter=10, cv=3, n_jobs=-1)
                        randomized_search.fit(X, binary_y_train)
                        clf = randomized_search.best_estimator_
                    else:
                        print('    Using GridSearchCV')
                        grid_search = GridSearchCV(self.late_estimator, self.grid_params, 
                                        cv=3, n_jobs=-1)
                        grid_search.fit(X, binary_y_train)
                        clf = grid_search.best_estimator_
                else:
                    clf = clone(self.late_estimator)
                    try:
                        clf.fit(X, binary_y_train)
                        self.classifiers.append(clf)
                    except Exception as e:
                        print('model erroring due to %s. Continuing...' %e)
            else:
                print(str(self.early_estimator).split("(")[0], "used as Early classifier")
                clf = clone(self.early_estimator)
                try:
                    clf.fit(X, binary_y_train)
                    self.classifiers.append(clf)
                except Exception as e:
                    print('model erroring due to %s. Continuing...' %e)
                
            iteration += 1 
            
            # Filter out the samples classified as the current class for the next iteration
            rest_indices = np.where(clf.predict(X) == 0)[0]
            if len(rest_indices) == 0:
                break
            X, y = X[rest_indices], y[rest_indices]

        return self

    def predict(self, X):
        prob_scores = pd.DataFrame(columns=self.classes_)

        for class_label, clf in zip(self.classes_, self.classifiers):
            try:
                prob_scores[class_label] = clf.predict_proba(X)[:, 1]
            except:
                prob_scores[class_label] = np.vstack([np.zeros(X.shape[0]),clf.predict_proba(X).ravel()]).T[:,1]
        
        final_predictions = prob_scores.fillna(0).idxmax(axis=1)
        return final_predictions

    def predict_proba(self, X):
        prob_scores = pd.DataFrame( columns=self.classes_)

        for class_label, clf in zip(self.classes_, self.classifiers):
            try:
                prob_scores[class_label] = clf.predict_proba(X)[:, 1]
            except:
                prob_scores[class_label] = np.vstack([np.zeros(X.shape[0]),clf.predict_proba(X).ravel()]).T[:,1]

        normalized_probs = prob_scores.div(prob_scores.sum(axis=1), axis=0)
        return normalized_probs.values
##########################################################################
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.base import clone
import pandas as pd

class IterativeBackwardClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, early_estimator=None, 
                 late_estimator=None, 
                 grid_params=None, threshold=None):
        self.early_estimator = early_estimator
        self.late_estimator = late_estimator
        self.threshold = threshold
        self.classifiers = []
        self.classes_ = []
        self.grid_params = grid_params

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.values
        if not isinstance(y, np.ndarray):
            y = y.values
        self.classifiers = []
        # Sort classes by frequency in ascending order
        unique_classes = sorted(np.unique(y), key=lambda x: np.sum(y == x))
        self.classes_ = unique_classes
        iteration = 1

        for class_label in unique_classes:
            binary_y_train = np.where(y == class_label, 1, 0)
            if self.early_estimator is None:
                class_weights_dict_corrected = get_class_weights(binary_y_train)
                #self.early_estimator = RandomForestClassifier(class_weight=class_weights_dict_corrected, random_state=42)
                self.early_estimator = LinearDiscriminantAnalysis()
            if self.late_estimator is None:
                class_weights_dict_corrected = get_class_weights(binary_y_train)
                self.late_estimator = RandomForestClassifier(class_weight=class_weights_dict_corrected,
                                 random_state=42)
            current_class_size = np.sum(binary_y_train)
            print('Current class size = %s' %current_class_size)
            # Dynamic threshold: e.g., 50% of the current class size
            if self.threshold is None:
                dynamic_threshold = max(100, current_class_size * 0.5)  # Ensuring a minimum threshold
            else:
                dynamic_threshold =  copy.deepcopy(self.threshold)
            print('Current threshold = %s' %dynamic_threshold)
            print('Backward Iteration %s:' % iteration)
            if current_class_size <= dynamic_threshold:
                classifier_string = str(self.late_estimator).split("(")[0]
                print(classifier_string, "used as Late classifier")
                if self.grid_params:
                    if classifier_string in ['SVC', 'NuSVC', 'GaussianNB','LinearDiscriminantAnalysis']:
                        print('    Using RandomizedSearchCV')
                        randomized_search = RandomizedSearchCV(self.late_estimator, 
                                            param_distributions=self.grid_params, n_iter=10, cv=3, n_jobs=-1)
                        randomized_search.fit(X, binary_y_train)
                        clf = randomized_search.best_estimator_
                    else:
                        print('    Using GridSearchCV')
                        grid_search = GridSearchCV(self.late_estimator, self.grid_params, cv=3, n_jobs=-1)
                        grid_search.fit(X, binary_y_train)
                        clf = grid_search.best_estimator_
                else:
                    clf = clone(self.late_estimator)
                    clf.fit(X, binary_y_train)
            else:
                print(str(self.early_estimator).split("(")[0], "used as Early classifier")
                clf = clone(self.early_estimator)
                clf.fit(X, binary_y_train)
                
            self.classifiers.append(clf)
            iteration += 1 

            # Filter out the samples classified as the current class for the next iteration
            rest_indices = np.where(clf.predict(X) == 0)[0]
            if len(rest_indices) == 0:
                break
            X, y = X[rest_indices], y[rest_indices]

        return self

    def predict(self, X):
        prob_scores = pd.DataFrame( columns=self.classes_)

        for class_label, clf in zip(self.classes_, self.classifiers):
            try:
                prob_scores[class_label] = clf.predict_proba(X)[:, 1]
            except:
                prob_scores[class_label] = np.vstack([np.zeros(X.shape[0]),clf.predict_proba(X).ravel()]).T[:,1]

        final_predictions = prob_scores.fillna(0).idxmax(axis=1)
        return final_predictions

    def predict_proba(self, X):
        prob_scores = pd.DataFrame(columns=self.classes_)

        for class_label, clf in zip(self.classes_, self.classifiers):
            try:
                prob_scores[class_label] = clf.predict_proba(X)[:, 1]
            except:
                prob_scores[class_label] = np.vstack([np.zeros(X.shape[0]),clf.predict_proba(X).ravel()]).T[:,1]

        normalized_probs = prob_scores.div(prob_scores.sum(axis=1), axis=0)
        return normalized_probs.values
#######################################################################
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd

class IterativeSearchClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that integrates two distinct approaches for classification: a forward pass from
    majority to minority classes and a backward pass from minority to majority classes. It is designed
    to balance class representation and performance across varied class distributions, making it suitable
    for imbalanced datasets.

    The classifier employs two base estimators, one for each pass, and optionally uses grid search
    parameters to optimize their configurations. The final predictions are derived by combining the
    inferences from both the forward and backward classifiers, potentially using a simple voting or
    averaging scheme.

    Parameters:
    -----------
    base_estimator1 : estimator object, optional
        The base estimator used for the forward pass. If not provided, a default will be used
        based on the implementation of `IterativeForwardClassifier`.

    base_estimator2 : estimator object, optional
        The base estimator used for the backward pass. If not provided, a default will be used
        based on the implementation of `IterativeBackwardClassifier`.

    grid_params : dict or list of dictionaries, optional
        The parameters to use for grid search optimization of the base estimators.

    threshold : float, optional
        A threshold used to determine a decision boundary, potentially used by both the
        forward and backward classifiers.

    Attributes:
    -----------
    forward_classifier : IterativeForwardClassifier object
        The classifier instance performing the forward pass.

    backward_classifier : IterativeBackwardClassifier object
        The classifier instance performing the backward pass.

    Methods:
    --------
    fit(X, y):
        Fit the IterativeSearchClassifier to the training data by fitting both the forward and
        backward classifiers.

    predict(X):
        Predict class labels for samples in `X` by averaging the predictions from both the
        forward and backward classifiers.

    predict_proba(X):
        Predict class probabilities for samples in `X` by averaging the prediction probabilities
        from both the forward and backward classifiers, potentially with weights.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, weights=[0.1, 0.2, 0.7])
    >>> clf = IterativeSearchClassifier()
    >>> clf.fit(X, y)
    >>> print(clf.predict(X[:5]))

    Note:
    -----
    The effectiveness of this classifier can be influenced by the choice of base estimators, the
    quality of grid search parameters, and the method used to combine the predictions from the
    forward and backward passes.
    """
    def __init__(self, 
        base_estimator1=None, base_estimator2=None, 
        grid_params=None, threshold=None):
        self.base_estimator1 = base_estimator1
        self.base_estimator2 = base_estimator2
        self.grid_params = grid_params
        self.threshold = threshold
        self.forward_classifier = IterativeForwardClassifier(
            early_estimator=base_estimator1, late_estimator=base_estimator2, 
            grid_params=grid_params, threshold=threshold
        )
        self.backward_classifier = IterativeBackwardClassifier(
            early_estimator=base_estimator1, late_estimator=base_estimator2, 
            grid_params=grid_params, threshold=threshold
        )

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.values
        if not isinstance(y, np.ndarray):
            y = y.values
        self.forward_classifier.fit(X, y)
        self.backward_classifier.fit(X, y)
        return self

    def predict(self, X):
        forward_predictions = self.forward_classifier.predict(X)
        backward_predictions = self.backward_classifier.predict(X)

        # Simple voting or averaging scheme
        final_predictions = (forward_predictions + backward_predictions) / 2
        final_predictions = np.round(final_predictions).astype(int)

        return final_predictions

    def predict_proba(self, X):
        forward_probas = self.forward_classifier.predict_proba(X)
        backward_probas = self.backward_classifier.predict_proba(X)

        # Example of weighted averaging - weights can be adjusted
        forward_weight = 0.5
        backward_weight = 0.5
        final_probas = forward_probas * forward_weight + backward_probas * backward_weight
        return final_probas
#######################################################################
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

class MultiClassSVM(BaseEstimator, ClassifierMixin):
    """
    A unique multi-class classifier built upon the One-Class SVM, designed to handle multi-class
    problems by training a separate One-Class SVM for each class. This approach treats
    each class as an independent binary classification problem, distinguishing instances of a
    single class from all others. The classifier supports hyperparameter tuning for each One-Class
    SVM using grid search.

    Parameters:
    -----------
    param_grid : dict or list of dictionaries, optional
        Example: 
            param_grid = {
            'svc__C': [0.01, 0.1, 1, 10],
            'svc__max_iter': [1000, 5000]
            }

    scaling : default 'minmax'
        MinMaxScaler() is the default. If you want to use StandardScaler, specify "std".
        If you want to use RobustScaler, specify scaling="robust".
        If scaling=None or scaling=False, then no Scaler will be used.

    Attributes:
    -----------
    models : dict
        A dictionary where keys are class indices and values are the trained One-Class SVM
        models corresponding to each class.

    label_binarizer : LabelBinarizer object
        Used to binarize the class labels and later invert the binary predictions back to
        multi-class labels.

    Methods:
    --------
    fit(X, y):
        Fit the MultiClassSVM to the training data by training a separate One-Class SVM for
        each class with optional hyperparameter tuning.

    predict(X):
        Predict class labels for samples in `X` by evaluating each sample with all One-Class
        SVM models and assigning the class label based on the combined predictions.

    predict_proba(X):
        Estimate class probabilities for samples in `X` by normalizing the decision function
        scores from each One-Class SVM model.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)
    >>> clf = MultiClassSVM(param_grid=param_grid, scaling="minmax")
    >>> clf.fit(X, y)
    >>> print(clf.predict(X[:5]))

    Note:
    -----
    The effectiveness of this classifier relies on the suitability of the One-Class SVM for
    the given data and the chosen hyperparameters. Fine-tuning the `param_grid` and interpreting
    the probabilistic outputs can be crucial for achieving optimal performance.
    """
    def __init__(self, param_grid=None, scaling="minmax"):
        if scaling:
            if scaling == "std":
                self.scaler = StandardScaler()
            elif scaling == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaling == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = None
        # Define the parameter grid for the pipeline
        self.param_grid = param_grid if param_grid is not None else {
            'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
            'svc__max_iter': [1000, 5000, 10000]
        }

        self.models = {}
        self.label_binarizer = LabelBinarizer()

    def fit(self, X, y):
        self.label_binarizer.fit(y)
        y_bin = self.label_binarizer.transform(y)

        if y_bin.ndim == 1:
            y_bin = y_bin[:, np.newaxis]

        classes = self.label_binarizer.classes_

        # Define the pipeline
        if self.scaler:
            pipeline = Pipeline([
                ('scaler', self.scaler),
                ('svc', LinearSVC(dual=False, penalty='l2', loss='squared_hinge', random_state=42))
            ])
            scaler = self.scaler.fit(X)
        else:
            #### No scaling will be done #######
            pipeline = Pipeline([
                ('svc', LinearSVC(dual=False, penalty='l2', loss='squared_hinge', random_state=42))
            ])
            scaler = None

        ### Now perform model pipeline training ####################
        if len(classes) == 2 and y_bin.shape[1] == 1:  # Binary classification
            # Perform grid search on the pipeline
            grid_search = GridSearchCV(pipeline, self.param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X, y_bin.ravel())

            # Retrieve the best pipeline from grid search
            best_pipeline = grid_search.best_estimator_

            # Calibrate the best model from the pipeline
            calibrated_svc = CalibratedClassifierCV(best_pipeline.named_steps['svc'], cv='prefit', method='sigmoid')
            if self.scaler:
                calibrated_svc.fit(best_pipeline.named_steps['scaler'].transform(X), y_bin.ravel())
            else:
                calibrated_svc.fit(X, y_bin.ravel())

            # Store the calibrated_svc for both classes
            if self.scaler:
                self.models[0] = (best_pipeline.named_steps['scaler'], calibrated_svc)  # Negative class
                self.models[1] = (best_pipeline.named_steps['scaler'], calibrated_svc)  # Positive class
            else:
                self.models[0] = (None, calibrated_svc)  # Negative class
                self.models[1] = (None, calibrated_svc)  # Positive class

        else:  # Multiclass classification
            for i, class_label in enumerate(classes):
                # Perform grid search on the pipeline
                grid_search = GridSearchCV(pipeline, self.param_grid, cv=5, scoring='balanced_accuracy')
                grid_search.fit(X, y_bin[:, i])

                # Retrieve the best pipeline from grid search
                best_pipeline = grid_search.best_estimator_

                # Calibrate the best model from the pipeline
                calibrated_svc = CalibratedClassifierCV(best_pipeline.named_steps['svc'], cv='prefit', method='sigmoid')
                if self.scaler:
                    calibrated_svc.fit(best_pipeline.named_steps['scaler'].transform(X), y_bin[:, i])
                else:
                    calibrated_svc.fit(X, y_bin[:, i])

                # Store the calibrated_svc for the current class
                if self.scaler:
                    self.models[class_label] = (best_pipeline.named_steps['scaler'], calibrated_svc)
                else:
                    self.models[class_label] = (None, calibrated_svc)

        return self

    def predict_proba(self, X):
        class_probabilities = np.zeros((X.shape[0], len(self.models)))

        for i, model in self.models.items():
            if self.scaler:
                scaler, calibrated_svc = model
                X_scaled = scaler.transform(X)
            else:
                _, calibrated_svc = model
                X_scaled = copy.deepcopy(X)

            probabilities = calibrated_svc.predict_proba(X_scaled)[:, 1]
            class_probabilities[:, i] = probabilities

        # Normalize the scores to obtain probabilities
        y_proba = (class_probabilities - class_probabilities.min(axis=1)[:, np.newaxis]) / (class_probabilities.max(axis=1)[:, np.newaxis] - class_probabilities.min(axis=1)[:, np.newaxis])
        sums = np.expand_dims(np.sum(y_proba,axis=1), axis=1)
        arr = np.hstack([y_proba, sums])
        result = arr / arr[:, -1:]
        class_probabilities = result[:,:-1]
        return class_probabilities

    def predict(self, X):
        # Get the probability estimates for all classes
        class_probabilities = self.predict_proba(X)
        
        # For binary classification, class_probabilities will have 2 columns, and we're interested in the second one
        if len(self.models) == 2 and class_probabilities.shape[1] == 2:
            predicted_class_indices = class_probabilities[:, 1] >= 0.5
            predicted_classes = predicted_class_indices.astype(int)
        else:  # For multiclass, pick the class with the highest probability
            predicted_class_indices = np.argmax(class_probabilities, axis=1)
            # Convert class indices back to original class labels - Don't do this since it doesn't work!
            #predicted_classes = self.label_binarizer.inverse_transform(predicted_class_indices.reshape(-1, 1))
            predicted_classes = predicted_class_indices[:]

        return predicted_classes

    
####################################################################################


