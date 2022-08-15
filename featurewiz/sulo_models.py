import numpy as np
import pandas as pd
np.random.seed(99)
################################################################################
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
################################################################################
import math
from collections import Counter
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
import time
from sklearn.cluster import KMeans
from matplotlib.pyplot import figure
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import VotingRegressor, VotingClassifier
import copy
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.base import ClassifierMixin, RegressorMixin
from imblearn.over_sampling import SMOTENC, ADASYN
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek 
import lightgbm
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.multioutput import ClassifierChain, RegressorChain
import scipy as sp
import pdb
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from .featurewiz import get_class_distribution

class SuloClassifier(BaseEstimator, ClassifierMixin):
    """
    SuloClassifier works really fast and very well for all kinds of datasets.
    It works on small as well as big data. It works in multi-class as well as multi-labels.
    It works on regular balanced data as well as imbalanced data sets.
    The reason it works so well is that it is an ensemble of highly tuned models.
    You don't have to send any inputs but if you wanted to, you can send in two inputs:
    n_estimators: number of models you want in the final ensemble.
    base_estimator: base model you want to train in each of the ensembles.
    If you want, you can igore both these inputs and it will automatically choose these.
    It is fully compatible with scikit-learn pipelines and other models.
    """
    def __init__(self, base_estimator=None, n_estimators=None, pipeline=True, weights=False, 
                                       imbalanced=False, verbose=0):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.pipeline = pipeline
        self.weights = weights
        self.imbalanced = imbalanced
        self.verbose = verbose
        self.models = []
        self.multi_label =  False
        self.max_number_of_classes = 1
        self.scores = []
        self.classes = []
        self.regression_min_max = []
        self.model_name = ''
        self.features = []

    def fit(self, X, y):
        X = copy.deepcopy(X)
        print('Input data shapes: X = %s, y = %s' %(X.shape, y.shape,))
        seed = 42
        shuffleFlag = True
        modeltype = 'Classification'
        features_limit = 50 ## if there are more than 50 features in dataset, better to use LGBM ##
        start = time.time()
        if isinstance(X, pd.DataFrame):
            self.features = X.columns.tolist()
        else:
            print('Cannot operate SuloClassifier on numpy arrays. Must be dataframes. Returning...')
            return self
        # Use KFold for understanding the performance
        if self.weights:
            print('Remember that using class weights will wrongly skew predict_probas from any classifier')
        if self.imbalanced:
            class_weights = None
        else:
            class_weights = get_class_weights(y, verbose=0)
        ### Remember that putting class weights will totally destroy predict_probas ###
        self.classes = print_flatten_dict(class_weights)
        scale_pos_weight = get_scale_pos_weight(y)
        #print('Class weights = %s' %class_weights)
        gpu_exists = check_if_GPU_exists()
        if gpu_exists:
            device="gpu"
        else:
            device="cpu"
        ## Don't change this since it gives an error ##
        metric  = 'auc'
        ### don't change this metric and eval metric - it gives error if you change it ##
        eval_metric = 'auc'
        row_limit = 10000
        if self.imbalanced:
            print('    Imbalanced classes of y = %s' %find_rare_class(y, verbose=self.verbose))
        ################          P I P E L I N E        ##########################
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean", add_indicator=True)), ("scaler", StandardScaler())]
        )

        categorical_transformer_low = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True)),
                ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

        categorical_transformer_high = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True)),
                ("encoding", LabelEncoder()),
            ]
        )

        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_cardinality(X, categorical_features)
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        ####################################################################################
        if isinstance(y, pd.DataFrame):
            if len(y.columns) >= 2:
                number_of_classes = num_classes(y)
                for each_i in y.columns:
                    number_of_classes[each_i] = int(number_of_classes[each_i] - 1)
                max_number_of_classes = np.max(list(number_of_classes.values()))
            else:
                number_of_classes = int(num_classes(y) - 1)
                max_number_of_classes = np.max(number_of_classes)
        else:
            number_of_classes = int(num_classes(y) - 1)
            max_number_of_classes = np.max(number_of_classes)
        data_samples = X.shape[0]
        self.max_number_of_classes = max_number_of_classes
        if self.n_estimators is None:
            if data_samples <= row_limit:
                self.n_estimators = min(5, int(1.5*np.log10(data_samples)))
            else:
                self.n_estimators = min(10, int(1.5*np.log10(data_samples)))
        self.model_name = 'lgb'
        num_splits = self.n_estimators
        num_repeats = 2
        kfold = RepeatedKFold(n_splits=num_splits, random_state=seed, n_repeats=num_repeats)
        num_iterations = int(num_splits * num_repeats)
        scoring = 'balanced_accuracy'
        print('    Number of estimators used in SuloClassifier = %s' %num_iterations)
        ##### This is where we check if y is single label or multi-label ##
        if isinstance(y, pd.DataFrame):
            ###############################################################
            ### This is for Multi-Label problems only #####################
            ###############################################################
            targets = y.columns.tolist()
            if is_y_object(y):
                print('Cannot perform classification using object or string targets. Please convert to numeric and try again.')
                return self
            if len(targets) > 1:
                self.multi_label = y.columns.tolist()
                ### You need to initialize the class before each run - otherwise, error!
                if self.base_estimator is None:
                    if self.max_number_of_classes <= 1:
                        ##############################################################
                        ###   This is for Binary Classification problems only ########
                        ##############################################################
                        if self.imbalanced:
                            try:
                                from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                                self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=99)
                            except:
                                print('pip install imbalanced_ensemble and re-run this again.')
                                return self
                            self.model_name = 'other'
                        else:
                            ### make it a regular dictionary with weights for pos and neg classes ##
                            class_weights = dict([v for k,v in class_weights.items()][0])
                            if data_samples <= row_limit:
                                if (X.dtypes==float).all() and len(self.features) <= features_limit:
                                    if self.verbose:
                                        print('    Selecting Label Propagation since it will work great for this dataset...')
                                        print('        however it will skew probabilities and show lower ROC AUC score than normal.')
                                    self.base_estimator =  LabelPropagation()
                                    self.model_name = 'lp'
                                else:
                                    if len(self.features) <= features_limit:
                                        if self.verbose:
                                            print('    Selecting Bagging Classifier for this dataset...')
                                        self.base_estimator = BaggingClassifier(n_estimators=150)
                                        self.model_name = 'bg'
                                    else:
                                        if self.verbose:
                                            print('    Selecting LGBM Regressor as base estimator...')
                                        self.base_estimator = LGBMClassifier(device=device, random_state=99,
                                                           class_weight=class_weights,
                                                            )
                            else:
                                ### This is for large datasets in Binary classes ###########
                                if self.verbose:
                                    print('    Selecting LGBM Regressor as base estimator...')
                                if gpu_exists:
                                    self.base_estimator = XGBClassifier(n_estimators=250, 
                                        n_jobs=-1,tree_method = 'gpu_hist',gpu_id=0, predictor="gpu_predictor")
                                else:
                                    self.base_estimator = LGBMClassifier(device=device, random_state=99, n_jobs=-1,
                                                        #is_unbalance=True, 
                                                        #max_depth=10, metric=metric,
                                                        #num_class=self.max_number_of_classes,
                                                        #n_estimators=100,  num_leaves=84, 
                                                        #objective='binary',
                                                        #boosting_type ='goss', 
                                                        #scale_pos_weight=scale_pos_weight,
                                                       class_weight=class_weights,
                                                        )
                    else:
                        #############################################################
                        ###   This is for Multi Classification problems only ########
                        ### Make sure you don't put any class weights here since it won't work in multi-labels ##
                        ##############################################################
                        if self.imbalanced:
                            if self.verbose:
                                print('    Selecting Self Paced ensemble classifier since imbalanced flag is set...')
                                try:
                                    from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                                    self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=99)
                                except:
                                    print('pip install imbalanced_ensemble and re-run this again.')
                                    return self
                            self.model_name = 'other'
                        else:
                            if data_samples <= row_limit:
                                if len(self.features) <= features_limit:
                                    if self.verbose:
                                        print('    Selecting Extra Trees Classifier for small datasets...')
                                    self.base_estimator = ExtraTreesClassifier(n_estimators=200, random_state=99)
                                    self.model_name = 'rf'
                                else:
                                    self.base_estimator = LGBMRegressor(device=device, random_state=99)                                    
                            else:
                                if (X.dtypes==float).all() and len(self.features) <= features_limit:
                                    print('    Selecting Label Propagation since it works great for multiclass problems...')
                                    print('        however it will skew probabilities a little so be aware of this')
                                    self.base_estimator =  LabelPropagation()
                                    self.model_name = 'lp'
                                else:
                                    self.base_estimator = LGBMClassifier(device=device, random_state=99, n_jobs=-1,)
                else:
                    self.model_name == 'other'
                ### Remember we don't to HPT Tuning for Multi-label problems since it errors ####
                if self.verbose and self.model_name=='lgb':
                    print('    Selecting LGBM Classifier as base estimator...')
                for i, (train_index, test_index) in enumerate(kfold.split(X)):
                    start_time = time.time()
                    # Split data into train and test based on folds          
                    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]                
                    else:
                        y_train, y_test = y[train_index], y[test_index]

                    if isinstance(X, pd.DataFrame):
                        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
                    else:
                        x_train, x_test = X[train_index], X[test_index]

                    ###### Do this only the first time ################################################
                    if i == 0:
                        ### It does not make sense to do hyper-param tuning for multi-label models ##
                        ###    since ClassifierChains do not have many hyper params #################
                        #self.base_estimator = rand_search(self.base_estimator, x_train, y_train, 
                        #                        self.model_name, verbose=self.verbose)
                        #print('    hyper tuned base estimator = %s' %self.base_estimator)
                        if self.max_number_of_classes <= 1:
                            est_list = [ClassifierChain(self.base_estimator, order=None, cv=3, random_state=i) 
                                        for i in range(num_iterations)] 
                            if self.verbose:
                                print('Fitting a %s for %s targets with MultiOutputClassifier. This will take time...' %(
                                            str(self.base_estimator).split("(")[0],y.shape[1]))
                        else:
                            if self.imbalanced:
                                if self.verbose:
                                    print('    Training with ClassifierChain since multi_label dataset. This will take time...')
                                est_list = [ClassifierChain(self.base_estimator, order=None, random_state=i)
                                            for i in range(num_iterations)] 
                            else:
                                ### You must use multioutputclassifier since it is the only one predicts probas correctly ##
                                est_list = [MultiOutputClassifier(self.base_estimator)#, order="random", random_state=i) 
                                            for i in range(num_iterations)] 
                                if self.verbose:
                                    print('Training a %s for %s targets with MultiOutputClassifier. This will take time...' %(
                                                str(self.base_estimator).split("(")[0],y.shape[1]))

                    # Initialize model with your supervised algorithm of choice
                    model = est_list[i]

                    # Train model and use it to train on the fold
                    if self.pipeline:
                        ### This is only with a pipeline ########
                        pipe = Pipeline(
                            steps=[("preprocessor", preprocessor), ("model", model)]
                        )

                        pipe.fit(x_train, y_train)
                        self.models.append(pipe)

                        # Predict on remaining data of each fold
                        preds = pipe.predict(x_test)

                    else:
                        #### This is without a pipeline ###
                        model.fit(x_train, y_train)
                        self.models.append(model)

                        # Predict on remaining data of each fold
                        preds = model.predict(x_test)


                    # Use best classification metric to measure performance of model
                    if self.imbalanced:
                        ### Use Regression predictions and convert them into classes here ##
                        score = print_sulo_accuracy(y_test, preds, y_probas="", verbose=self.verbose)
                        print("    Fold %s: Average OOF Score (higher is better): %0.3f" %(i+1, score))
                    else:
                        score = print_accuracy(targets, y_test, preds, verbose=self.verbose)
                        print("    Fold %s: Average OOF Score: %0.0f%%" %(i+1, 100*score))
                    self.scores.append(score)
                    
                    # Finally, check out the total time taken
                    end_time = time.time()
                    timeTaken = end_time - start_time
                    print("Time Taken for fold %s: %0.0f (seconds)" %(i+1, timeTaken))

                # Compute average score
                averageAccuracy = sum(self.scores)/len(self.scores)
                if self.verbose:
                    if self.imbalanced:
                        print("Average Balanced Accuracy score of %s-model SuloClassifier: %0.3f" %(
                                        self.n_estimators, averageAccuracy))
                    else:                        
                        print("Average Balanced Accuracy of %s-model SuloClassifier: %0.0f%%" %(
                                        self.n_estimators, 100*averageAccuracy))
                end = time.time()
                timeTaken = end - start
                print("Time Taken overall: %0.0f (seconds)" %(timeTaken))
                return self
        ########################################################
        #####  This is for Single Label Classification problems 
        ########################################################
        
        if isinstance(y, pd.Series):
            targets = y.name
            if targets is None:
                targets = []
        else:
            targets = []
        if self.base_estimator is None:
            if data_samples <= row_limit:
                ### For small datasets use RFC for Binary Class   ########################
                if number_of_classes <= 1:
                    if self.imbalanced:
                        if self.verbose:
                            print('    Selecting Self Paced ensemble classifier as base estimator...')
                            try:
                                from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                                self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=99)
                            except:
                                print('pip install imbalanced_ensemble and re-run this again.')
                        self.model_name = 'other'
                    else:
                        ### For binary-class problems use RandomForest or the faster ET Classifier ######
                        if (X.dtypes==float).all() and len(self.features) <= features_limit:
                            print('    Selecting Label Propagation since it will work great for this dataset...')
                            print('        however it will skew probabilities and show lower ROC AUC score than normal.')
                            self.base_estimator =  LabelPropagation()
                            self.model_name = 'lp'
                        else:
                            if len(self.features) <= features_limit:
                                if self.verbose:
                                    print('    Selecting Bagging Classifier for this dataset...')
                                ### The Bagging classifier outperforms ETC most of the time ####
                                self.base_estimator = BaggingClassifier(n_estimators=20)
                                self.model_name = 'bg'
                            else:
                                if self.verbose:
                                    print('    Selecting LGBM Classifier as base estimator...')
                                self.base_estimator = LGBMClassifier(device=device, random_state=99, n_jobs=-1,
                                                        scale_pos_weight=scale_pos_weight,)
                else:
                    ### For Multi-class datasets you can use Regressors for numeric classes ####################
                    if self.imbalanced:
                        if self.verbose:
                            print('    Selecting Self Paced ensemble classifier as base estimator...')
                        try:
                            from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                            self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=99)
                        except:
                            print('pip install imbalanced_ensemble and re-run this again.')
                            return self
                        self.model_name = 'other'
                    else:
                        ### For multi-class problems use Label Propagation which is faster and better ##
                        if (X.dtypes==float).all() and len(self.features) <= features_limit:
                                print('    Selecting Label Propagation since it will work great for this dataset...')
                                print('        however it will skew probabilities and show lower ROC AUC score than normal.')
                                self.base_estimator =  LabelPropagation()
                                self.model_name = 'lp'
                        else:
                            if len(self.features) <= features_limit:
                                if self.verbose:
                                    print('    Selecting Bagging Classifier for this dataset...')
                                self.base_estimator = BaggingClassifier(n_estimators=20)
                                self.model_name = 'bg'
                            else:
                                if self.verbose:
                                    print('    Selecting LGBM Classifier as base estimator...')
                                self.base_estimator = LGBMClassifier(device=device, random_state=99, n_jobs=-1,
                                                #is_unbalance=False,
                                                #learning_rate=0.3,
                                                #max_depth=10,
                                                metric='multi_logloss',
                                                #n_estimators=130, num_leaves=84,
                                                num_class=number_of_classes, objective='multiclass',
                                                #boosting_type ='goss', 
                                                #scale_pos_weight=None,
                                                class_weight=class_weights
                                                )
            else:
                ### For large datasets use LGBM or Regressors as well ########################
                if number_of_classes <= 1:
                    if self.imbalanced:
                        if self.verbose:
                            print('    Selecting Self Paced ensemble classifier as base estimator...')
                        try:
                            from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                            self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=99)
                        except:
                            print('pip install imbalanced_ensemble and re-run this again.')
                            return self
                        self.model_name = 'other'
                    else:
                        if self.verbose:
                            print('    Selecting LGBM Classifier for this dataset...')
                        #self.base_estimator = LGBMClassifier(n_estimators=250, random_state=99, 
                        #            boosting_type ='goss', scale_pos_weight=scale_pos_weight)
                        self.base_estimator = LGBMClassifier(device=device, random_state=99, n_jobs=-1,
                                                #is_unbalance=True,
                                                #learning_rate=0.3, 
                                                #max_depth=10, 
                                                #metric=metric,
                                                #n_estimators=230, num_leaves=84, 
                                                #num_class=number_of_classes,
                                                #objective='binary',
                                                #boosting_type ='goss', 
                                                scale_pos_weight=scale_pos_weight
                                                )
                else:
                    ### For Multi-class datasets you can use Regressors for numeric classes ####################
                    if self.imbalanced:
                        if self.verbose:
                            print('    Selecting Self Paced ensemble classifier as base estimator...')
                        try:
                            from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
                            self.base_estimator = SelfPacedEnsembleClassifier(n_jobs=-1, random_state=99)
                        except:
                            print('pip install imbalanced_ensemble and re-run this again.')
                            return self
                        self.model_name = 'other'
                    else:
                        #if self.weights:
                        #    class_weights = None
                        if self.verbose:
                            print('    Selecting LGBM Classifier as base estimator...')
                        self.base_estimator = LGBMClassifier(device=device, random_state=99, n_jobs=-1,
                                                #is_unbalance=False, learning_rate=0.3,
                                                #max_depth=10, 
                                                metric='multi_logloss',
                                                #n_estimators=230, num_leaves=84,
                                                num_class=number_of_classes, objective='multiclass',
                                                #boosting_type ='goss', 
                                                scale_pos_weight=None,
                                                class_weight=class_weights
                                                )
        else:
            self.model_name = 'other'

        est_list = num_iterations*[self.base_estimator]
        
        # Perform CV
        for i, (train_index, test_index) in enumerate(kfold.split(X)):
            # Split data into train and test based on folds          
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]                
            else:
                y_train, y_test = y[train_index], y[test_index]

            if isinstance(X, pd.DataFrame):
                x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            else:
                x_train, x_test = X[train_index], X[test_index]

            
            ##   small datasets processing #####
            if i == 0:
                if self.pipeline:
                    # Train model and use it in a pipeline to train on the fold  ##
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("model", self.base_estimator)])
                    if self.model_name == 'other':
                        print('No HPT tuning performed since base estimator is given by input...')
                        self.base_estimator = copy.deepcopy(pipe)
                    else:
                        if len(self.features) <= features_limit:
                            self.base_estimator = rand_search(pipe, x_train, y_train, 
                                                    self.model_name, self.pipeline, scoring, verbose=self.verbose)
                        else:
                            print('No HPT tuning performed since number of features is too large...')
                            self.base_estimator = copy.deepcopy(pipe)
                else:
                    ### This is for without a pipeline #######
                    if self.model_name == 'other':
                        ### leave the base estimator as is ###
                        print('No HPT tuning performed since base estimator is given by input...')
                    else:
                        if len(self.features) <= features_limit:
                            ### leave the base estimator as is ###
                            self.base_estimator = rand_search(self.base_estimator, x_train, 
                                                y_train, self.model_name, self.pipeline, scoring, verbose=self.verbose)
                        else:
                            print('No HPT tuning performed since number of features is too large...')

                est_list = num_iterations*[self.base_estimator]
                #print('    base estimator = %s' %self.base_estimator)
            
            # Initialize model with your supervised algorithm of choice
            model = est_list[i]
            
            model.fit(x_train, y_train)
            self.models.append(model)

            # Predict on remaining data of each fold
            preds = model.predict(x_test)

            # Use best classification metric to measure performance of model

            if self.imbalanced:
                ### Use Regression predictions and convert them into classes here ##
                score = print_sulo_accuracy(y_test, preds, y_probas="", verbose=self.verbose)
                print("    Fold %s: Average OOF Score (higher is better): %0.3f" %(i+1, score))
            else:
                #score = balanced_accuracy_score(y_test, preds)
                score = print_accuracy(targets, y_test, preds, verbose=self.verbose)
                print("    Fold %s: OOF Score: %0.0f%%" %(i+1, 100*score))
            self.scores.append(score)

        # Compute average score
        averageAccuracy = sum(self.scores)/len(self.scores)
        if self.verbose:
            if self.imbalanced:
                print("Average Balanced Accuracy of %s-model SuloClassifier: %0.3f" %(num_iterations, 100*averageAccuracy))
            else:
                print("Average Balanced Accuracy of %s-model SuloClassifier: %0.0f%%" %(num_iterations, 100*averageAccuracy))

        # Finally, check out the total time taken
        end = time.time()
        timeTaken = end-start
        print("Time Taken: %0.0f (seconds)" %timeTaken)
        return self

    def predict(self, X):
        from scipy import stats
        weights = self.scores
        if self.multi_label:
            ### In multi-label, targets have to be numeric, so you can leave weights as-is ##
            ypre = np.array([model.predict(X) for model in self.models ])
            y_predis = np.average(ypre, axis=0, weights=weights)
            y_preds = np.round(y_predis,0).astype(int)
            return y_preds
        y_predis = np.column_stack([model.predict(X) for model in self.models ])
        ### This weights the model's predictions according to OOB scores obtained
        #### In single label, targets can be object or string, so weights cannot be applied always ##
        if y_predis.dtype == object or y_predis.dtype == bool:
            ### in the case of predictions that are strings, no need for weights ##
            y_predis = stats.mode(y_predis, axis=1)[0].ravel()
        else:
            if str(y_predis.dtype) == 'category':
                y_predis = stats.mode(y_predis, axis=1)[0].ravel()
            else:
                y_predis = np.average(y_predis, weights=weights, axis=1)
                y_predis = np.round(y_predis,0).astype(int)
        if self.imbalanced:
            y_predis = copy.deepcopy(y_predis)
        return y_predis
    
    def predict_proba(self, X):
        weights = self.scores
        y_probas = [model.predict_proba(X) for model in self.models ]
        y_probas = return_predict_proba(y_probas)
        return y_probas

    def print_pipeline(self):
        from sklearn import set_config
        set_config(display="text")
        return self.modelformer

    def plot_pipeline(self):
        from sklearn import set_config
        set_config(display="diagram")
        return self

    def plot_importance(self, max_features=10):
        import lightgbm as lgbm
        from xgboost import plot_importance
        model_name = self.model_name
        feature_names = self.features
        if self.multi_label:
            print('No feature importances available for multi_label problems')
            return
        if  model_name == 'lgb' or model_name == 'xgb':
            for i, model in enumerate(self.models):
                if self.pipeline:
                    model_object = model.named_steps['model']
                else:
                    model_object = model
                feature_importances = model_object.booster_.feature_importance(importance_type='gain')
                if i == 0:
                    feature_imp = pd.DataFrame({'Value':feature_importances,'Feature':feature_names})
                else:
                    feature_imp = pd.concat([feature_imp, 
                        pd.DataFrame({'Value':feature_importances,'Feature':feature_names})], axis=0)
                #lgbm.plot_importance(model_object, importance_type='gain', max_num_features=max_features)
            feature_imp = feature_imp.groupby('Feature').mean().sort_values('Value',ascending=False).reset_index()
            feature_imp.set_index('Feature')[:max_features].plot(kind='barh', title='Top 10 Features')
            ### This is for XGB ###
            #plot_importance(self.model.named_steps['model'], importance_type='gain', max_num_features=max_features)
        elif model_name == 'lp':
            print('No feature importances available for LabelPropagation algorithm. Returning...')
            return
        elif model_name == 'rf':
            ### These are for RandomForestClassifier kind of scikit-learn models ###
            try:
                for i, model in enumerate(self.models):
                    if self.pipeline:
                        model_object = model.named_steps['model']
                    else:
                        model_object = model
                    feature_importances = model_object.feature_importances_
                    if i == 0:
                        feature_imp = pd.DataFrame({'Value':feature_importances,'Feature':feature_names})
                    else:
                        feature_imp = pd.concat([feature_imp, 
                            pd.DataFrame({'Value':feature_importances,'Feature':feature_names})], axis=0)
                feature_imp = feature_imp.groupby('Feature').mean().sort_values('Value',ascending=False).reset_index()
                feature_imp.set_index('Feature')[:max_features].plot(kind='barh', title='Top 10 Features')
            except:
                    print('Could not plot feature importances. Please check your model and try again.')                
        else:
            print('No feature importances available for this algorithm. Returning...')
            return
#####################################################################################################
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
def rand_search(model, X, y, model_name, pipe_flag=False, scoring=None, verbose=0):
    start = time.time()
    if pipe_flag:
        model_string = 'model__'
    else:
        model_string = ''
    ### set n_iter here ####
    n_iter = 3
    if model_name == 'rf':
        #criterion = ["gini", "entropy", "log_loss"]
        # Number of trees in random forest
        n_estimators = sp_randInt(100, 300)
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt', 'log']
        # Maximum number of levels in tree
        max_depth = sp_randInt(2, 10)
        # Minimum number of samples required to split a node
        min_samples_split = sp_randInt(2, 10)
        # Minimum number of samples required at each leaf node
        min_samples_leaf = sp_randInt(2, 10)
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        ###  These are the RandomForest params ########        
        params = {
            #model_string+'criterion': criterion,
            model_string+'n_estimators': n_estimators,
            #model_string+'max_features': max_features,
            #model_string+'max_depth': max_depth,
            #model_string+'min_samples_split': min_samples_split,
            #model_string+'min_samples_leaf': min_samples_leaf,
           #model_string+'bootstrap': bootstrap,
                       }
    elif model_name == 'bg':
        criterion = ["gini", "entropy", "log_loss"]
        # Number of trees in random forest
        n_estimators = sp_randInt(100, 300)
        # Number of features to consider at every split
        #max_features = ['auto', 'sqrt', 'log']
        max_features = sp_randFloat(0.3,0.9)
        # Maximum number of levels in tree
        max_depth = sp_randInt(2, 10)
        # Minimum number of samples required to split a node
        min_samples_split = sp_randInt(2, 10)
        # Minimum number of samples required at each leaf node
        min_samples_leaf = sp_randInt(2, 10)
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        ###  These are the RandomForest params ########
        params = {
            #model_string+'criterion': criterion,
            model_string+'n_estimators': n_estimators,
            #model_string+'max_features': max_features,
            #model_string+'max_depth': max_depth,
            #model_string+'min_samples_split': min_samples_split,
            #model_string+'min_samples_leaf': min_samples_leaf,
            #model_string+'bootstrap': bootstrap,
            #model_string+'bootstrap_features': bootstrap,
                       }
    elif model_name == 'lgb':
        # Number of estimators in LGBM Classifier ##
        n_estimators = sp_randInt(100, 500)
        ### number of leaves is only for LGBM ###
        num_leaves = sp_randInt(5, 300)
        ## learning rate is very important for LGBM ##
        learning_rate = sp.stats.uniform(scale=1)
        params = {
            model_string+'n_estimators': n_estimators,
            #model_string+'num_leaves': num_leaves,
            model_string+'learning_rate': learning_rate,
                    }
    elif model_name == 'lp':
        params =  {
            ### Don't overly complicate this simple model. It works best with no tuning!
            model_string+'gamma': sp_randInt(0, 32),
            model_string+'kernel': ['knn', 'rbf'],
            #model_string+'max_iter': sp_randInt(50, 500),
            #model_string+'n_neighbors': sp_randInt(2, 5),
                }
    else:
        ### Since we don't know what model will be sent, we cannot tune it ##
        params = {}
        return model
    ### You must leave Random State as None since shuffle is False. Otherwise, error! 
    kfold = KFold(n_splits=5, random_state=None, shuffle=False)
    if verbose:
        print("Finding best params for base estimator using RandomizedSearchCV...")
    clf = RandomizedSearchCV(model, params, n_iter=n_iter, scoring=scoring,
                         cv = kfold, n_jobs=-1, random_state=100)
    
    clf.fit(X, y)

    if verbose:
        print("    best score is :" , clf.best_score_)
        #print("    best estimator is :" , clf.best_estimator_)
        print("    best Params is :" , clf.best_params_)
        print("Time Taken for RandomizedSearchCV: %0.0f (seconds)" %(time.time()-start))
    return clf.best_estimator_
##################################################################################
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
##################################################################################
from collections import OrderedDict    
def get_class_weights(y_input, verbose=0):    
    y_input = copy.deepcopy(y_input)
    if isinstance(y_input, np.ndarray):
        y_input = pd.Series(y_input)
    elif isinstance(y_input, pd.Series):
        pass
    elif isinstance(y_input, pd.DataFrame):
        if len(y_input.columns) >= 2:
            ### if it is a dataframe, return only if it is one column dataframe ##
            class_weights = dict()
            for each_target in y_input.columns:
                class_weights[each_target] = get_class_weights(y_input[each_target])
            return class_weights
        else:
            y_input = y_input.values.reshape(-1)
    else:
        ### if you cannot detect the type or if it is a multi-column dataframe, ignore it
        return None
    classes = np.unique(y_input)
    rare_class = find_rare_class(y_input)
    xp = Counter(y_input)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_input)
    class_weights = OrderedDict(zip(classes, np.round(class_weights/class_weights.min()).astype(int)))
    if verbose:
        print('Class weights used in classifier are: %s' %class_weights)
    return class_weights

from collections import OrderedDict
def get_scale_pos_weight(y_input, verbose=0):
    class_weighted_rows = get_class_weights(y_input)
    if isinstance(y_input, np.ndarray):
        y_input = pd.Series(y_input)
    elif isinstance(y_input, pd.Series):
        pass
    elif isinstance(y_input, pd.DataFrame):
        if len(y_input.columns) >= 2:
            ### if it is a dataframe, return only if it is one column dataframe ##
            rare_class_weights = OrderedDict()
            for each_target in y_input.columns:
                rare_class_weights[each_target] = get_scale_pos_weight(y_input[each_target])
            return rare_class_weights
        else:
            y_input = y_input.values.reshape(-1)
    
    rare_class = find_rare_class(y_input)
    rare_class_weight = class_weighted_rows[rare_class]
    if verbose:
        print('    For class %s, weight = %s' %(rare_class, rare_class_weight))
    return rare_class_weight
##########################################################################
from collections import defaultdict
from collections import OrderedDict
def return_minority_samples(y, verbose=0):
    """
    #### Calculates the % count of each class in y and returns a 
    #### smaller set of y based on being 5% or less of dataset.
    It returns the small y as an array or dataframe as input y was.
    """
    import copy
    y = copy.deepcopy(y)
    if isinstance(y, np.ndarray):
        ls = pd.Series(y).value_counts()[(pd.Series(y).value_counts(1)<=0.05).values].index
        return y[pd.Series(y).isin(ls).values]
    else:
        if isinstance(y, pd.Series):
            ls = y.value_counts()[(y.value_counts(1)<=0.05).values].index
        else:
            y = y.iloc[:,0]
            ls = y.value_counts()[(y.value_counts(1)<=0.05).values].index
        return y[y.isin(ls)]

def num_classes(y, verbose=0):
    """
    ### Returns number of classes in y
    """
    import copy
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

def return_minority_classes(y, verbose=0):
    """
    #### Calculates the % count of each class in y and returns a 
    #### smaller set of y based on being 5% or less of dataset.
    It returns the list of classes that are <=5% classes.
    """
    import copy
    y = copy.deepcopy(y)
    if isinstance(y, np.ndarray):
        ls = pd.Series(y).value_counts()[(pd.Series(y).value_counts(1)<=0.05).values].index
    else:
        if isinstance(y, pd.Series):
            ls = y.value_counts()[(y.value_counts(1)<=0.05).values].index
        else:
            y = y.iloc[:,0]
            ls = y.value_counts()[(y.value_counts(1)<=0.05).values].index
    return ls
#################################################################################
def get_cardinality(X, cat_features):
    ## pick a limit for cardinal variables here ##
    cat_limit = 30
    mask = X[cat_features].nunique() > cat_limit
    high_cardinal_vars = cat_features[mask]
    low_cardinal_vars = cat_features[~mask]
    return low_cardinal_vars, high_cardinal_vars
################################################################################
def is_y_object(y):
    test1 = (y.dtypes.any()==object) | (y.dtypes.any()==bool)
    test2 = str(y.dtypes.any())=='category'
    return test1 | test2

def print_flatten_dict(dd, separator='_', prefix=''):
    ### this function is to flatten dict to print classes and their order ###
    ### One solution here: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    ### I have modified it to make it work for me #################
    return { prefix + separator + str(k) if prefix else k : v
             for kk, vv in dd.items()
             for k, v in print_flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def print_accuracy(target, y_test, y_preds, verbose=0):
    bal_scores = []
    
    from sklearn.metrics import balanced_accuracy_score, classification_report
    if isinstance(target, str): 
        bal_score = balanced_accuracy_score(y_test,y_preds)
        bal_scores.append(bal_score)
        if verbose:
            print('Bal accu %0.0f%%' %(100*bal_score))
            print(classification_report(y_test,y_preds))
    elif len(target) <= 1:
        bal_score = balanced_accuracy_score(y_test,y_preds)
        bal_scores.append(bal_score)
        if verbose:
            print('Bal accu %0.0f%%' %(100*bal_score))
            print(classification_report(y_test,y_preds))
    else:
        for each_i, target_name in enumerate(target):
            bal_score = balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])
            bal_scores.append(bal_score)
            if verbose:
                if each_i == 0:
                    print('For %s:' %target_name)
                print('    Bal accu %0.0f%%' %(100*bal_score))
                print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
    return np.mean(bal_scores)
##########################################################################################
from collections import defaultdict
def return_predict_proba(y_probas):
    ### This is for detecting what-label what-class problems with probas ####
    problemtype = ""
    if isinstance(y_probas, list):
        ### y_probas is a list when y is multi-label. 
        if isinstance(y_probas[0], list):
            ##    1. If y is multi_label but has more than two classes, y_probas[0] is also a list ##
            problemtype = "multi_label_multi_class"
        else:
            initial = y_probas[0].shape[1]
            if np.all([x.shape[1]==initial for x in y_probas]):
                problemtype =  "multi_label_binary_class"
            else:
                problemtype = "multi_label_multi_class"
    else:
        problemtype = "single_label"
    #### This is for making multi-label multi-class predictions into a dictionary ##
    if problemtype == "multi_label_multi_class":
        probas_dict = defaultdict(list)
        ### Initialize the default dict #############
        for each_target in range(len(y_probas[0])):
            probas_dict[each_target] = []
        #### Now that it is is initialized, compile each class into its own list ###
        if isinstance(y_probas[0], list):
            for each_i in range(len(y_probas)):
                for each_j in range(len(y_probas[each_i])):
                    if y_probas[each_i][each_j].shape[1] > 2:
                        probas_dict[each_j].append(y_probas[each_i][each_j])
                    else:
                        probas_dict[each_j].append(y_probas[each_i][each_j][:,1])
            #### Once all of the probas have been put in a dictionary, now compute means ##
            for each_target in range(len(probas_dict)):
                probas_dict[each_target] = np.array(probas_dict[each_target]).mean(axis=0)
    elif problemtype == "multi_label_binary_class":
        initial = y_probas[0].shape[1]
        if np.all([x.shape[1]==initial for x in y_probas]):
            probas_dict = np.array(y_probas).mean(axis=0)
    return probas_dict   
###############################################################################################
from sklearn.metrics import roc_auc_score
import copy
from sklearn.metrics import balanced_accuracy_score, classification_report
import pdb
def print_sulo_accuracy(y_test, y_preds, y_probas='', verbose=0):
    bal_scores = []
    ####### Once you have detected what problem it is, now print its scores #####
    if y_test.ndim <= 1: 
        ### This is a single label problem # we need to test for multiclass ##
        bal_score = balanced_accuracy_score(y_test,y_preds)
        print('Bal accu %0.0f%%' %(100*bal_score))
        if not isinstance(y_probas, str):
            if y_probas.ndim <= 1:
                print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
            else:
                if y_probas.shape[1] == 2:
                    print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
                else:
                    print('Multi-class ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas, multi_class="ovr"))
        bal_scores.append(bal_score)
        if verbose:
            print(classification_report(y_test,y_preds))
    elif y_test.ndim >= 2:
        if y_test.shape[1] == 1:
            bal_score = balanced_accuracy_score(y_test,y_preds)
            bal_scores.append(bal_score)
            print('Bal accu %0.0f%%' %(100*bal_score))
            if not isinstance(y_probas, str):
                if y_probas.shape[1] > 2:
                    print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas, multi_class="ovr"))
                else:
                    print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
            if verbose:
                print(classification_report(y_test,y_preds))
        else:
            if isinstance(y_probas, str):
                ### This is for multi-label problems without probas ####
                for each_i in range(y_test.shape[1]):
                    bal_score = balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])
                    bal_scores.append(bal_score)
                    print('    Bal accu %0.0f%%' %(100*bal_score))
                    if verbose:
                        print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
            else:
                ##### This is only for multi_label_multi_class problems
                num_targets = y_test.shape[1]
                for each_i in range(num_targets):
                    print('    Bal accu %0.0f%%' %(100*balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])))
                    if len(np.unique(y_test.values[:,each_i])) > 2:
                        ### This nan problem happens due to Label Propagation but can be fixed as follows ##
                        mat = y_probas[each_i]
                        if np.any(np.isnan(mat)):
                            mat = pd.DataFrame(mat).fillna(method='ffill').values
                            bal_score = roc_auc_score(y_test.values[:,each_i],mat,multi_class="ovr")
                        else:
                            bal_score = roc_auc_score(y_test.values[:,each_i],mat,multi_class="ovr")
                    else:
                        if isinstance(y_probas, dict):
                            if y_probas[each_i].ndim <= 1:
                                ## This is caused by Label Propagation hence you must probas like this ##
                                mat = y_probas[each_i]
                                if np.any(np.isnan(mat)):
                                    mat = pd.DataFrame(mat).fillna(method='ffill').values
                                bal_score = roc_auc_score(y_test.values[:,each_i],mat)
                            else:
                                bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[each_i][:,1])
                        else:
                            if y_probas.shape[1] == num_targets:
                                ### This means Label Propagation was used which creates probas like this ##
                                bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[:,each_i])
                            else:
                                ### This means regular sklearn classifiers which predict multi dim probas #
                                bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[each_i])
                    print('Target number %s: ROC AUC score %0.0f%%' %(each_i+1,100*bal_score))
                    bal_scores.append(bal_score)
                    if verbose:
                        print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
    final_score = np.mean(bal_scores)
    if verbose:
        print("final average balanced accuracy score = %0.2f" %final_score)
    return final_score
##############################################################################
import os
def check_if_GPU_exists():
    try:
        os.environ['NVIDIA_VISIBLE_DEVICES']
        print('GPU available on this device. Please activate it to speed up lightgbm.')
        return True
    except:
        print('No GPU available on this device. Using CPU for lightgbm and others.')
        return False
###############################################################################
def get_max_min_from_y(actuals):
    if isinstance(actuals, pd.Series):
        Y_min = actuals.values.min()
        Y_max = actuals.values.max()
    elif isinstance(actuals, pd.DataFrame):
        Y_min = actuals.values.ravel().min()
        Y_max = actuals.values.ravel().max()
    else:
        Y_min = actuals.min()
        Y_max = actuals.max()
    return np.array([Y_min, Y_max])

def convert_regression_to_classes(predictions, actuals):
    if isinstance(actuals, pd.Series):
        Y_min = actuals.values.min()
        Y_max = actuals.values.max()
    elif isinstance(actuals, pd.DataFrame):
        Y_min = actuals.values.ravel().min()
        Y_max = actuals.values.ravel().max()
    else:
        Y_min = actuals.min()
        Y_max = actuals.max()
    predictions = np.round(predictions,0).astype(int)
    predictions = np.where(predictions<Y_min, Y_min, predictions)
    predictions = np.where(predictions>Y_max, Y_max, predictions)
    return predictions
###############################################################################
from sklearn.metrics import mean_squared_error

def rmse(y_actual, y_predicted):
    return mean_squared_error(y_actual, y_predicted, squared=False)
##############################################################################
class SuloRegressor(BaseEstimator, RegressorMixin):
    """
    SuloRegressor works really fast and very well for all kinds of datasets.
    It works on small as well as big data. It works in Integer mode as well as float-mode.
    It works on regular balanced data as well as skewed regression targets.
    The reason it works so well is that Sulo is an ensemble of highly tuned models.
    You don't have to send any inputs but if you wanted to, you can send in two inputs:
    If you want, you can igore both these inputs and it will automatically choose these.
    It is fully compatible with scikit-learn pipelines and other models.

    Inputs:
    n_estimators: number of models you want in the final ensemble.
    base_estimator: base model you want to train in each of the ensembles.

    """
    def __init__(self, base_estimator=None, n_estimators=None, pipeline=True, imbalanced=False, 
                                       integers_only=False, log_transform=False, verbose=0):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.pipeline = pipeline
        self.imbalanced = imbalanced
        self.integers_only = integers_only
        self.log_transform = log_transform
        self.verbose = verbose
        self.models = []
        self.multi_label =  False
        self.scores = []
        self.integers_only_min_max = []
        self.model_name = ''
        self.features = []

    def fit(self, X, y):
        X = copy.deepcopy(X)
        print('Input data shapes: X = %s, y = %s' %(X.shape, y.shape,))
        seed = 42
        shuffleFlag = True
        modeltype = 'Regression'
        features_limit = 50 ## if there are more than 50 features in dataset, better to use LGBM ##
        start = time.time()
        if isinstance(X, pd.DataFrame):
            self.features = X.columns.tolist()
        else:
            print('Cannot operate SuloClassifier on numpy arrays. Must be dataframes. Returning...')
            return self
        # Use KFold for understanding the performance
        if self.imbalanced:
            print('Remember that using class weights will wrongly skew predict_probas from any classifier')
        ### Remember that putting class weights will totally destroy predict_probas ###
        gpu_exists = check_if_GPU_exists()
        if gpu_exists:
            device="gpu"
        else:
            device="cpu"
        row_limit = 10000
        if self.integers_only:
            self.integers_only_min_max = get_max_min_from_y(y)
            print('    Min and max values of y = %s' %self.integers_only_min_max)
        ################          P I P E L I N E        ##########################
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean", add_indicator=True)), ("scaler", StandardScaler())])

        categorical_transformer_low = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True)),
                ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),])

        categorical_transformer_high = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True)),
                ("encoding", LabelEncoder()),])

        numeric_features = X.select_dtypes(include='number').columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_cardinality(X, categorical_features)
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        ####################################################################################
        data_samples = X.shape[0]
        if self.n_estimators is None:
            if data_samples <= row_limit:
                self.n_estimators = min(5, int(1.5*np.log10(data_samples)))
            else:
                self.n_estimators = min(5, int(1.2*np.log10(data_samples)))
        num_splits = self.n_estimators
        self.model_name = 'lgb'
        num_repeats = 2
        kfold = RepeatedKFold(n_splits=num_splits, random_state=seed, n_repeats=num_repeats)
        num_iterations = int(num_splits * num_repeats)
        scoring = 'neg_mean_squared_error'
        print('    Num. estimators = %s (will be larger than n_estimators since kfold is repeated twice)' %num_iterations)
        ##### This is where we check if y is single label or multi-label ##
        if isinstance(y, pd.DataFrame):
            if self.log_transform:
                if self.verbose:
                    print('    transforming targets into log targets')
                y = np.log1p(y)
            ###############################################################
            ### This is for Multi-Label problems only #####################
            ###############################################################
            targets = y.columns.tolist()
            if is_y_object(y):
                print('Cannot perform Regression using object or string targets. Please convert to numeric and try again.')
                return self
            if len(targets) > 1:
                self.multi_label = y.columns.tolist()
                ### You need to initialize the class before each run - otherwise, error!
                if self.base_estimator is None:
                    ################################################################
                    ###   This is for Single Label Regression problems only ########
                    ###   Make sure you don't do imbalanced SMOTE work here  #######
                    ################################################################
                    if y.shape[0] <= row_limit:
                        if self.integers_only:
                            if (X.dtypes==float).all() and len(self.features) <= features_limit:
                                print('    Selecting Ridge as base_estimator. Feel free to send in your own estimator.')
                                self.base_estimator = Ridge(normalize=False)
                                self.model_name = 'other'
                            else:
                                if self.verbose:
                                    print('    Selecting LGBM Regressor as base estimator...')
                                self.base_estimator = LGBMRegressor(device=device, random_state=99)                                    
                        else:
                            if len(self.features) <= features_limit:
                                if self.verbose:
                                    print('    Selecting Bagging Regressor since integers_only flag is set...')
                                self.base_estimator = BaggingRegressor(n_estimators=200, random_state=99)
                                self.model_name = 'rf'
                            else:
                                if gpu_exists:
                                    if self.verbose:
                                        print('    Selecting XGBRegressor with GPU as base estimator...')
                                    self.base_estimator = XGBRegressor(n_estimators=250, 
                                        n_jobs=-1,tree_method = 'gpu_hist',gpu_id=0, predictor="gpu_predictor")
                                else:
                                    if self.verbose:
                                        print('    Selecting LGBM Regressor as base estimator...')
                                    self.base_estimator = LGBMRegressor(device=device, random_state=99)                                    
                    else:
                        if gpu_exists:
                            if self.verbose:
                                print('    Selecting XGB Regressor with GPU as base estimator...')
                            self.base_estimator = XGBRegressor(n_estimators=250, 
                                n_jobs=-1,tree_method = 'gpu_hist',gpu_id=0, predictor="gpu_predictor")
                        else:
                            if self.verbose:
                                print('    Selecting LGBM Regressor as base estimator...')
                            self.base_estimator = LGBMRegressor(n_estimators=250)
                else:
                    self.model_name == 'other'
                ### Remember we don't to HPT Tuning for Multi-label problems since it errors ####
                for i, (train_index, test_index) in enumerate(kfold.split(X)):
                    start_time = time.time()
                    # Split data into train and test based on folds          
                    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]                
                    else:
                        y_train, y_test = y[train_index], y[test_index]

                    if isinstance(X, pd.DataFrame):
                        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
                    else:
                        x_train, x_test = X[train_index], X[test_index]

                    ###### Do this only the first time ################################################
                    if i == 0:
                        ### It does not make sense to do hyper-param tuning for multi-label models ##
                        ###    since ClassifierChains do not have many hyper params #################
                        #self.base_estimator = rand_search(self.base_estimator, x_train, y_train, 
                        #                        self.model_name, verbose=self.verbose)
                        #print('    hyper tuned base estimator = %s' %self.base_estimator)
                        if self.verbose:
                            print('    Fitting with RegressorChain...')
                        est_list = [RegressorChain(self.base_estimator, order=None, random_state=i)
                                    for i in range(num_iterations)] 

                    # Initialize model with your supervised algorithm of choice
                    model = est_list[i]

                    # Train model and use it to train on the fold
                    if self.pipeline:
                        ### This is only with a pipeline ########
                        pipe = Pipeline(
                            steps=[("preprocessor", preprocessor), ("model", model)]
                        )

                        pipe.fit(x_train, y_train)
                        self.models.append(pipe)

                        # Predict on remaining data of each fold
                        preds = pipe.predict(x_test)

                    else:
                        #### This is without a pipeline ###
                        model.fit(x_train, y_train)
                        self.models.append(model)

                        # Predict on remaining data of each fold
                        preds = model.predict(x_test)

                        if self.log_transform:
                            preds = np.expm1(preds)
                        elif self.integers_only:
                            ### Use Regression predictions and convert them into integers here ##
                            preds = np.round(preds,0).astype(int)
                            
                    score = rmse(y_test, preds)
                    print("    Fold %s: Average OOF Error (smaller is better): %0.3f" %(i+1, score))
                    self.scores.append(score)
                    
                    # Finally, check out the total time taken
                    end_time = time.time()
                    timeTaken = end_time - start_time
                    print("Time Taken for fold %s: %0.0f (seconds)" %(i+1, timeTaken))
                
                # Compute average score
                averageAccuracy = sum(self.scores)/len(self.scores)
                if self.verbose:
                    print("Average RMSE score of %s-model SuloRegressor: %0.3f" %(
                                    num_iterations, averageAccuracy))
                end = time.time()
                timeTaken = end - start
                print("Time Taken overall: %0.0f (seconds)" %(timeTaken))
                return self
        ########################################################
        #####  This is for Single Label Classification problems 
        ########################################################
        
        if isinstance(y, pd.Series):
            targets = y.name
        else:
            targets = []
        if self.base_estimator is None:
            if data_samples <= row_limit:
                ### For small datasets use RFR for Regressions   ########################
                if len(self.features) <= features_limit:
                    if self.verbose:
                        print('    Selecting Bagging Regressor for this dataset...')
                    ### The Bagging Regresor outperforms ETC most of the time ####
                    self.base_estimator = BaggingRegressor(n_estimators=20)
                    self.model_name = 'bg'
                else:
                    if (X.dtypes==float).all():
                            print('    Selecting Ridge as base_estimator. Feel free to send in your own estimator.')
                            self.base_estimator = Ridge(normalize=False)
                            self.model_name = 'other'
                    else:
                        if self.verbose:
                            print('    Selecting LGBM Regressor as base estimator...')
                        self.base_estimator = LGBMRegressor(device=device, random_state=99)
            else:
                ### For large datasets Better to use LGBM
                if data_samples >= 1e5:
                    if gpu_exists:
                        if self.verbose:
                            print('    Selecting XGBRegressor with GPU as base estimator...')
                        self.base_estimator = XGBRegressor(n_jobs=-1,tree_method = 'gpu_hist',
                            gpu_id=0, predictor="gpu_predictor")
                    else:
                        self.base_estimator = LGBMRegressor(random_state=99) 
                else:
                    ### For smaller than Big Data, use Label Propagation which is faster and better ##
                    if (X.dtypes==float).all():
                            print('    Selecting Ridge as base_estimator. Feel free to send in your own estimator.')
                            self.base_estimator = Ridge(normalize=False)
                            self.model_name = 'other'
                    else:
                        if len(self.features) <= features_limit:
                            if self.verbose:
                                print('    Selecting Bagging Regressor for this dataset...')
                            self.base_estimator = BaggingRegressor(n_estimators=20)
                            self.model_name = 'bg'
                        else:
                            scoring = 'neg_mean_squared_error'
                            ###   Extra Trees is not so great for large data sets - LGBM is better ####
                            if gpu_exists:
                                if self.verbose:
                                    print('    Selecting XGBRegressor with GPU as base estimator...')
                                self.base_estimator = XGBRegressor(n_estimators=250, 
                                    n_jobs=-1,tree_method = 'gpu_hist',gpu_id=0, predictor="gpu_predictor")
                            else:
                                if self.verbose:
                                    print('    Selecting LGBM Regressor as base estimator...')
                                self.base_estimator = LGBMRegressor(n_estimators=250, random_state=99) 
                            self.model_name = 'lgb'
        else:
            self.model_name = 'other'

        est_list = num_iterations*[self.base_estimator]
        
        ### if there is a need to do SMOTE do it here ##
        smote = False
        #list_classes = return_minority_classes(y)
        #if not list_classes.empty:
        #    smote = True
        #### For now, don't do SMOTE since it is making things really slow ##
        
        # Perform CV
        for i, (train_index, test_index) in enumerate(kfold.split(X)):
            # Split data into train and test based on folds          
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]                
            else:
                y_train, y_test = y[train_index], y[test_index]

            if isinstance(X, pd.DataFrame):
                x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            else:
                x_train, x_test = X[train_index], X[test_index]

            # Convert the data into numpy arrays
            #if not isinstance(x_train, np.ndarray):
            #    x_train, x_test = x_train.values, x_test.values
            
            ##   small datasets processing #####
            if i == 0:
                if self.pipeline:
                    # Train model and use it in a pipeline to train on the fold  ##
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("model", self.base_estimator)])
                    if self.model_name == 'other':
                        print('No HPT tuning performed since base estimator is given by input...')
                        self.base_estimator = copy.deepcopy(pipe)
                    else:
                        if len(self.features) <= features_limit:
                            self.base_estimator = rand_search(pipe, x_train, y_train, 
                                                    self.model_name, self.pipeline, scoring, verbose=self.verbose)
                        else:
                            print('No HPT tuning performed since number of features is too large...')
                            self.base_estimator = copy.deepcopy(pipe)
                else:
                    ### This is for without a pipeline #######
                    if self.model_name == 'other':
                        ### leave the base estimator as is ###
                        print('No HPT tuning performed since base estimator is given by input...')
                    else:
                        if len(self.features) <= features_limit:
                            ### leave the base estimator as is ###
                            self.base_estimator = rand_search(self.base_estimator, x_train, 
                                                y_train, self.model_name, self.pipeline, scoring, verbose=self.verbose)
                        else:
                            print('No HPT tuning performed since number of features is too large...')

                est_list = num_iterations*[self.base_estimator]
                #print('    base estimator = %s' %self.base_estimator)
            
            ### SMOTE processing #####
            if i == 0:
                if smote:
                    print('Performing SMOTE...')
                    if self.verbose:
                        print('    x_train shape before SMOTE = %s' %(x_train.shape,))
                    
            if smote:
                # Get the class distribution for perfoming relative sampling in the next line
                ### It does not appear that class weights work well in SMOTE - hence avoid ###
                #class_weighted_rows = get_class_distribution(y_train, verbose)
                
                try:
                    sm = ADASYN(n_neighbors=5, random_state=seed, )
                                #sampling_strategy=class_weighted_rows)
                    
                    x_train, y_train = sm.fit_resample(x_train, y_train)
                    if i == 0:
                        print('    x_train shape after SMOTE = %s' %(x_train.shape,))
                except:
                    sm = SMOTETomek(random_state=42,)
                    #sm = ADASYN(n_neighbors=2, random_state=seed, )
                                #sampling_strategy=class_weighted_rows)
                    x_train, y_train = sm.fit_resample(x_train, y_train)                    
                    if i == 0 and smote:
                        print('    x_train shape after SMOTE = %s' %(x_train.shape,))
            
            # Initialize model with your supervised algorithm of choice
            model = est_list[i]
            
            model.fit(x_train, y_train)
            self.models.append(model)

            # Predict on remaining data of each fold
            preds = model.predict(x_test)

            if self.log_transform:
                preds = np.expm1(preds)
            elif self.integers_only:
                ### Use Regression predictions and convert them into integers here ##
                preds = np.round(preds,0).astype(int)

            score = rmse(y_test, preds)
            print("    Fold %s: Average OOF Error (smaller is better): %0.3f" %(i+1, score))
            self.scores.append(score)

        # Compute average score
        averageAccuracy = sum(self.scores)/len(self.scores)
        if self.verbose:
            print("Average RMSE of %s-model SuloRegressor: %0.3f" %(num_iterations, averageAccuracy))

        # Finally, check out the total time taken
        end = time.time()
        timeTaken = end-start
        print("Time Taken: %0.0f (seconds)" %timeTaken)
        return self

    def predict(self, X):
        from scipy import stats
        weights = 1/np.array(self.scores)
        if self.multi_label:
            ### In multi-label, targets have to be numeric, so you can leave weights as-is ##
            ypre = np.array([model.predict(X) for model in self.models ])
            y_predis = np.average(ypre, axis=0, weights=weights)
            if self.log_transform:
                y_predis = np.expm1(y_predis)
            ### leave the next line as if since you want to check for it separately
            if self.integers_only:
                y_predis = np.round(y_predis,0).astype(int)
            return y_predis
        y_predis = np.column_stack([model.predict(X) for model in self.models ])
        ### This weights the model's predictions according to OOB scores obtained
        #### In single label, targets can be object or string, so weights cannot be applied always ##
        y_predis = np.average(y_predis, weights=weights, axis=1)
        if self.log_transform:
            y_predis = np.expm1(y_predis)
        if self.integers_only:
            y_predis = np.round(y_predis,0).astype(int)
        return y_predis
    
    def predict_proba(self, X):
        print('In regression, no probabilities can be obtained.')
        return X

    def plot_importance(self, max_features=10):
        import lightgbm as lgbm
        from xgboost import plot_importance
        model_name = self.model_name
        feature_names = self.features

        if  model_name == 'lgb' or model_name == 'xgb':
            for i, model in enumerate(self.models):
                if self.pipeline:
                    if self.multi_label:
                        #model_object = model.named_steps['model'].base_estimator
                        print('No feature importances available for multi_label targets. Returning...')
                        return
                    else:
                        model_object = model.named_steps['model']
                else:
                    if self.multi_label:
                        #model_object = model.base_estimator
                        print('No feature importances available for multi_label targets. Returning...')
                        return
                    else:
                        model_object = model
                feature_importances = model_object.booster_.feature_importance(importance_type='gain')
                if i == 0:
                    feature_imp = pd.DataFrame({'Value':feature_importances,'Feature':feature_names})
                else:
                    feature_imp = pd.concat([feature_imp, 
                        pd.DataFrame({'Value':feature_importances,'Feature':feature_names})], axis=0)
                #lgbm.plot_importance(model_object, importance_type='gain', max_num_features=max_features)
            feature_imp = feature_imp.groupby('Feature').mean().sort_values('Value',ascending=False).reset_index()
            feature_imp.set_index('Feature')[:max_features].plot(kind='barh', title='Top 10 Features')
            ### This is for XGB ###
            #plot_importance(self.model.named_steps['model'], importance_type='gain', max_num_features=max_features)
        elif model_name == 'lp':
            print('No feature importances available for LabelPropagation algorithm. Returning...')
            return
        elif model_name == 'rf':
            ### These are for RandomForestClassifier kind of scikit-learn models ###
            try:
                for i, model in enumerate(self.models):
                    if self.pipeline:
                        if self.multi_label:
                            #model_object = model.named_steps['model'].base_estimator
                            print('No feature importances available for multi_label targets. Returning...')
                            return
                        else:
                            model_object = model.named_steps['model']
                    else:
                        if self.multi_label:
                            #model_object = model.base_estimator
                            print('No feature importances available for multi_label targets. Returning...')
                            return
                        else:
                            model_object = model
                    feature_importances = model_object.feature_importances_
                    if i == 0:
                        feature_imp = pd.DataFrame({'Value':feature_importances,'Feature':feature_names})
                    else:
                        feature_imp = pd.concat([feature_imp, 
                            pd.DataFrame({'Value':feature_importances,'Feature':feature_names})], axis=0)
                feature_imp = feature_imp.groupby('Feature').mean().sort_values('Value',ascending=False).reset_index()
                feature_imp.set_index('Feature')[:max_features].plot(kind='barh', title='Top 10 Features')
            except:
                    print('Could not plot feature importances. Please check your model and try again.')                
        else:
            print('No feature importances available for this algorithm. Returning...')
            return
#######################################################################################################