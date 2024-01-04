
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import pdb
###############################################################################
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
        for i in range(y_train.shape[1]):  # Iterate over columns, not dimensions
            ### Test dtypes for each target column ###
            if y_train.iloc[:, i].dtype in ['int64', 'int32', 'int16']:
                unique_values = np.unique(y_train.iloc[:, i])  # Use np.unique for NumPy arrays
                if len(unique_values) <= 2:
                    model_class = 'Binary_Classification'
                elif len(unique_values) > 2 and len(unique_values) <= cat_limit:
                    model_class = 'Multi_Classification'
                else:
                    model_class = 'Regression'
            elif y_train.iloc[:, i].dtype in ['float16','float32','float64']:
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
    if verbose:
        if multi_label:
            print('''    %s %s problem ''' %('Multi_Label', model_class))
        else:
            print('''    %s %s problem ''' %('Single_Label', model_class))
    return model_class, multi_label

from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score, mean_squared_error
from sklearn.utils import shuffle
import numpy as np
from sklearn.utils import shuffle
import copy
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Get predictions and calculate residuals for Regression
import matplotlib.pyplot as plt
def plot_residuals(svm_classifier, encoded_X_train, encoded_y_train,
                   encoded_X_test, y_test):
    y_train_pred = svm_classifier.predict(encoded_X_train)
    y_test_pred = svm_classifier.predict(encoded_X_test)

    # Calculate residuals
    train_residuals = encoded_y_train - (y_train_pred.flatten())
    test_residuals = y_test - (y_test_pred.flatten())

    # Plot residuals
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, train_residuals, alpha=0.5)
    plt.title('Training Residuals' )
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, test_residuals, alpha=0.5)
    plt.title('Test Residuals')
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')

    plt.tight_layout()
    plt.show()

###############################################################################
# Assuming `print_regression_metrics`, `print_classification_metrics`, `permutation_importance`, 
# and `plot_residuals` are defined elsewhere in your code
# These functions should be defined to evaluate the model and print/plot the metrics
import tensorflow as tf
from collections import OrderedDict
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import copy
#######################################################################################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

class print_on_10(Callback):
    def on_batch_end(self, epoch, logs=None):
        if epoch%5==0: 
            print()

def print_nn_model_summary(history, classifier):
    print(classifier.summary())
    metrics = ['loss'] + [m for m in history.history.keys() if 'val_' not in m and m != 'loss']
    for m in metrics:
        if 'lr' in m:
            continue
        plt.plot(history.history[m])
        plt.plot(history.history['val_' + m])
        plt.title('Model ' + m)
        plt.ylabel(m)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

########################################################################################
def _build_nn_model(input_dim, num_samples, num_classes=None, modeltype='Regression',
                   batch_size=32, learning_rate=0.001, activation='relu', 
                   num_hidden_layers=2, verbose=0):
    # Define activation and regularization
    initial_lr=0.001
    dropout_rate = 0.2 ### this is just a placeholder for now
    l1_reg = 0.1*initial_lr
    l2_reg = 0.1*initial_lr
    regularization = l1_l2(l1=l1_reg, l2=l2_reg)
    
    ### You need to over-ride the num classes in two cases ###
    if modeltype == 'Binary_Classification':
        last_layer_activation = 'sigmoid'
        num_classes = 1
        #loss_function = 'binary_crossentropy'
        loss_function = weighted_binary_crossentropy
        #metrics = 'accuracy'
        metrics = F1Score()
    elif modeltype == 'Multi_Classification':
        last_layer_activation = 'softmax'
        #loss_function = 'categorical_crossentropy'
        loss_function = weighted_categorical_crossentropy
        ### try f1_score or accuracy ###########
        #metrics = 'accuracy'
        metrics = F1Score()
    else: 
        last_layer_activation = 'linear'
        num_classes = 1
        loss_function = 'mean_squared_error'
        loss_function = weighted_mse
        metrics = 'mse'

    if verbose:
        print(modeltype, num_classes, last_layer_activation, loss_function, metrics)
        
    ##### This is the plain old DNN with dense and hidden layers
    # Dynamically set the layer sizes based on the input dimension
    if input_dim <= 5:
        base_size = min(512, input_dim*10)  # Increased the potential size for more complexity
    else:
        base_size = min(512, input_dim*5)  # Increased the potential size for more complexity
    layer_sizes = []
    for num in range(num_hidden_layers):
        layer_sizes.append(int(base_size*(1/(num+1))))

    if verbose:
        print(f"Number of hidden layers: {num_hidden_layers}")
        print(f"Neurons in each layer: {layer_sizes}")

    model = Sequential()
    # Input layer
    model.add(Dense(layer_sizes[0], input_dim=input_dim, activation=activation, kernel_regularizer=regularization))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Dynamic hidden layers
    for size in layer_sizes[1:]:
        model.add(Dense(size, activation=activation, kernel_regularizer=regularization))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(num_classes, activation=last_layer_activation))

    # Compile the model with the optimizer
    if modeltype == 'Regression':
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                      loss=loss_function, metrics=[metrics])  # Use MSE for regression
    elif modeltype == 'Binary_Classification':
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                      loss=loss_function, metrics=[metrics],
                      )  
        # Use binary crossentropy and F1 score for binary classification
    else:
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                      loss=loss_function, metrics=[metrics],
                      )  
        # Use categorical crossentropy for multi-class

    return model

from sklearn.base import BaseEstimator, TransformerMixin
class NN_Modeler(BaseEstimator, TransformerMixin):
    def __init__(self, activation='selu', learning_rate=0.001, 
                 batch_size=32, num_hidden_layers=2,
                epochs=200,
                validation_split=0.2,
                callbacks=None,
                verbose=0):
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_hidden_layers = num_hidden_layers
        self.epochs = epochs
        self.validation_split = validation_split
        self.callbacks = callbacks
        self.verbose=verbose
        self.class_weights = None
        self.modeltype = None
        self.history = None
        self.model = None
        self.num_classes = None
        self.encoder = None

    def fit(self, X, y):
        #### set some 
        input_dim = X.shape[1]
        num_samples = X.shape[0]

        # Adjust batch size based on the input provided
        if num_samples < self.batch_size:
            self.batch_size = max(num_samples // 2, 2)  # Ensure the batch size is not 0

        
        self.modeltype, multi_label = analyze_problem_type_array(y)

        if self.modeltype == 'Multi_Classification':
            ### You must get the class weights before y is one-hot encoded
            self.class_weights = get_class_weights(y)
            self.num_classes = len(np.unique(y))
            encoded_y = to_categorical(y, num_classes=self.num_classes)
        elif self.modeltype == 'Binary_Classification':
            ### You must get the class weights before y is one-hot encoded
            self.class_weights = get_class_weights(y)
            self.num_classes = 2
            encoded_y = copy.deepcopy(y)
        else:
            self.num_classes = 1
            encoded_y = copy.deepcopy(y)
        
        ### You need to over-ride the num classes in two cases ###
        if self.modeltype == 'Binary_Classification':
            ### this is a string used to find the val_string: used below
            metrics = 'f1_score'
            mode = 'max'
            # Assuming you have a function to calculate class weights
        elif self.modeltype == 'Multi_Classification':
            ### this is a string used to find the val_string: used below
            metrics = 'f1_score'
            #metrics = 'accuracy'
            mode = 'max'
        else: 
            metrics = 'mse'
            mode = 'min'

        if self.callbacks is None:
            reduce_lr = ReduceLROnPlateau(monitor='val_'+metrics, factor=0.90, patience=5, min_lr=0.0001)
            early_stopping = EarlyStopping(monitor='val_'+metrics, patience=25, mode=mode, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
            self.callbacks = [reduce_lr, early_stopping]
        else:
            ### You need this since you need to save and load checkpoints of best model below ##
            model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
            self.callbacks += [model_checkpoint]

        # Build the model #### This needs num_classes and modeltype ##
        self.model = _build_nn_model(input_dim=input_dim, 
                                num_samples=num_samples, num_classes=self.num_classes, 
                                modeltype=self.modeltype,
                                batch_size=self.batch_size, 
                                learning_rate=self.learning_rate, 
                                activation=self.activation, 
                                num_hidden_layers=self.num_hidden_layers,
                                verbose=self.verbose)
        start_time = time.time()

        self.history = self.model.fit(X, encoded_y, epochs=self.epochs,
                                 validation_split=self.validation_split,
                                 callbacks=self.callbacks,
                                 class_weight = self.class_weights,
                                 )

        # Load the best weights and evaluate the model
        self.model.load_weights('best_model.h5')

        print('Time taken = %0.0f seconds' %(time.time()-start_time))
        print_nn_model_summary(self.history, self.model)        
        return self

    def predict(self, X, y=None):
        if self.modeltype != 'Regression':
            probas = self.model.predict(X)
            predictions = np.argmax(probas, axis=1)
        else:
            predictions = self.model.predict(X).flatten()
            if log_y:
                predictions = np.exp(predictions) - 1
        return predictions

    def predict_proba(self, X, y=None):
        if self.modeltype == 'Multi_Classification':
            probas = self.model.predict(X)
        elif self.modeltype == 'Binary_Classification':
            y_probas = self.model.predict(X)
            probas = np.concatenate([1-y_probas, y_probas],axis=1)
        else:
            probas = None
        return probas
##############################################################################
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import balanced_accuracy_score, classification_report
import numpy as np

def get_class_weights(y_input):
    """
    Compute class weights based on the frequency of classes in the input labels.

    Parameters:
        y_input (numpy.ndarray): Array of labels.

    Returns:
        class_weights (list): List of class weights.
    """
    # Ensure y_input is a numpy array
    y_input = np.array(y_input)

    # Compute class frequencies
    class_counts = np.bincount(y_input)

    # Compute class weights based on class frequencies
    total_samples = len(y_input)
    class_weights = (total_samples / (len(class_counts) * class_counts)).astype(int)
    class_weights = np.where(class_weights==0.0,1.0,class_weights).astype(int)
    class_weights = dict(zip(np.unique(y_input), class_weights))
    return class_weights


# Define a custom loss function using the precomputed threshold
# This approach provides an approximation and might not be as
#  precise as other methods, especially for small tensors.
def weighted_mse(y_true, y_pred):
    scaling_factor = 2.0  # This can be tuned

    # Flatten y_true to 1D and sort the values
    values = tf.sort(tf.reshape(y_true, [-1]), axis=-1)

    # Calculate the index for the 75th percentile
    index = tf.cast(tf.round(0.75 * (tf.cast(tf.size(values), tf.float32) - 1)), tf.int32)

    # Select the value at the 75th percentile
    threshold = tf.gather(values, index)

    # Compute weights based on the threshold
    weights = tf.where(y_true > threshold, scaling_factor, 1.0)

    # Calculate weighted MSE
    weighted_squared_error = tf.multiply(weights, tf.square(y_true - y_pred))

    # Return the mean loss
    return tf.reduce_mean(weighted_squared_error)


### This requires tensoflow probability function which comes only with tf 2.11
class WeightedMSELoss(tf.keras.losses.Loss):
    def __init__(self, quantile=0.75, scaling_factor=2.0, name="weighted_mse_loss"):
        super().__init__(name=name)
        self.quantile = quantile
        self.scaling_factor = scaling_factor

    def call(self, y_true, y_pred):
        # Calculate the dynamic threshold based on the quantile of y_true
        threshold = tfp.stats.percentile(y_true, self.quantile * 100)

        # Calculate weights based on the threshold
        weights = tf.where(y_true > threshold, self.scaling_factor, 1.0)

        # Calculate weighted MSE
        squared_error = tf.square(y_true - y_pred)
        weighted_squared_error = weights * squared_error
        return tf.reduce_mean(weighted_squared_error)

# Define a simple learning rate schedule
# It drops a tiny amount every 20 epochs
def lr_schedule(epoch, lr):
    # Drop the learning rate every 20 epochs
    if (epoch + 1) % 20 == 0:  # +1 makes it check for epochs 20, 40, 60, ...
        return lr * 0.99
    else:
        return lr
    
def weighted_binary_crossentropy(y_true, y_pred):
    ### Use this if using sigmoid activation function for binary classes ####
    
    # Define the weights for each class, ensuring they are of type float32
    # Example weights for classes must be a list of weights like this: [1.0, 2.0, 3.0]
    #class_weights = tf.constant([1.0, 10.0], dtype=tf.float32)

    # Ensure y_true is of type float32
    y_true = tf.cast(y_true, tf.float32)
    
    # Calculate the binary cross-entropy loss
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = bce(y_true, y_pred)

    # Apply the class weights
    CLASS_WTS = tf.constant(list(get_class_weights_fast(y_true)), dtype=tf.float32)
    weight_vector = tf.reduce_sum(CLASS_WTS * y_true, axis=-1)
    
    # Ensure weight_vector is of the same type as loss
    weight_vector = tf.cast(weight_vector, loss.dtype)

    # Calculate the weighted loss
    weighted_loss = loss * weight_vector

    # Return the mean loss
    return tf.reduce_mean(weighted_loss)

def weighted_categorical_crossentropy(y_true, y_pred):
    # Compute the categorical cross-entropy loss without reduction
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = cce(y_true, y_pred)

    # Compute weights based on y_true
    class_weights = get_class_weights_fast(y_true)

    # Expand dimensions of class_weights to match the shape of y_true
    class_weights = tf.reshape(class_weights, (-1, 1))

    # Calculate the weighted loss
    weighted_loss = loss * class_weights

    # Return the mean loss
    return tf.reduce_mean(weighted_loss)

def get_class_weights_fast(y_true):
    # Compute class frequencies
    class_counts = tf.math.bincount(tf.cast(y_true, dtype=tf.int32))

    # Compute class weights based on class frequencies
    total_samples = tf.size(y_true)
    class_weights = tf.cast(total_samples, dtype=tf.float32) / (tf.cast(tf.size(class_counts), dtype=tf.float32) * tf.cast(class_counts, dtype=tf.float32))

    return class_weights
#################################################################################################
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.metrics import Metric
import time

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), dtype=self.dtype))
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), dtype=self.dtype))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), dtype=self.dtype))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = tf.divide(self.true_positives, tf.add(self.true_positives, self.false_positives))
        recall = tf.divide(self.true_positives, tf.add(self.true_positives, self.false_negatives))
        f1_score = 2 * ((precision * recall) / (precision + recall + 1e-12))
        return f1_score

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
#################################################################################