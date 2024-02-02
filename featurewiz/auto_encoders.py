import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import f1_score
import copy
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from .stacking_models import analyze_problem_type_array
from sklearn.utils import check_X_y
import pdb

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
#############################################################################################
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

    #### This is to make sure dims are less than train shape for FeatureExtractor #####
    ls = [4, 8, 16]
    final_dims = [x for x in ls if x < X_train.shape[1]]
    param_grid = {
        'feature_extractor__batch_size': [16, 32, 64],
        'feature_extractor__encoding_dim': final_dims,
        #'feature_extractor__noise_type': ['gaussian', 'dropout'],
        #'feature_extractor__noise_factor': [0.1, 0.2],
        'feature_extractor__epochs': [10],
        'feature_extractor__learning_rate': [0.001, 0.01],    
    }

    # Setup grid search
    grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', 
                               cv=3, n_jobs=-1,verbose=1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    return grid_search

def vae_hyperparam_selection(dae, X_train, y_train):
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

    #### This is to make sure dims are less than train shape for FeatureExtractor #####
    ls = [4, 8, 16]
    final_dims = [x for x in ls if x < X_train.shape[1]]
    param_grid = {
        'feature_extractor__batch_size': [16, 32, 64],
        'feature_extractor__latent_dim': final_dims,
        'feature_extractor__epochs': [10],
        'feature_extractor__learning_rate': [0.001, 0.01],    
    }

    # Setup grid search
    grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', 
                               cv=3, n_jobs=-1,verbose=1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    return grid_search

#############################################################################################
class CNNAutoEncoder(BaseEstimator, TransformerMixin):
    """
    Convolutional Neural Network Autoencoder Transformer

    This class implements a Convolutional Neural Network (CNN) based autoencoder transformer.
    It is designed to encode and decode input data using convolutional layers, which can be
    useful for feature extraction and dimensionality reduction.

    Parameters:
    ----------
    latent_dim : int, optional (default=10)
        The dimension of the latent space representation.

    kernel_size : int, optional (default=3)
        The size of the convolutional kernel used in the encoder and decoder.

    filters : int, optional (default=32)
        The number of filters (feature maps) used in the convolutional layers.

    activation : str, optional (default='relu')
        The activation function used in the convolutional layers.

    pool_size : int, optional (default=2)
        The size of the max-pooling window used in the encoder.

    epochs : int, optional (default=100)
        The number of training epochs for the autoencoder.

    batch_size : int, optional (default=32)
        The batch size used during training.

    Methods:
    -------
    fit(X, y=None)
        Train the CNN autoencoder on the input data.

    transform(X)
        Encode the input data into the latent space.

    Attributes:
    ----------
    encoder_ : tensorflow.keras.Model
        The encoder component of the CNN autoencoder.

    decoder_ : tensorflow.keras.Model
        The decoder component of the CNN autoencoder.

    Examples:
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> data = load_digits()
    >>> X = data.data
    >>> scaler = MinMaxScaler()
    >>> X = scaler.fit_transform(X)
    >>> autoencoder = CNNAutoencoder(latent_dim=2, epochs=50, batch_size=32)
    >>> autoencoder.fit(X)
    >>> encoded_data = autoencoder.transform(X)

    Notes:
    -----
    - The autoencoder is trained to learn a compact representation of the input data in the
      latent space by encoding and decoding it using convolutional layers.
    - The encoder and decoder models can be accessed through the `encoder_` and `decoder_` attributes.
    """
    def __init__(self, latent_dim=10, kernel_size=3, filters=32, activation='relu',
                 pool_size=2, epochs=100, batch_size=32):
        try:
            from tensorflow import keras
            def set_seed(seed=42):
                np.random.seed(seed)
                random.seed(seed)
                tf.random.set_seed(seed)
            set_seed(42)
        except:
            print('    You need to pip install tensorflow>= 2.5 in order to use this Autoencoder option.')
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.filters = filters
        self.activation = activation
        self.pool_size = pool_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = None
        self.autoencoder_ = None
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10,
                        verbose=1, mode='min', baseline=None, restore_best_weights=True)
        # Learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                            patience=5, min_lr=0.0001)

        self.callbacks = [es, lr_scheduler]

    def _build_autoencoder(self):
        from tensorflow import keras
        inputs = keras.layers.Input(shape=self.input_shape)
        # Encoder
        x = keras.layers.Reshape((*self.input_shape, 1))(inputs)  # Add 3rd dimension
        x = keras.layers.Conv1D(self.filters, self.kernel_size, activation=self.activation, padding='same')(x)
        x = keras.layers.MaxPooling1D(self.pool_size)(x)
        x = keras.layers.Conv1D(self.filters * 2, self.kernel_size, activation=self.activation, padding='same')(x)
        x = keras.layers.MaxPooling1D(self.pool_size)(x)
        x = keras.layers.Flatten()(x)
        encoded = keras.layers.Dense(self.latent_dim)(x)

        # Decoder
        x = keras.layers.Dense(np.prod(self.input_shape))(encoded)
        x = keras.layers.Reshape((self.input_shape[0] // (self.pool_size ** 2), -1))(x)  # Adjust the shape accordingly
        x = keras.layers.Conv1DTranspose(self.filters * 2, self.kernel_size, activation=self.activation, padding='same')(x)
        x = keras.layers.UpSampling1D(self.pool_size)(x)
        x = keras.layers.Conv1DTranspose(self.filters, self.kernel_size, activation=self.activation, padding='same')(x)
        x = keras.layers.UpSampling1D(self.pool_size)(x)
        x = keras.layers.Flatten()(x)
        decoded = keras.layers.Dense(np.prod(self.input_shape), activation='linear')(x)

        autoencoder = keras.Model(inputs, decoded, name="autoencoder")
        return autoencoder

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.input_shape = X.shape[1:]

        if self.autoencoder_ is None:
            self.autoencoder_ = self._build_autoencoder()

        self.autoencoder_.compile(optimizer='adam', loss='mse')
        self.autoencoder_.fit(X, X, epochs=self.epochs, batch_size=self.batch_size,
                              callbacks=self.callbacks, shuffle=True, validation_split=0.20,
                             )
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        encoded_features = self.autoencoder_.predict(X)
        return encoded_features
############################################################################################
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
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
        print('tensorflow >= 2.5 not installed in machine. Please install and try again. ')

    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        def call(self, inputs):
            try:
                import tensorflow as tf
                from tensorflow.keras import layers
                from tensorflow.keras import backend as K
                def set_seed(seed=42):
                    np.random.seed(seed)
                    random.seed(seed)
                    tf.random.set_seed(seed)
                set_seed(42)
            except:
                print('tensorflow>= 2.5 not installed in machine. Please install and try again. ')
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def __init__(self, latent_dim=2, activation='selu', epochs = 200,
                    batch_size=32, learning_rate=0.001):
        try:
            from tensorflow import keras
        except:
            print('tensorflow>= 2.5 not installed in machine. Please install and try again. ')
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.epochs = epochs
        #### These are not defined by users but intelligent defaults chosen by algorithm ###
        self.original_latent_dim = latent_dim
        self.intermediate_dim = 32
        self.original_intermediate_dim = 32
        self.original_batch_size = batch_size
        self.input_dim = None
        self.tasktype = 'Regression'  ### set the default to Regression ####
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10,
                        verbose=1, mode='min', baseline=None, restore_best_weights=False)
        # Learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                            patience=10, min_lr=0.0001)

        self.callbacks = [es, lr_scheduler]

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers, Model, backend as K
        from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
        from tensorflow.keras.losses import mse

        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)
        # Manually specify the activation function of the last layer
        # Adjust based on your model's specific configuration
        if self.tasktype == 'Regression':
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
            x = layers.Dense(layer_size, activation=self.activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)  # Assuming a dropout rate of 10%
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = self.Sampling()([z_mean, z_log_var])
        encoder = Model(inputs=original_inputs, outputs=[z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,), name='z_sampling')
        x = latent_inputs
        for _ in range(hidden_layers):
            x = layers.Dense(layer_size, activation=self.activation)(x)
            x = BatchNormalization()(x)
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
            self.tasktype = analyze_problem_type_array(y)
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
    def __init__(self, input_dim, embedding_dim, output_dim, epochs=200, batch_size=32):
        try:
            import tensorflow as tf
            def set_seed(seed=42):
                np.random.seed(seed)
                random.seed(seed)
                tf.random.set_seed(seed)
            set_seed(42)
        except:
            print('tensorflow>= 2.5 not installed in machine. Please install and try again. ')
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Initialize optimizers
        self.gen_optimizer = self._get_optimizer()
        self.disc_optimizer = self._get_optimizer()

    def _get_optimizer(self):
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
        return Adam()

    class Generator:
        def __init__(self, input_dim, embedding_dim, output_dim):
            self.input_dim = input_dim
            self.embedding_dim = embedding_dim
            self.output_dim = output_dim

        def build(self):
            import tensorflow as tf
            from tensorflow.keras.layers import Dense, Activation
            model = tf.keras.Sequential([
                Dense(self.embedding_dim, input_shape=(self.input_dim,)),
                Activation('relu'),
                Dense(self.embedding_dim * 2),
                Activation('relu'),
                Dense(self.output_dim, activation='sigmoid')
            ])
            return model

    class Discriminator:
        def __init__(self, input_dim):
            self.input_dim = input_dim

        def build(self):
            import tensorflow as tf
            from tensorflow.keras.layers import Dense, Activation, Dropout
            model = tf.keras.Sequential([
                Dense(self.input_dim, input_shape=(self.input_dim,)),
                Activation(tf.nn.leaky_relu),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            return model

    def _build_generator(self):
        generator = self.Generator(self.input_dim, self.embedding_dim, self.output_dim)
        return generator.build()

    def _build_discriminator(self):
        discriminator = self.Discriminator(self.output_dim)
        return discriminator.build()

    def fit(self, X, y):
        import tensorflow as tf
        from tensorflow.keras.losses import BinaryCrossentropy

        loss_fn = BinaryCrossentropy()
        X, y = check_X_y(X, y)

        # Initialize a variable to keep track of the best discriminator loss
        best_disc_loss = float('inf')
        patience = 5  # You can adjust this value as needed

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
        return self

    def generate_data(self, num_samples):
        import tensorflow as tf
        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
        set_seed(42)

        noise = tf.random.normal([num_samples, self.input_dim])
        return self.generator(noise).numpy()
############################################################################################
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
                ### if input dimension is not given, use the same size as X's features
                self.input_dim = int(rlog(X_class.shape[1], 4))*5

            ### Don't change this! This needs to be the same as X's features
            output_dim = X_class.shape[1]

            gan = self.gan_model(self.input_dim, self.embedding_dim, output_dim, self.epochs)
            gan.fit(X_class, y[y == class_label])  # Pass y for the specific class

            self.gans[class_label] = gan
        
        print('GANAugmenter adds more rows, not columns. %s rows added for this train dataset..' %(
                len(np.unique(y)*self.num_synthetic_samples)))
        return self

    def transform(self, X, y=None):
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
        
        if y is not None:
            print("Shape of y:", y.shape)
            print("Shape of all_synthetic_labels:", all_synthetic_labels.shape)
            combined_data = np.vstack([X, all_synthetic_data])
            combined_labels = np.concatenate([y, all_synthetic_labels])
        else:
            combined_data = all_synthetic_data
            combined_labels = all_synthetic_labels

        return combined_data, combined_labels
############################################################################
