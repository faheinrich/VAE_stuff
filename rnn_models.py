import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
import numpy as np

class MDN_RNN(tf.keras.Model):
    def __init__(self, latent_size, lstm_size, number_mixtures, temperature):
        super(MDN_RNN, self).__init__()
        self.latent_size = latent_size
        #self.action_size = action_size
        self.lstm_size = lstm_size
        #self.T = T
        self.number_mixtures = number_mixtures

        self.temperature = temperature

        self.lstm_cell = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True, time_major=False)

        self.mdn = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.lstm_size)),
            tf.keras.layers.Dense(units=3*latent_size*number_mixtures),
        ])

    def predict(self, inputs):
        rnn_out, h, c  = self.lstm_cell(inputs) # zeitliche Komponente

        mdn_out = self.mdn(h) # 3*number_mixtures* z_size Parameter der Mixture Models
        mdn_params = mdn_out
        mdn_out = tf.reshape(mdn_out, [-1, 3*self.number_mixtures])
        mu, logstd, logpi = tf.split(mdn_out, num_or_size_splits=3, axis=1)

        logpi = logpi / self.temperature # temperature
        logpi = tf.keras.activations.softmax(logpi, axis=-1)

        
        cat = tfd.Categorical(logits=logpi)
        component_splits = [1] * self.number_mixtures
        mus = tf.split(mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(tf.exp(logstd) * tf.sqrt(self.temperature), component_splits, axis=1) 

        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        z = tf.reshape(mixture.sample(), shape=(-1, self.latent_size))
        return z, h, c

    def predict_with_state(self, inputs, h, c):
        rnn_out, h, c  = self.lstm_cell(inputs, initial_state=[h,c]) # zeitliche Komponente

        mdn_out = self.mdn(h) # 3*number_mixtures* z_size Parameter der Mixture Models
        mdn_params = mdn_out
        mdn_out = tf.reshape(mdn_out, [-1, 3*self.number_mixtures])
        mu, logstd, logpi = tf.split(mdn_out, num_or_size_splits=3, axis=1)

        logpi = logpi / self.temperature # temperature
        logpi = tf.keras.activations.softmax(logpi, axis=-1)

        
        cat = tfd.Categorical(logits=logpi)
        component_splits = [1] * self.number_mixtures
        mus = tf.split(mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(tf.exp(logstd) * tf.sqrt(self.temperature), component_splits, axis=1) 

        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        z = tf.reshape(mixture.sample(), shape=(-1, self.latent_size))
        return z, h, c


        

    def call(self, inputs):
        rnn_out, _, _  = self.lstm_cell(inputs) # zeitliche Komponente

        mdn_out = self.mdn(rnn_out) # 3*number_mixtures* z_size Parameter der Mixture Models
        mdn_params = mdn_out
        
        return mdn_params


    def get_loss_function(self):

        def calc_loss(target, output):
            z_true = target
            mdnrnn_params = output
            batch_size = 32


            mdnrnn_params = tf.reshape(mdnrnn_params, [-1, 3*self.number_mixtures], name='reshape_ypreds')
            
            vae_z = tf.reshape(z_true, [-1, 1])

            out_mu, out_logstd, out_logpi = tf.split(mdnrnn_params, num_or_size_splits=3, axis=1, name='mdn_coef_split')
            
            out_logpi = out_logpi - tf.reduce_logsumexp(input_tensor=out_logpi, axis=1, keepdims=True) # normalize

            logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
            lognormal = -0.5 * ((vae_z - out_mu) / tf.exp(out_logstd)) ** 2 - out_logstd - logSqrtTwoPI
            v = out_logpi + lognormal
            
            z_loss = -tf.reduce_logsumexp(input_tensor=v, axis=1, keepdims=True)
            z_loss = tf.reduce_sum(z_loss) # batch
            return z_loss

        return calc_loss


class SIMPLE_LSTM(tf.keras.Model):
    def __init__(self, latent_size, lstm_size, number_mixtures):
        super(SIMPLE_LSTM, self).__init__()
        self.latent_size = latent_size
        self.lstm_size = lstm_size
        #self.T = T

        self.lstm_cell = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True, time_major=False)

        self.mdn = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.lstm_size)),
            tf.keras.layers.Dense(units=latent_size),
        ])

    def predict(self, inputs):
        rnn_out, h, c  = self.lstm_cell(inputs)
        z = self.mdn(h)
        return z, h, c

        
    def predict_with_state(self, inputs, h, c):
        rnn_out, h, c  = self.lstm_cell(inputs, initial_state=[h,c])
        z = self.mdn(h)
        return z, h, c

    def call(self, inputs):
        rnn_out, h, c  = self.lstm_cell(inputs)
        z = self.mdn(h)
        model_out = tf.map_fn(self.mdn, rnn_out)
        return model_out, z

    #def calc_loss(self, truth_z, output):
    #    rnn_out, z = output
    #    stacked = tf.stack((truth_z, rnn_out), axis=1)
    #    mse_loss = tf.reduce_sum(tf.map_fn(lambda x: tf.keras.losses.MeanSquaredError()(x[0], x[1]), stacked))
    #    return mse_loss

    def get_loss_function(self):

        def calc_loss(truth_z, output):
            rnn_out, z = output
            stacked = tf.stack((truth_z, rnn_out), axis=1)
            mse_loss = tf.reduce_sum(tf.map_fn(lambda x: tf.keras.losses.MeanSquaredError()(x[0], x[1]), stacked))
            return mse_loss

        return calc_loss




# Arbeitet auf den Bilder direkt, nicht auf z-Vektoren.
class LSTM_ON_IMAGES(tf.keras.Model):
    def __init__(self, latent_size, lstm_size, T, number_mixtures):
        super(LSTM_ON_IMAGES, self).__init__()
        self.latent_size = latent_size
        self.lstm_size = lstm_size
        self.T = T


        self.network = tf.keras.models.Sequential([
            tf.keras.layers.Input((T, 64*64*3 + 2)),
            tf.keras.layers.LSTM(lstm_size, return_sequences=False, return_state=False, time_major=False),
            tf.keras.layers.Dense(units=64*64*3, activation='sigmoid'),

            tf.keras.layers.Reshape([64,64,3]), 
        ])


    def call(self, inputs):
        return self.network(inputs)

    def calc_loss(self, true_image, prediction):
        return tf.keras.losses.MeanSquaredError()(true_image, prediction)

    def get_loss_function(self):
        def calc_loss(truth, prediction):
            return tf.keras.losses.MeanSquaredError()(truth, prediction)
        return calc_loss