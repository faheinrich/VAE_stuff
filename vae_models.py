import tensorflow as tf


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, data, latent_dim, beta):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.data = data
        self.beta = beta
        
        self.model_name = f"cvae_{data}_L{latent_dim}_b{beta}"

        self.epoch_counter = 0

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64,64,3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=1*1*1024, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(1,1,1024)),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation='sigmoid')
        ])

    # q(z|X)
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        return z

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    # P(X|z)
    def decode(self, z):
        x = self.decoder(z)
        return x

    def call(self, inputs):
        mean, logvar = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, z, mean, logvar

    def calc_loss(self, images, model_outputs):
        reconstructions, z, mean, logvar = model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[3]

        reconstruction_loss = tf.reduce_sum(tf.square(images-reconstructions), axis=[1,2,3])
        mean_reconstruction_loss = tf.reduce_mean(reconstruction_loss)


        kld = -0.5 * tf.reduce_sum(1 + logvar - tf.math.pow(mean,2) - tf.math.exp(logvar), axis=1)
        mean_kld = tf.reduce_mean(kld)
        
        loss = mean_reconstruction_loss + (self.beta * mean_kld)

        return loss, reconstructions


    def train_step(self, data):

        images, actions = data
        images = tf.math.divide(images, 255)

        with tf.GradientTape() as tape: 
            tf.summary.scalar("test", 1, step=self.optimizer.iterations)
            model_outputs = self(images, training=True)
            #loss, reconstructions = self.compiled_loss(images, model_outputs)
            loss, reconstructions = self.calc_loss(images, model_outputs)

        # calc gradients and update
        trainable_variables = self.trainable_variables
        grads_g = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads_g, trainable_variables))

        self.compiled_metrics.update_state(images, reconstructions)
        
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, actions = data
        images = tf.math.divide(images, 255)

        model_outputs = self(images, training=False)
        loss, reconstructions = self.calc_loss(images, model_outputs)

        self.compiled_metrics.update_state(images, reconstructions)
        return {m.name: m.result() for m in self.metrics}







class DenseVAE(tf.keras.Model):

    def __init__(self, data, latent_dim, beta):
        super(DenseVAE, self).__init__()

        self.latent_dim = latent_dim
        self.data = data
        self.beta = beta
        
        self.model_name = f"dense_{data}_L{latent_dim}_b{beta}"

        self.epoch_counter = 0

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64,64,3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(units=64*64*3, activation='sigmoid'),
            tf.keras.layers.Reshape((64,64,3))
        ])

    # q(z|X)
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        return z

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    # P(X|z)
    def decode(self, z):
        x = self.decoder(z)
        return x

    def call(self, inputs):
        mean, logvar = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, z, mean, logvar

    def calc_loss(self, images, model_outputs):
        reconstructions, z, mean, logvar = model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[3]

        reconstruction_loss = tf.reduce_sum(tf.square(images-reconstructions), axis=[1,2,3])
        mean_reconstruction_loss = tf.reduce_mean(reconstruction_loss)


        kld = -0.5 * tf.reduce_sum(1 + logvar - tf.math.pow(mean,2) - tf.math.exp(logvar), axis=1)
        mean_kld = tf.reduce_mean(kld)
        
        loss = mean_reconstruction_loss + (self.beta * mean_kld)

        return loss, reconstructions


    def train_step(self, data):

        images, actions = data
        images = tf.math.divide(images, 255)

        with tf.GradientTape() as tape: 
            model_outputs = self(images, training=True)
            #loss, reconstructions = self.compiled_loss(images, model_outputs)
            loss, reconstructions = self.calc_loss(images, model_outputs)

        # calc gradients and update
        trainable_variables = self.trainable_variables
        grads_g = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads_g, trainable_variables))

        self.compiled_metrics.update_state(images, reconstructions)
        
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, actions = data
        images = tf.math.divide(images, 255)

        model_outputs = self(images, training=False)
        loss, reconstructions = self.calc_loss(images, model_outputs)

        self.compiled_metrics.update_state(images, reconstructions)
        return {m.name: m.result() for m in self.metrics}