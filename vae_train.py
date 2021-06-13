import numpy as np
np.set_printoptions(precision=4)
import shutil

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import vae_models
from custom_training_classes import SSIM_Metric, PSNR_Metric, SaveReconstructions


log_dir = "logs" 



def train_vae(data, vae_type, latent_size, beta, epochs, batch_size, keep=False, train=False):

    # MODEL CREATION AND LOADING
    if vae_type == "cvae":
        model = vae_models.CVAE(data, latent_size, beta)
    elif vae_type == "dense":
        model = vae_models.DenseVAE(data, latent_size, beta)
    

    optimizer = tf.keras.optimizers.Adam(0.0001)
    model.compile(optimizer=optimizer, metrics=[SSIM_Metric(), PSNR_Metric()])
    
    if keep:
        try:
            model.load_weights(f"checkpoints_vae/{model.model_name}")
            print(model.model_name, "loaded")
        except:
            print(model.model_name, "failed")

    print("Training", model.model_name)
    

    # LOADING DATASETS
    dataset = tfds.load(data, as_supervised=True, batch_size=-1)
    X_train, Y_train = tfds.as_numpy(dataset["train"])
    X_test, Y_test = tfds.as_numpy(dataset["test"])
    
    if X_train.shape[3] == 1:
        X_train = X_train.repeat(3, axis=-1)
        X_test = X_test.repeat(3, axis=-1)
    
    shuffle_buffer = 1000

    train_data = tf.data.Dataset.from_tensor_slices(X_train)
    train_data = train_data.map(lambda x: (tf.image.resize(x, (64,64)), 0)).shuffle(shuffle_buffer, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    train_steps = int(X_train.shape[0] / batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(X_test)
    test_data = test_data.map(lambda x: (tf.image.resize(x, (64,64)), 0)).batch(batch_size, drop_remainder=True)
    test_steps = int(X_test.shape[0] / batch_size)

    val_data = test_data
    val_steps = test_steps


    # TRAIN
    callbacks = [
        #keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(f"checkpoints_vae/{model.model_name}", save_weights_only=True),
        SaveReconstructions(val_data),
        keras.callbacks.TensorBoard(log_dir=log_dir + f"/{model.model_name}", histogram_freq=1, write_images=True) 
    ]

    if train:
        shutil.rmtree(log_dir + f"/{model.model_name}", ignore_errors=True)
        model.fit(train_data, epochs=epochs, steps_per_epoch=train_steps, validation_data=val_data, validation_steps=val_steps, callbacks=callbacks)
    else:
        try:
            model.load_weights(f"checkpoints_vae/{model.model_name}")
            print(model.model_name, "loaded")
        except:
            print(model.model_name, "failed")
        model.evaluate(val_data, steps=test_steps)


def main():
        
    # Fixing some Memory Issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)



    if not os.path.isdir("output_images"):
        os.mkdir("output_images")



    batch_size = 128
    epochs = 10
    train_vae("cifar10", "cvae", 64, 1, epochs, batch_size, keep=True, train=True)



if __name__ == "__main__":
    main()