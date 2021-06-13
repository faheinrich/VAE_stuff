import numpy as np
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt
import random

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons import image
import vae_models
from mobrob_sim_loader import DatasetLoader
#from vae_von_lars import ResidualModuleType, SylvesterFlow, ResidualModuleConv, vonLarsVAE
import utils
#import gym_dataset_util

import rnn_models

utils.fix_gpu_mem()




#########################################
# PARAMETERS

# data properties
height = 64
width = 64
color_channels = 3
action_dim = 2

# training parameters
training_steps = 10000
batch_size = 32
use_z_vectors_in_loss = True
output_series_bool = False

# select model, if none then cvae from paper is used
sylvester = True

# model parameters
latent_size = 64
lstm_cells = 512
T = 20
number_mixtures = 5 # currently unused

# gaussian blurring
gauss_filter = False
filter_shape = [7,7]
filter_sigma = 1.5

# display parameters
display_scale = 4





def eval_rnn(vae, rnn, num_test_batches, prediction_length, batch_size, data):


    latent_size = rnn.latent_size
    #T = rnn.T
    T = 5
    test_series_length = T + prediction_length



    if data == "sim":
        input_path_train = "data/f_mobrob_sim_test.tfrecord"
        ds_loader = DatasetLoader(batch_size, test_series_length, height, width, color_channels, action_dim)
        test_data = ds_loader.load(input_path_train, 10000)
        test_data_float = test_data.map(lambda x,y: (tf.image.convert_image_dtype(x, dtype=tf.float32), y))

    if data == "gym":
        test_data_float = gym_dataset_util.load_test_data(batch_size, test_series_length).unbatch().map(lambda imgs,acts: (tf.image.convert_image_dtype(imgs, dtype=tf.float32), acts)).batch(batch_size, drop_remainder=True).repeat(10000)


  
    test_iterator = test_data_float.__iter__()


    ssim = 0.0
    pstnr = 0.0
    for i in range(num_test_batches):

        images, actions = next(test_iterator)

        # warmup

        z_vecs = tf.map_fn(vae.encode, images)
        warmup_input = tf.concat((z_vecs[:, 0:T], actions[:, 0:T]), axis=2)
        z, h, c = rnn.predict(warmup_input)

        predicted_zvecs = np.zeros((batch_size, prediction_length, latent_size))
        predicted_zvecs[:, 0] = z

        for i in range(prediction_length-1):


            new_input =  tf.concat((z, actions[:, T + i]), axis=1)
            new_input = tf.expand_dims(new_input, axis=1)
            #print(new_input.shape)
            z, h, c = rnn.predict_with_state(new_input, h,c)
            predicted_zvecs[:, i+1] = z

    

        truth_images = images[:,T:]
        predicted_images = tf.map_fn(vae.decode, tf.cast(predicted_zvecs, tf.float32))
        

        stacked = tf.stack((truth_images, predicted_images), axis=2)

        ssim_stack = tf.map_fn(lambda x: tf.image.ssim(x[:,0], x[:,1], max_val=1.0), stacked)
        pstnr_stack = tf.map_fn(lambda x: tf.image.psnr(x[:,0], x[:,1], max_val=1.0), stacked)

        ssim += tf.reduce_mean(ssim_stack)
        pstnr += tf.reduce_mean(pstnr_stack)

        
    ssim /= num_test_batches
    pstnr /= num_test_batches
    return ssim, pstnr




