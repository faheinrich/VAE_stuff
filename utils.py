from tensorflow_addons import image as tf_image
from mobrob_sim_loader import DatasetLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import tensorflow as tf
import vae_models
#from vae_von_lars import ResidualModuleType, SylvesterFlow, ResidualModuleConv, vonLarsVAE
import rnn_models

def show_training_progress(vae, test_images, c, num_show):

    display_scale = 2
    dpi = 300
    height = 64
    width = 64

    test_reconstruction = vae.call(test_images)[0]
    reconstructions_concat = np.concatenate(np.uint8(test_reconstruction*255),axis=1)
    test_images_concat = np.concatenate(np.uint8(test_images*255), axis=1)
    concat = np.concatenate( (test_images_concat, reconstructions_concat), axis=0)
    
    resize_shape = (64*num_show*display_scale ,64*display_scale*2) # 2 images
    cv2.imwrite("output_images/{}_step{}.png".format(vae.model_name, c), cv2.resize(concat, resize_shape)[:,:,::-1])



def generate_gauss_low(size):
    pascals_triangle = []
    for i in range(size):
        value = math.factorial(size - 1) / (math.factorial(i) * math.factorial(size - 1 - i))
        pascals_triangle.append(value)
    pascals_triangle = np.array(pascals_triangle)

    gauss = np.outer(pascals_triangle, pascals_triangle)
    gauss = gauss / np.sum(gauss)

    gauss = np.expand_dims(gauss, axis=2)
    gauss = np.repeat(gauss, 3, axis=2)
    gauss = np.expand_dims(gauss, axis=-1)
    

    return tf.constant(gauss, tf.float32)

def generate_gauss_high(size):
    pascals_triangle = []
    for i in range(size):
        value = math.factorial(size - 1) / (math.factorial(i) * math.factorial(size - 1 - i))
        pascals_triangle.append(value)
    pascals_triangle = np.array(pascals_triangle)

    gauss = np.outer(pascals_triangle, pascals_triangle)
    gauss = gauss / np.sum(gauss)

    inv_gauss = np.zeros((size,size))
    inv_gauss[int(size/2), int(size/2)] = 1.0

    inv_gauss = inv_gauss - gauss
    
    kernel = np.expand_dims(inv_gauss, axis=2)
    kernel = np.repeat(kernel, 3, axis=2)
    kernel = np.expand_dims(kernel, axis=-1)


    return tf.constant(kernel, tf.float32)



def fix_gpu_mem():
        
    # Fehler durch fehlenden GPU Speicher umgehen
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


def create_vae_model(vae_type, latent_size, data, batch_size, beta, lowpass, highpass, filter_size):

    if vae_type=="cvae": 
        vae = vae_models.CVAE(data, latent_size, beta)
    elif vae_type=="dense":
        vae = vae_models.DenseVAE(data, latent_size, beta)
    return vae
    #vae_name = "{}_{}_{}_b{}".format(vae_type, latent_size, data, beta)

    #if lowpass:
    #    vae_name += "_low_{}".format(filter_size)
    #elif highpass:
    #    vae_name += "_high_{}".format(filter_size)

    #return vae, vae_name

def create_rnn_model(vae_type, rnn_type, latent_size, T, lstm_cells, data, number_mixtures, batch_size, beta, temperature):
    rnn_name = "{}_{}_{}_{}_b{}_T{}".format(vae_type, latent_size, rnn_type, data, beta, T)

    if rnn_type=="simple":
        rnn_model = rnn_models.SIMPLE_LSTM(latent_size, lstm_cells, number_mixtures)
    elif rnn_type=="mdn":
        rnn_model = rnn_models.MDN_RNN(latent_size, lstm_cells, number_mixtures, temperature)
        rnn_name += "_m{}".format(number_mixtures)

    return rnn_model, rnn_name