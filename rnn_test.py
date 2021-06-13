import numpy as np
np.set_printoptions(precision=4)
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import cv2
from PIL import Image


import tensorflow as tf

import utils

import rnn_models

import vae_models
from mobrob_sim_loader import DatasetLoader


#########################################
# PARAMETERS





# data properties
height = 64
width = 64
color_channels = 3 

# training parameters
batch_size = 1

# model parameters
lstm_cells = 512
T = 5

# gaussian blurring
gauss_filter = False
filter_shape = [7,7]
filter_sigma = 1.5

# display parameters
display_scale = 4

prediction_length = 40

test_series_length = T + prediction_length


utils.fix_gpu_mem()



data = "sim"

if data == "sim":
    input_path_train = "data/f_mobrob_sim_test.tfrecord"
    ds_loader = DatasetLoader(batch_size, test_series_length, 64, 64, 3, 2)
    test_data = ds_loader.load(input_path_train, 10000)
    test_data_float = test_data.map(lambda x,y: (tf.math.divide(x, 255),y)).shuffle(100).unbatch()


vae_types = ["cvae", "dense"]

rnn_types = ["simple", "mdn"]
temperatures = [0.0001]

betas = [1]
trainT = 40
warmup = 1
rnns_vaes = []
mixtures = 1
latents = [64]

for latent_size in latents:
    for rnn_type in rnn_types:
        for beta in betas:
            for temperature in temperatures:
                for vae_type in vae_types:

                        
                    vae = utils.create_vae_model(vae_type, latent_size, data, warmup, beta, False, False, -1)
                    vae_checkpoint = "./checkpoints_vae/{}".format(vae.model_name)
                    try:
                        vae.load_weights(vae_checkpoint)
                        print("   vae:", vae.model_name, "loaded")
                    except:
                        print("   !!! vae:", vae_checkpoint, "failed")


                    rnn, rnn_name = utils.create_rnn_model(vae_type, rnn_type, latent_size, trainT, lstm_cells, data, mixtures, batch_size, beta, temperature)
                    rnn_checkpoint = "./checkpoints_rnn/{}".format(rnn_name)
                    try:
                        rnn.load_weights(rnn_checkpoint)
                        rnns_vaes.append((rnn, vae))
                        print(rnn_name, "loaded")
                    except:
                        print("!!!", rnn_name, "failed")


print("num vaes:", len(rnns_vaes))




#########################################
# MAIN LOOP

for images, actions in  test_data_float.__iter__():
    
    ## show warmup images
    #for time, img in enumerate(images[0:T]):
    #    disp_start = []
    #    for i in range(len(rnns_vaes)+1):
    #        disp_start.append(np.uint8(img * 255))
    #    display = np.concatenate(disp_start, axis=1)
    #    pil_img = Image.fromarray(display).resize((64 * display_scale * (len(rnns_vaes)+1), 64 * display_scale))
    #    cv2.imshow("image", np.uint8(pil_img)[:,:,::-1])
    #    cv2.waitKey()


    model_outputs = []


    # warmup
    truth_image = np.uint8(images[T] * 255)
    show_images = [truth_image]
    for i, element in enumerate(rnns_vaes):
        rnn, vae = element

        warmup_input = tf.concat((vae.encode(images[0:T]), actions[0:T]), axis=1)
        z, h, c = rnn.predict(tf.expand_dims(warmup_input, axis=0))
        model_outputs.append((z,h,c))
        decoded_out = vae.decode(z)[0]
    
        show_images.append(np.uint8(decoded_out*255))

    display = np.concatenate(show_images, axis=1)
    pil_img = Image.fromarray(display).resize((height * display_scale * (len(rnns_vaes)+1), width * display_scale))
    cv2.imshow("image", np.uint8(pil_img)[:,:,::-1])
    cv2.waitKey()


    # prediction
    for shift in range(test_series_length - T):

        truth_image = np.uint8(images[shift+T] * 255)
        show_images = [truth_image]

        for i, element in enumerate(rnns_vaes):
            rnn, vae = element

            z, h, c = model_outputs[i]
            action = actions[shift]
            
            inputs = tf.expand_dims(tf.concat((z, [actions[shift+T]]), axis=1), axis=0)

            z, h, c = rnn.predict_with_state(inputs, h, c)
            model_outputs[i] = (z,h,c)

            decoded_out = vae.decode(z)[0]
            show_images.append(np.uint8(decoded_out*255))
 
        concat = np.concatenate(show_images, axis=1)
        display = cv2.resize(concat, (concat.shape[1]*display_scale, concat.shape[0]*display_scale))
        cv2.imshow("image", np.uint8(display)[:,:,::-1])
        if cv2.waitKey(0) & 0xFF == 27: # use ESC to quit
            cv2.destroyAllWindows()
            quit()



    
    



print("fertig\n")
