import numpy as np
np.set_printoptions(precision=4)

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from mobrob_sim_loader import DatasetLoader
import tensorflow as tf
import vae_models
from tkinter import *
from PIL import Image, ImageTk
import utils
from tensorflow_addons import image
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
batch_size = 1


# select model, if none then cvae from paper is used
sylvester = True
testing = False

# model parameters
latent_size = 64
lstm_cells = 512
T = 40
number_mixtures = 1

# gaussian blurring
gauss_filter = False
filter_shape = [7,7]
filter_sigma = 1.5

# display parameters
display_scale = 4
window_width = 600
window_height = 600

# simulation parameters
action_power = 3.5

data = "sim"
vae_type = "cvae"
rnn_type = "mdn"
beta = 1
temperature = 0.001
warmup = 15

#########################################
# TRAINING DATA

if data=="sim":
    # load training data
    batch_size = 1
    input_path_train = "data/f_mobrob_sim_test.tfrecord"
    ds_loader = DatasetLoader(batch_size, T, height, width, color_channels, action_dim)
    test_data = ds_loader.load(input_path_train, 10000)
    test_data_float = test_data.map(lambda x,y: (tf.math.divide(x, 255),y)) # zu float in 0.0...1.0
        
    if gauss_filter:
        # gauss filterung / unschaerfe in den bildern
        test_data_float = test_data_float.map(lambda x, y: (tf.map_fn(lambda y: image.gaussian_filter2d(image=y, filter_shape=filter_shape, sigma=filter_sigma), x), y))

    test_data_float = test_data_float.unbatch()



elif data=="gym":

    test_data_float = gym_dataset_util.load_test_data(batch_size, T).unbatch().map(lambda imgs,acts: (tf.image.convert_image_dtype(imgs, dtype=tf.float32), acts)).repeat(10000)



#########################################
# MODEL CREATION AND LOADING


vae = utils.create_vae_model(vae_type, latent_size, data, warmup, beta, False, False, -1)
vae_checkpoint = "./checkpoints_vae/{}".format(vae.model_name)
try:
    vae.load_weights(vae_checkpoint)
    print(vae.model_name, "loaded")
except:
    print(vae.model_name, "failed")



rnn, rnn_name = utils.create_rnn_model(vae_type, rnn_type, latent_size, 40, lstm_cells, data, number_mixtures, batch_size, beta, temperature)
rnn_checkpoint = './checkpoints_rnn/{}'.format(rnn_name)
try:
    rnn.load_weights(rnn_checkpoint)
    print(rnn_name, "loaded")
except:
    print(rnn_name, "failed")





#########################################
# RESTLICHES ZEUG


window = Tk()
window.title("rnn test and stuff ...")
window.geometry("{}x{}".format(window_width, window_height))
label = Label(window, image=ImageTk.PhotoImage(image=Image.fromarray(np.zeros((64*display_scale, 64*display_scale, 3), dtype=np.uint8))))
label.grid(column=1, row=0)

current_images = np.zeros((T-1, 64,64,3))
current_actions = np.zeros((T-1, action_dim))

z_now = 0
h_now = 0
c_now = 0

test_iter = test_data_float.__iter__()


def reset_current():
    global test_iter, z_now, h_now, c_now

    warmup_images, warmup_actions = next(test_iter)

    warmup_z = vae.encode(warmup_images)
    warmup_input = tf.expand_dims(tf.concat((warmup_z, warmup_actions), axis=1), axis=0)

    z_now, h_now, c_now = rnn.predict(warmup_input)

reset_current()



def generate_next_image(action):
    global z_now, h_now, c_now

    inputs = tf.expand_dims(tf.concat((z_now, [action]), axis=1), axis=0)

    z_now, h_now, c_now = rnn.predict_with_state(inputs, h_now, c_now)
 
    image_now = np.uint8(vae.decode(z_now)[0]*255)

    return image_now




counter = 0

# wird benoetigt, weil der garbagecollector sonst die bilder rausschmeisst
keep_out_of_garbage = []

def get_next_img(action):
    global counter

    image_now = generate_next_image(action)    
    img = Image.fromarray(image_now).resize((64*display_scale, 64*display_scale))
    imgTk = ImageTk.PhotoImage(image=img)

    keep_out_of_garbage.append(imgTk)

    return imgTk





# beide + : vorwaerts

# -, + : rechts drehen

# +, - : links drehen


def w_click():
    
    if data=="gym": action = [0.0, 0.5, 0.0]
    else: action = [action_power, action_power]
    
    #print("vorwaerts  ", action)
    img = get_next_img(action)
    label.configure(image=img)

def a_click():
    if data=="gym": action = [-0.2, 0.0, 0.0]
    else: action = [action_power, -action_power]
    
    #print("links      ", action)
    img = get_next_img(action)
    label.configure(image=img)

def s_click():
    if data=="gym":action = [0.0, 0.0, 1.0]
    else: action = [-action_power, -action_power]
    
    #print("hinten     ", action)
    img = get_next_img(action)
    label.configure(image=img)

def d_click():
    if data=="gym": action = [0.2, 0.0, 0.0]
    else: action = [-action_power, action_power]
    
    #print("rechts     ", action)
    img = get_next_img(action)
    label.configure(image=img)

def empty_click():
    if data=="gym": action = [0.0, 0.0, 0.0]
    else: action = [0, 0]
    
    img = get_next_img(action)
    label.configure(image=img)

def reset():
    global counter
    counter = 0
    reset_current()

window.bind('w', lambda event: w_click())
window.bind('a', lambda event: a_click())
window.bind('s', lambda event: s_click())
window.bind('d', lambda event: d_click())
window.bind('<space>', lambda event: empty_click())

window.bind('r', lambda event: reset())


vorward = Button(window, height=5, width=15, text="w", command=w_click)
vorward.grid(column=1, row=2)

left = Button(window, height=5, width=15, text="a", command=a_click)
left.grid(column=0, row=3)

right = Button(window, height=5, width=15, text="s", command=s_click)
right.grid(column=1, row=3)

back = Button(window, height=5, width=15, text="d", command=d_click)
back.grid(column=2, row=3)

resetB = Button(window, height=5, width=15, text="reset", command=reset)
resetB.grid(column=1, row=4)

window.mainloop()
