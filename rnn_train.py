import numpy as np
np.set_printoptions(precision=4)
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import time
import utils
import tensorflow as tf
from mobrob_sim_loader import DatasetLoader
import vae_models
import rnn_models
import rnn_eval



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




    # PARAMETERS
    data = "sim"


    # data properties
    if data=="gym": action_dim = 3
    else: action_dim = 2

    # training parameters
    batch_size = 32
    interval = 100

    # model parameters
    lstm_cells = 512


    num_test_batches = 10
    prediction_length = 20



    images_directly = False


    def train_rnn(vae_type, rnn_type, latent_size, data, training_steps, beta, lowpass, highpass, filter_size, mixtures, T):

        whole_time = time.time()

        if data == "gym":
            train_data_float = gym_dataset_util.load_train_data(batch_size, T+1).unbatch().map(lambda imgs,acts: (tf.image.convert_image_dtype(imgs, dtype=tf.float32), acts)).batch(batch_size, drop_remainder=True).repeat(10000)

        elif data == "sim":
            input_path_train = "data/f_mobrob_sim_train.tfrecord"
            ds_loader = DatasetLoader(16, T+1, 64, 64, 3, 2)
            train_data_float = ds_loader.load(input_path_train, 1000000).unbatch().shuffle(1000, reshuffle_each_iteration=True).map(lambda imgs,acts: (tf.image.convert_image_dtype(imgs, dtype=tf.float32), acts)).batch(batch_size, drop_remainder=True)
        else:
            print("WRONG DATASET NAME")


        print("\n----------------------------------\n")


        if vae_type=="cvae": 
            vae = vae_models.CVAE(data, latent_size, beta)
        elif vae_type=="dense":
            vae = vae_models.DenseVAE(data, latent_size, beta)
        #vae = utils.create_vae_model(vae_type, latent_size, data, batch_size, beta, False, False, -1)
        vae_checkpoint = "./checkpoints_vae/{}".format(vae.model_name)
        try:
            vae.load_weights(vae_checkpoint)
            print(vae.model_name, "loaded")
        except:
            print(vae.model_name, "failed")


        
        rnn_model, rnn_name = utils.create_rnn_model(vae_type, rnn_type, latent_size, T, lstm_cells, data, mixtures, batch_size, beta, 0.1)

        print("training", rnn_name)

        rnn_checkpoint = "./checkpoints_rnn/{}".format(rnn_name)
        try:
            rnn_model.load_weights(rnn_checkpoint)
            print(rnn_name, "loaded")
        except:
            print(rnn_name, "failed")


                


        #########################################
        # TRAINING

        loss_function = rnn_model.get_loss_function()

        optimizer = tf.keras.optimizers.Adam(1e-4)


        # custom train fuction
        @tf.function
        def train(data_batch):

            images, actions = data_batch
            z_vectors = tf.map_fn(vae.encode, images)

            model_input = tf.concat((z_vectors, actions), axis=2)[:, :-1, ...]

            target_z_vectors = z_vectors[:, 1: ,...]



            with tf.GradientTape() as tape:

                output = rnn_model(model_input)
    
                loss = loss_function(target_z_vectors, output)
            

            grads_g = tape.gradient(loss, rnn_model.trainable_variables)
            optimizer.apply_gradients(zip(grads_g, rnn_model.trainable_variables))

            

        ################
        # MAIN TRAINLOOP


        train_iterator = train_data_float.__iter__()

        plot_i = []
        plot_ssim = []
        plot_pstnr = []
        time_start = time.time()

        for i in range(training_steps+1):
            
            data_batch = next(train_iterator)
            train(data_batch)

            if(i % interval == 0):
                ssim, pstnr = rnn_eval.eval_rnn(vae, rnn_model, num_test_batches, prediction_length, batch_size, data)
                plot_i.append(i)
                plot_ssim.append(ssim)
                plot_pstnr.append(pstnr)
                print("training step", "{}".format(i).rjust(6), "      time = %.4f" % (time.time()-time_start), "      ssim = %.4f" % ssim, "      pstnr = %.4f" % pstnr)
                time_start = time.time()

                rnn_model.save_weights(rnn_checkpoint)
            
        rnn_model.save_weights(rnn_checkpoint)


        #plot_save = np.stack((np.array(plot_i), np.array(plot_ssim), np.array(plot_pstnr)), axis=0)
        #np.save("./plots/rnn/progress_{}".format(rnn_name), plot_save)
        print("time for {} batches: {} s\n".format(training_steps, "%.2f" % (time.time() - whole_time)))


        print("fertig")

        
    train_rnn("cvae", "mdn", 64, "sim", 10000, 1, False, False, -1, 1, 40)


if __name__ == "__main__":
    main()