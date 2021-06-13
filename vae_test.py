import numpy as np
np.set_printoptions(precision=4)
import cv2
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
import tensorflow_datasets as tfds

import vae_models


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



    batch_size = 40



    # LOADING DATASETS
    data = "mnist"

    dataset = tfds.load(data, split="test", as_supervised=True, batch_size=-1)
    X_test, Y_test = tfds.as_numpy(dataset)

    if X_test.shape[3] == 1:
        X_test = X_test.repeat(3, axis=-1)

    test_data = tf.data.Dataset.from_tensor_slices(X_test)
    test_data = test_data.map(lambda x: (tf.image.resize(x, (64,64)), 0)).batch(batch_size, drop_remainder=True)



    # ADD MODELS TO COMPARE TO EACH OTHER
    vaes = []
    latent_sizes = [2,4,8,16,32]
    betas = [1]
    for i in latent_sizes:
        for beta in betas:
            vae = vae_models.CVAE(data, i, beta)
            try:
                vae.load_weights("./checkpoints_vae/{}".format(vae.model_name))
                vaes.append(vae)
                print(vae.model_name, "added")
            except:
                print(vae.model_name, "failed")


    
    # test the model visually
    for images, _ in test_data:
        outputs = []
        for vae in vaes:
            z = vae.encode(images)
            x = vae.decode(z)
            outputs.append(vae.call(x)[0])

        for c, img in enumerate(images):
            visualize = []
            visualize.append(img / 255.0)

            for output in outputs: 
                visualize.append(output[c])

            concat = np.concatenate(visualize, axis=1)
            display = cv2.resize(concat, (64*len(visualize)*2, 64 * 2))

            cv2.imshow("image", np.uint8(display * 255)[:,:,::-1])
            if cv2.waitKey(0) & 0xFF == 27: # use ESC to quit
                break



        
if __name__ == "__main__":
    main()

