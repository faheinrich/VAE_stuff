import tensorflow as tf
import numpy as np
import cv2
import os

class SSIM_Metric(tf.keras.metrics.Metric):

    def __init__(self, name='SSIM', **kwargs):
        super(SSIM_Metric, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name='ssim', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
        self.ssim.assign(ssim)   

    def result(self):
        return self.ssim

    def reset_states(self):
        self.ssim.assign(0)


class PSNR_Metric(tf.keras.metrics.Metric):

    def __init__(self, name='PSNR', **kwargs):
        super(PSNR_Metric, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        psnr = tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))
        self.psnr.assign(psnr) 

    def result(self):
        return self.psnr

    def reset_states(self):
        self.psnr.assign(0)



class SaveReconstructions(tf.keras.callbacks.Callback):
    def __init__(self, test_loader):
        super(SaveReconstructions, self).__init__()
        images, actions = next(test_loader.__iter__())
        self.images = tf.math.divide(images[0:8], 255)


    def on_epoch_end(self, epoch, logs=None):
        reconstructions, _, _, _ = self.model(self.images)

        imgs_concat = np.concatenate(np.uint8(self.images*255), axis=1)
        rec_concat = np.concatenate(np.uint8(reconstructions*255),axis=1)
        
        display = np.concatenate( (imgs_concat, rec_concat), axis=0)

        create_dir = f"./output_images/{self.model.model_name}" 
        if not os.path.isdir(create_dir):
            os.mkdir(create_dir)
        cv2.imwrite(f"output_images/{self.model.model_name}/{epoch}.png", display[:,:,::-1])

