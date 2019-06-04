"""
train.py
Runs chosen model given params.
"""

import tensorflow as tf

from models.progressive.progressive import ProgressiveModel
from utils.process_in import tf_dataset

PARAMS = {
    'model': 'ProgressiveModel', # Model to run
    'vid_size': 64, # Size of video
    'start_size': 8,  # Starting size of video (only applies to progressive training)
    'num_frames': 32, # Number of frames
    'batch_size': 4,  # Batch size
    'epochs': 1000,  # Number of epochs
    'z_dim': 100,  # Size of noise vector
    'conv_init': 'he_normal',  # Initialization of convolutions
    'optimizer': 'adam',  # Optimizer
    'lr': 0.0002,  # Learning rate
    'b1': 0.5,  # Adam beta1
    'b2': 0.99,   # Adam beta2
    'disc_iterations': 1,  # Number of discriminator updates
    'gen_iterations': 1,  # Number of generator updates
    'save_int': 50,  # Number of epochs before generating output
    'num_out': 1,  # Number of outputs generated
    'data_dir': '/content/drive/My Drive/Colab Data/video-gan/inputs',  # Input data directory
    'save_dir': '/content/drive/My Drive/Colab Data/video-gan/progressive/outputs'  # Output data directory
}

if __name__ == '__main__':
    tf.enable_eager_execution()
    dataset = tf_dataset(dir=PARAMS['data_dir'],
                         frames=PARAMS['num_frames'],
                         target_size=PARAMS['vid_size'],
                         batch_size=PARAMS['batch_size'])
    model = locals()[PARAMS['model']](**PARAMS)
    model.train(videos=dataset, **PARAMS)
