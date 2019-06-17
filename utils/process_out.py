"""
process_out.py
Tensor to video.
"""

import os
import cv2
import numpy as np
import tensorflow as tf


def write_avi(frames, directory, name='', frate=24):
    writer = cv2.VideoWriter(os.path.join(directory, name + '.avi'),
                             cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             frate, frames[0][0].shape[-3:-1])
    frames_concat = None
    for f_num, frame in enumerate(frames):
        if frames_concat is None:
            frames_concat = frame[0]
        else:
            frames_concat = np.hstack((frames_concat, frame[0]))
        writer.write(frame[0])
    cv2.imwrite(os.path.join(directory, 'epoch' + name + '.jpg'), frames_concat)
    writer.release()


def convert_image(images):
    images = tf.cast(np.clip(((images + 1.0) * 127.5), 0, 255), tf.uint8)
    images = np.split(images.numpy(), images.get_shape().as_list()[0], axis=0)
    return images