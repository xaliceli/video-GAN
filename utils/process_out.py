"""
process_out.py
Tensor to video.
"""

import os
import cv2
import numpy as np
import tensorflow as tf


def write_avi(frames, directory, name='', frate=24):
    writer = cv2.VideoWriter(os.path.join(directory, name),
                             cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             frate, frames[0].shape[:2])
    frames_concat = None
    for f_num, frame in enumerate(frames):
        if frames_concat is None:
            frames_concat = frame[0]
        else:
            frames_concat = np.hstack((frames_concat, frame[0]))
        writer.write(frame[0])
    cv2.imwrite(os.path.join(directory, 'epoch' + name + '.jpg'), frames_concat)
    writer.release()


def convert_image(images, samples):
    images = tf.cast(np.clip(((images + 1.0) * 127.5), 0, 255), tf.uint8)
    images = [tf.squeeze(image).numpy() for image in tf.split(images, samples, axis=1)]
    return images