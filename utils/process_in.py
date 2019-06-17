"""
process_in.py
Video to image as needed.
Image to Tensorflow dataset.
"""

import glob
import os
import cv2
import numpy as np
import tensorflow as tf


def vid_to_img(dir, frame_cap, frame_int=1):
    """
    Converts videos to image files with frames vertically stacked.
    """
    videos = glob.glob(os.path.join(dir, '*.avi'))
    for vnum, video in enumerate(videos):
        description = os.path.splitext(video)[0]
        vidcap = cv2.VideoCapture(os.path.join(dir, video))
        success, image = vidcap.read()
        output = np.zeros((frame_cap * image.shape[0], image.shape[1], image.shape[2]))
        loc, frames = 0, 0
        while success and frames < frame_cap:
            output[frames * image.shape[0]:(frames + 1) * image.shape[0]] = image
            loc += frame_int
            frames += 1
            vidcap.set(cv2.CAP_PROP_POS_MSEC, loc)
            success, image = vidcap.read()
        if frames == frame_cap:
            cv2.imwrite(os.path.join(dir, description + str(vnum) + '.jpg'), np.float32(output))
    vidcap.release()


def parse_video(filename, target_size, og_size, frames):
    """
    Structure each sample as tensor of size (frames, height, width, channels)
    and normalize.
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.cast(tf.image.decode_jpeg(image_string, channels=3), tf.float32)
    og_size[0] /= frames
    frames = tf.reshape(image_decoded, [-1, int(og_size[0]), int(og_size[1]), 3])
    image_resized = tf.image.resize_images(frames, (target_size, target_size))
    return tf.subtract(tf.math.divide(image_resized, 127.5), 1.0)


def tf_dataset(dir, frames, target_size, batch_size):
    """
    Converts videos/images to tensorflow dataset.
    """
    videos = glob.glob(os.path.join(dir, '*.avi'))
    images = glob.glob(os.path.join(dir, '*.jpg'))
    if len(videos) > len(images):
        vid_to_img(dir, frames)
        images = glob.glob(os.path.join(dir, '*.jpg'))

    # TODO: Pad batches instead
    num_use = int(len(images) / batch_size) * batch_size
    images_to_use = [str(path) for path in images[:num_use]]

    og_img_size = list(cv2.imread(images_to_use[0]).shape[:2])

    # Construct dataset.
    dataset = tf.data.Dataset.from_tensor_slices(images_to_use).shuffle(2*len(images_to_use))
    dataset = dataset.map(lambda x: parse_video(x, target_size, og_img_size, frames)).batch(batch_size)

    return dataset