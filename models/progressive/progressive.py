"""
progressive.py
Basic progressive generator.
"""

import os
import numpy as np
import time
import tensorflow as tf
import tensorflow.keras.layers as kl

from tensorflow.python import pywrap_tensorflow

from utils.process_out import convert_image, write_avi


class ProgressiveModel():

    def __init__(self,
                 batch_size,
                 z_dim,
                 num_frames,
                 conv_init,
                 disc_iterations,
                 gen_iterations,
                 save_checkpts=True,
                 verbose=True,
                 **kwargs):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.num_frames = num_frames
        self.conv_init = conv_init
        self.disc_iterations = disc_iterations
        self.gen_iterations = gen_iterations
        self.save_checkpts = save_checkpts
        self.verbose = verbose
        self.fade = None

    def init_models(self, start_size, lr, b1, b2):
        # Build models
        self.generator = self.generator_model(start_size)
        self.discriminator = self.discriminator_model(start_size)

        if self.verbose:
            print(self.generator.summary())
            print(self.discriminator.summary())

        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2)
        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                              discriminator_optimizer=self.disc_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def train_step(self, videos, fade):
        # Generate noise from normal distribution
        noise = tf.random_normal([self.batch_size, self.z_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_videos = self.generator(inputs=[noise, fade], training=True)

            real_disc = self.discriminator(inputs=[videos, fade], training=True)
            generated_disc = self.discriminator(inputs=generated_videos, training=True)
            print('Disc scores real/fake:', tf.reduce_mean(real_disc), tf.reduce_mean(generated_disc))

            gen_loss = self.generator_loss(generated_disc)
            print('Gen loss:', gen_loss)
            disc_loss = self.discriminator_loss(videos, generated_videos, real_disc, generated_disc)
            print('Disc loss:', disc_loss)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        for iter in range(self.gen_iterations):
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        for iter in range(self.disc_iterations):
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, videos, vid_size, start_size, epochs, lr, save_dir, b1, b2, save_int, num_out, **kwargs):
        # Load from checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if latest_checkpoint:
            print('Loading checkpoint ' + latest_checkpoint)
            loop_start_size = start_size * 2**(int(os.path.basename(latest_checkpoint)[4])-1)
        else:
            loop_start_size = start_size
        self.init_models(loop_start_size, lr, b1, b2)
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)

        # Number of progressive resolution stages
        resolutions = int(np.log2(vid_size/loop_start_size)) + 1

        for resolution in range(resolutions):
            print('Resolution: ', loop_start_size*2**resolution)
            self.fade = True if (resolution > 0 or loop_start_size > start_size) else False
            for epoch in range(epochs):
                start = time.time()

                for batch in videos:
                    if resolution < resolutions - 1:
                        # Set floor for frames such that min value is 1/2 of frame size
                        pool_factor = 2**(resolutions - resolution - 1)
                        batch = kl.AveragePooling3D(
                            (min(pool_factor, batch.get_shape().as_list()[1]/(loop_start_size*2**(resolution-1))),
                             pool_factor, pool_factor), padding='same')(batch)
                    fade = epoch/(epochs//2)
                    if fade == 1:
                        self.fade = False
                    self.train_step(videos=tf.cast(batch, tf.float32),
                                    fade=tf.constant(fade, shape=(self.batch_size, 1), dtype=tf.float32))

                # Save every n intervals
                if (epoch + 1) % save_int == 0:
                    self.generate(epoch + 1, save_dir, num_out)
                    if self.save_checkpts:
                        self.checkpoint.save(
                            file_prefix=os.path.join(save_dir,
                                                     "ckpt" + str(resolution+np.log2(loop_start_size/start_size)+1)))

                print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

            if resolution < resolutions - 1:
                print('Updating models to add new layers for next resolution.')
                self.update_models(loop_start_size*2**resolution, start_size)

        # Generate samples after final epoch
        self.generate(epochs, save_dir, num_out)

    def generator_model(self, out_size, start_size=8, start_filters=512):

        # Fading function
        def blend_resolutions(upper, lower, alpha):
            upper = tf.multiply(upper, alpha)
            lower = tf.multiply(lower, tf.subtract(1, alpha))
            return kl.Add()([upper, lower])

        # For now we start at 4x8x8 and upsample by 2x each time, e.g. 8x16x16
        conv_loop = int(np.log2(out_size/start_size))

        z = kl.Input(shape=(self.z_dim,))
        fade = kl.Input(shape=(1,))

        # First resolution
        x = kl.Dense(start_filters * start_size**2 * start_size/2,
                     kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                     name='dense')(z)
        x = kl.Reshape((int(start_size/2), start_size, start_size, start_filters))(x)
        x = kl.BatchNormalization()(x)
        x = kl.ReLU()(x)

        lower_res = None
        for resolution in range(conv_loop):
            filters = max(start_filters // 2**(resolution+1), 4)
            x = kl.Conv3DTranspose(filters=filters, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=self.conv_init, use_bias=True,
                                   name='conv_'+str(start_size*2**(resolution+1))+'_'+str(filters))(x)
            x = kl.BatchNormalization()(x)
            x = kl.ReLU()(x)
            if resolution == conv_loop - 1 and conv_loop > 1:
                lower_res = x

        # Conversion to 3-channel color
        # This is explicitly defined so we can reuse it for the upsampled lower-resolution frames as well
        convert_to_image = kl.Conv3DTranspose(filters=3, kernel_size=1, strides=1, padding='same',
                                              kernel_initializer=self.conv_init, use_bias=True, activation='tanh',
                                              name='conv_to_img_'+str(x.get_shape().as_list()[-1]))
        x = convert_to_image(x)

        # Fade output of previous resolution stage into final resolution stage
        if self.fade and lower_res:
            lower_upsampled = kl.UpSampling3D()(lower_res)
            lower_upsampled = convert_to_image(lower_upsampled)
            x = kl.Lambda(lambda x, y, alpha: blend_resolutions(x, y, alpha))([x, lower_upsampled, fade])

        return tf.keras.models.Model(inputs=[z, fade], outputs=x, name='generator')

    def discriminator_model(self, out_size, max_filters=512):

        # Fading function
        def blend_resolutions(upper, lower, alpha):
            upper = tf.multiply(upper, alpha)
            lower = tf.multiply(lower, tf.subtract(1.0, alpha)[..., tf.newaxis, tf.newaxis, tf.newaxis])
            return kl.Add()([upper, lower])

        conv_loop = int(np.log2(out_size)) - 3

        vid = kl.Input(shape=(out_size/2, out_size, out_size, 3,))
        fade = kl.Input(shape=(1,))

        # Convert from RGB frames
        start_filters = int(max(max_filters/2**conv_loop, 4))
        converted = kl.Conv3D(filters=int(start_filters/2), kernel_size=1, strides=1, padding='same',
                      kernel_initializer=self.conv_init, use_bias=True, name='conv_from_img_'+str(out_size))(vid)
        # First convolution downsamples by factor of 2
        x = kl.Conv3D(filters=start_filters, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=self.conv_init,
                      name='conv_'+str(out_size/2)+'_'+str(start_filters))(converted)

        # Calculate discriminator score using alpha-blended combination of new discriminator layer outputs
        # versus downsampled version of input videos
        if self.fade:
            downsampled = kl.AveragePooling3D(pool_size=(2, 2, 2), padding='same')(converted)
            x = kl.Lambda(lambda args: blend_resolutions(args[0], args[1], args[2]))([x, downsampled, fade])
        x = kl.Lambda(lambda x: tf.contrib.layers.layer_norm(x))(x)
        x = kl.LeakyReLU(.2)(x)

        for resolution in range(conv_loop):
            filters = start_filters * 2**(resolution + 1)
            x = kl.Conv3D(filters=filters, kernel_size=4, strides=2, padding='same',
                          kernel_initializer=self.conv_init,
                          name='conv_' + str(out_size / 2**(resolution+2))+'_'+str(filters))(x)
            x = kl.Lambda(lambda x: tf.contrib.layers.layer_norm(x))(x)
            x = kl.LeakyReLU(.2)(x)

        # Convert to single value
        x = kl.Conv3D(filters=1, kernel_size=4, strides=2, padding='same',
                      kernel_initializer=self.conv_init, name='conv_1')(x)
        x = kl.LeakyReLU(.2)(x)
        x = kl.Flatten()(x)
        x = kl.Dense(1, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                     name='dense_'+str(x.get_shape().as_list()[-1]))(x)

        return tf.keras.models.Model(inputs=[vid, fade], outputs=x, name='discriminator')

    def update_models(self, size, start_size=4):
        # Updates generator and discriminator models to add new layers corresponding to next resolution size
        # Retains weights previously learned from lower resolutions
        new_size = size*2
        new_generator, new_discriminator = self.generator_model(new_size), self.discriminator_model(new_size)
        new_gen_layers = [layer.name for layer in new_generator.layers]
        new_disc_layers = [layer.name for layer in new_discriminator.layers]
        for layer in self.generator.layers:
            if ('dense' in layer.name or 'conv' in layer.name) and layer.name in new_gen_layers:
                print('Updating ', layer.name)
                new_generator.get_layer(layer.name).set_weights(self.generator.get_layer(layer.name).get_weights())
        for layer in self.discriminator.layers:
            if ('dense' in layer.name or 'conv' in layer.name) and layer.name in new_disc_layers:
                print('Updating ', layer.name)
                new_discriminator.get_layer(layer.name).set_weights(self.discriminator.get_layer(layer.name).get_weights())
        self.generator, self.discriminator = new_generator, new_discriminator
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                              discriminator_optimizer=self.disc_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        if self.verbose:
            print(self.generator.summary())
            print(self.discriminator.summary())

    def plot_models(self, size, save_dir):
        # Plot model structure
        tf.keras.utils.plot_model(self.generator, show_shapes=True,
                                  to_file=os.path.join(save_dir, str(size) + '_gen.jpg'))
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True,
                                  to_file=os.path.join(save_dir, str(size) + '_disc.jpg'))

    def generator_loss(self, generated_disc):
        # WGAN-GP loss: https://arxiv.org/pdf/1704.00028.pdf
        # Negative so that gradient descent maximizes critic score received by generated output
        return -tf.reduce_mean(generated_disc)

    def discriminator_loss(self, real_videos, generated_videos, real_disc, generated_disc):
        # WGAN-GP loss: https://arxiv.org/pdf/1704.00028.pdf
        # Difference between critic scores received by generated output vs real video
        # Lower values mean that the real video samples are receiving higher scores, therefore
        # gradient descent maximizes discriminator accuracy
        out_size = real_videos.get_shape().as_list()
        d_cost = tf.reduce_mean(generated_disc) - tf.reduce_mean(real_disc)
        alpha = tf.random_uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
        )
        dim = out_size[1] * out_size[2]**2 * 3
        real = tf.reshape(real_videos, [self.batch_size, dim])
        fake = tf.reshape(generated_videos, [self.batch_size, dim])
        diff = fake - real
        # Real videos adjusted by randomly weighted difference between real vs generated
        interpolates = real + (alpha * diff)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            interpolates_reshaped = tf.reshape(interpolates, out_size)
            interpolates_disc = self.discriminator(interpolates_reshaped)
        # Gradient of critic score wrt interpolated videos
        gradients = tape.gradient(interpolates_disc, [interpolates])[0]
        # Euclidean norm of gradient for each sample
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        # Gradient norm penalty is the average distance from 1
        gradient_penalty = tf.reduce_mean((norm - 1.) ** 2)

        return d_cost + 10 * gradient_penalty

    def generate(self, epoch, save_dir, num_out):
        gen_noise = tf.random_normal([num_out, self.z_dim])
        output = self.generator([gen_noise, tf.constant(1, shape=[self.batch_size, 1])], training=False)
        frames = [convert_image(output[i, ...]) for i in range(output.get_shape().as_list()[0])]
        for out in range(num_out):
            write_avi(frames[out], save_dir, name=str(frames[out][0].shape[-2]) + '_' + str(epoch) + '_' + str(out))
