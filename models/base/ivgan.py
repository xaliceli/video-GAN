"""
ivgan.py
iVGAN based on https://arxiv.org/pdf/1711.11453.pdf
"""
import os
import time
import tensorflow as tf
import tensorflow.python.keras.layers as kl

from utils.process_out import convert_image, write_avi

class iVGAN():

    def __init__(self,
                 input,
                 batch_size,
                 num_frames,
                 crop_size,
                 learning_rate,
                 z_dim,
                 conv_init,
                 beta1,
                 disc_iterations,
                 gen_iterations,
                 num_out,
                 epochs,
                 save_int,
                 save_dir,
                 save_checkpts=True):
        self.videos = input
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.conv_init = conv_init
        self.beta1 = beta1
        self.disc_iterations = disc_iterations
        self.gen_iterations = gen_iterations
        self.num_out = num_out
        self.epochs = epochs
        self.save_int = save_int
        self.save_dir = save_dir
        self.save_checkpts = save_checkpts

        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()

        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999)
        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                              discriminator_optimizer=self.disc_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.save_dir))

    def train_step(self, videos):
        # Generate noise from normal distribution
        noise = tf.random_normal([self.batch_size, self.z_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_videos = self.generator(noise, training=True)

            real_disc = self.discriminator(videos, training=True)
            generated_disc = self.discriminator(generated_videos, training=True)
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

    def train(self):
        # Plot model structure
        tf.keras.utils.plot_model(self.generator, show_shapes=True,
                                  to_file=os.path.join(self.save_dir, 'gen.jpg'))
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True,
                                  to_file=os.path.join(self.save_dir, 'disc.jpg'))

        # Generate noise from normal distribution
        for epoch in range(self.epochs):
            start = time.time()

            for batch in self.videos:
                self.train_step(batch)

            # Save every n intervals
            if (epoch + 1) % self.save_int == 0:
                self.generate(self.generator, epoch + 1, self.num_out)
                if self.save_checkpts:
                    self.checkpoint.save(file_prefix=os.path.join(self.save_dir, "ckpt"))

            print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                             time.time() - start))
        # Generate samples after final epoch
        self.generate(self.generator, self.epochs, self.num_out)

    def generator_model(self):
        model = tf.keras.Sequential()

        # Linear block
        model.add(kl.Dense(self.crop_size * 8 * 4 * 4 * 2, input_shape=(self.z_dim,),
                                        kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01)))
        model.add(kl.Reshape((2, 4, 4, self.crop_size * 8)))
        model.add(kl.BatchNormalization())
        model.add(kl.ReLU())

        # Convolution block 1
        model.add(kl.Conv3DTranspose(filters=self.crop_size * 4, kernel_size=4, strides=2, padding='same',
                                                  kernel_initializer=self.conv_init, use_bias=True))
        model.add(kl.BatchNormalization())
        model.add(kl.ReLU())

        # Convolution block 2
        model.add(kl.Conv3DTranspose(filters=self.crop_size * 2, kernel_size=4, strides=2, padding='same',
                                                  kernel_initializer=self.conv_init, use_bias=True))
        model.add(kl.BatchNormalization())
        model.add(kl.ReLU())

        # Convolution block 3
        model.add(kl.Conv3DTranspose(filters=self.crop_size, kernel_size=4, strides=2, padding='same',
                                                  kernel_initializer=self.conv_init, use_bias=True))
        model.add(kl.BatchNormalization())
        model.add(kl.ReLU())

        # Convolution block 4
        model.add(kl.Conv3DTranspose(filters=3, kernel_size=4, strides=2, padding='same',
                                                  kernel_initializer=self.conv_init, use_bias=True, activation='tanh'))

        return model

    def discriminator_model(self):
        model = tf.keras.Sequential()

        # Convolution block 1
        model.add(kl.Conv3D(filters=self.crop_size,
                                         input_shape=(self.num_frames, self.crop_size, self.crop_size, 3),
                                         kernel_size=4, strides=2, padding='same', kernel_initializer=self.conv_init))
        model.add(kl.Lambda(lambda x: tf.contrib.layers.layer_norm(x)))
        model.add(kl.LeakyReLU(.2))

        # Convolution block 2
        model.add(kl.Conv3D(filters=self.crop_size * 2, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=self.conv_init))
        model.add(kl.Lambda(lambda x: tf.contrib.layers.layer_norm(x)))
        model.add(kl.LeakyReLU(.2))

        # Convolution block 3
        model.add(kl.Conv3D(filters=self.crop_size * 4, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=self.conv_init))
        model.add(kl.Lambda(lambda x: tf.contrib.layers.layer_norm(x)))
        model.add(kl.LeakyReLU(.2))

        # Convolution block 4
        model.add(kl.Conv3D(filters=self.crop_size * 8, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=self.conv_init))
        model.add(kl.Lambda(lambda x: tf.contrib.layers.layer_norm(x)))
        model.add(kl.LeakyReLU(.2))

        # Convolution block 5
        model.add(kl.Conv3D(filters=1, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=self.conv_init))
        model.add(kl.LeakyReLU(.2))

        # Linear block
        model.add(kl.Flatten())
        model.add(kl.Dense(1, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01)))

        return model

    def generator_loss(self, generated_disc):
        # WGAN-GP loss: https://arxiv.org/pdf/1704.00028.pdf
        # Negative so that gradient descent maximizes critic score received by generated output
        return -tf.reduce_mean(generated_disc)

    def discriminator_loss(self, real_videos, generated_videos, real_disc, generated_disc):
        # WGAN-GP loss: https://arxiv.org/pdf/1704.00028.pdf
        # Difference between critic scores received by generated output vs real video
        # Lower values mean that the real video samples are receiving higher scores, therefore
        # gradient descent maximizes discriminator accuracy
        d_cost = tf.reduce_mean(generated_disc) - tf.reduce_mean(real_disc)
        alpha = tf.random_uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
        )
        dim = self.num_frames * self.crop_size * self.crop_size * 3
        real = tf.reshape(real_videos, [self.batch_size, dim])
        fake = tf.reshape(generated_videos, [self.batch_size, dim])
        diff = fake - real
        # Real videos adjusted by randomly weighted difference between real vs generated
        interpolates = real + (alpha * diff)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            interpolates_reshaped = tf.reshape(interpolates,
                                               [self.batch_size, self.num_frames,
                                                self.crop_size, self.crop_size, 3])
            interpolates_disc = self.discriminator(interpolates_reshaped)
        # Gradient of critic score wrt interpolated videos
        gradients = tape.gradient(interpolates_disc, [interpolates])[0]
        # Euclidean norm of gradient for each sample
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        # Gradient norm penalty is the average distance from 1
        gradient_penalty = tf.reduce_mean((norm - 1.) ** 2)

        return d_cost + 10 * gradient_penalty

    def generate(self, model, epoch, num_out):
        gen_noise = tf.random_normal([num_out, self.z_dim])
        output = model(gen_noise, training=False)
        frames = [convert_image(output[:, i, :, :, :], self.num_out) for i in range(self.num_frames)]
        write_avi(frames, self.save_dir, name=str(epoch) + '.avi')
