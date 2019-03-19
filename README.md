# video-GAN

**Status: Training**

This is a replication exercise for learning, an experiment in novel artistic outputs, and (hopefully, eventually) a research contribution to generative adversarial networks as applied to video synthesis. The goal is to generate abstracted videos evoking the subjective affect of certain objects, actions, and scenes in motion, through GANs trained on input videos of the desired subjects.

## Overview

In its current form, this model learns to produce relatively small (64x64) videos using:
* 3D deconvolutional layers for generator, 3D convolutional layers for discriminator
* Batch normalization for generator, layer normalization for discriminator
* ReLU activations for generator, leaky ReLU activations for discriminator
* WGAN-GP loss function

Next steps include the following extensions (which will alter the current architecture substantially): 
* Investigating architectures allowing for much larger outputs
* Separating motion dynamics for motion transfer/modeling

Languages & libraries:
* Python 3.6.7
* Tensorflow 1.13.1
* Keras 2.2.4
* NumPy 1.14.6
* OpenCV 3.4.5.20

## Timeline

* v0: March 2019
  * First draft of model validated on small inputs, using Google Colab

## References

- ["Delving Deep Into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" (2015)](https://arxiv.org/pdf/1502.01852.pdf)
- ["Generating Videos with Scene Dynamics" (2016)](http://www.cs.columbia.edu/~vondrick/tinyvideo/paper.pdf)
- ["Improved Training of Wasserstein GANs" (2017)](https://arxiv.org/abs/1704.00028) [[code]](https://github.com/igul222/improved_wgan_training)
- ["Improving Video Generation for Multi-functional Applications" (2017)](https://arxiv.org/pdf/1711.11453.pdf) [[code]](https://github.com/bernhard2202/improved-video-gan/)
