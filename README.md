# video-GAN

**Status: In Progress - Not Yet Validated**

This is a replication exercise for learning, an experiment in novel artistic outputs, and (hopefully, eventually) a research contribution to GANs as applied to video synthesis. The goal is to generate abstracted videos evoking the subjective affect of certain objects, actions, and scenes in motion, through generative adversarial networks trained on input videos of the desired subjects.

Generative models are based on ["Generating Videos with Scene Dynamics" (2016)](http://www.cs.columbia.edu/~vondrick/tinyvideo/paper.pdf) and ["Improving Video Generation for Multi-functional Applications" (2017)](https://arxiv.org/pdf/1711.11453.pdf). Code is based on the latter's [GitHub repo](https://github.com/bernhard2202/improved-video-gan/). Kratzwald's implementation appears to be a better fit for desired use case due to its ability to handle inputs without static backgrounds.

The creative part of this project is more nebulous for now but will require manipulating the generated videos such that they're able to be projected in a live setting paired with musical compositions. At minimum this will require interpolating the outputs which will be pretty small, or figuring out a way to generate larger outputs without significant runtime cost.

Using Google Colaboratory for TPU access. Will refactor into Python module once validated.
