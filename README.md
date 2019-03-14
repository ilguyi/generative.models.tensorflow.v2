# Generative models with tensorflow version 2.x style
* Final update: 2019. 03. 14.
* All right reserved @ Il Gu Yi 2019

This repository is a collection of various generative models (GAN, VAE, Flow based, etc)
implemented by TensorFlow version 2.0 style


## Getting Started

### Prerequisites
* [`TensorFlow`](https://www.tensorflow.org) above 1.12
* Python 3.6
* Python libraries:
  * `numpy`, `matplotlib`, `PIL`, `imageio`
  * `urllib`, `zipfile`
* Jupyter notebook
* OS X and Linux (Not validated on Windows OS)


## Contents

### Generative Adversarial Networks (GANs) [with MNIST]

#### DCGAN (Deep Convolutional GAN)
* Unsupervised Representation Learning with Deep Convolutional
Generative Adversarial Networks paper [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
* [dcgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/dcgan.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/54118813-43a09c00-4437-11e9-98e1-69b8668dd8c7.gif'>
</div>


### Conditional GAN
* Conditional Generative Adversarial Nets [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
* [cgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/cgan.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/54118866-5c10b680-4437-11e9-9908-3e98f3b8fcb3.gif'>
</div>


### LSGAN
* Least Squares Generative Adversarial Networks [arXiv:1611.04076](https://arxiv.org/abs/1611.04076)
* [lsgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/lsgan.ipynb)


### BiGAN
* Adversarial Feature Learning [arXiv:1605.09782](https://arxiv.org/abs/1605.09782)
* [bigan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/bigan.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/54355815-66cc8500-469d-11e9-975b-77e37a687f0f.gif'>
</div>


### Wasserstein GAN
* Wasserstein GAN [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
* [wgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/wgan.ipynb)



### Pix2Pix (Image Translation)
* Image-to-Image Translation with Conditional Adversarial Networks [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
* [pix2pix.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/pix2pix.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/51429242-195d0a00-1c50-11e9-8c11-1b19cf86eee8.gif'>
</div>


### CycleGAN (Unpaired Image Translation)
* Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
* [cyclegan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/cyclegan.ipynb)





### AutoRegressive Models [with MNIST]

#### Fully Visible Sigmoid Belief Networks
* [fvsbn.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/autoregressive/fvsbn.ipynb)




### Flow based Models [with MNIST]

#### NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION
* [nice.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/flow/nice.ipynb)



## Author
Il Gu Yi



