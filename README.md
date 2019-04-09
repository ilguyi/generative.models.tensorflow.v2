# Generative models with tensorflow version 2.0 style
* Final update: 2019. 04. 09.
* All right reserved @ Il Gu Yi 2019

This repository is a collection of various generative models (GAN, VAE, Normalizing flow, Autoregressive models, etc)
implemented by TensorFlow version 2.0 style


## Getting Started

### Prerequisites
* [`TensorFlow`](https://www.tensorflow.org) above 1.13
* Python 3.6
* Python libraries:
  * `numpy`, `matplotlib`, `PIL`, `imageio`
  * `urllib`, `zipfile`
* TensorFlow libraries & extensions:
  * [`tensorflow_probability`](https://www.tensorflow.org/probability/)
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


#### Conditional GAN
* Conditional Generative Adversarial Nets [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
* [cgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/cgan.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/55290776-80e2c300-5412-11e9-8ff8-340097be2375.gif'>
</div>


#### LSGAN
* Least Squares Generative Adversarial Networks [arXiv:1611.04076](https://arxiv.org/abs/1611.04076)
* [lsgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/lsgan.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/55290789-a7a0f980-5412-11e9-808f-d176012ca56c.gif'>
</div>


#### BiGAN
* Adversarial Feature Learning [arXiv:1605.09782](https://arxiv.org/abs/1605.09782)
* [bigan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/bigan.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/55290791-b8ea0600-5412-11e9-8683-57b84f1052c7.gif'>
</div>


#### Wasserstein GAN
* Wasserstein GAN [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
* [wgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/wgan.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/55290796-ce5f3000-5412-11e9-9618-36472625e2c3.gif'>
</div>


#### Pix2Pix (Image Translation)
* Image-to-Image Translation with Conditional Adversarial Networks [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
* [pix2pix.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/pix2pix.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/51429242-195d0a00-1c50-11e9-8c11-1b19cf86eee8.gif'>
</div>


#### CycleGAN (Unpaired Image Translation)
* Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
* [cyclegan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/cyclegan.ipynb)




### Latent Variable Models [with MNIST]

#### AutoEncoder (actually not generative model)
* [autoencoder.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/latentvariable/autoencoder.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/55270473-55f95180-52e2-11e9-8671-bd12983d53f4.gif'>
</div>


#### Denosing AutoEncoder
* [dae.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/latentvariable/dae.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/55270736-0ae13d80-52e6-11e9-9ca1-a6310336db7a.gif'>
</div>


### AutoRegressive Models [with MNIST]

#### Fully Visible Sigmoid Belief Networks
* [fvsbn.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/autoregressive/fvsbn.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/55416175-43646e00-55a9-11e9-9512-97970027e7fa.gif'>

</div>


#### Neural Autoregressive Density Estimation
* Neural Autoregressive Distribution Estimation [arXiv:1605.02226](https://arxiv.org/abs/1605.02226)
* [nade.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/autoregressive/nade.ipynb)
<div align="center">
<img src='https://user-images.githubusercontent.com/11681225/55416194-4d866c80-55a9-11e9-8ffe-ed7d3de47d31.gif'>
</div>


### Normalizing Flow Models [with MNIST]

#### NICE: Non-Linear Independent Components Estimation
* [nice.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/flow/nice.ipynb)



## Author
Il Gu Yi


### Slides
[Notion link](https://www.notion.so/Generative-models-4331d990702d40159cb163ff070f44e7)
