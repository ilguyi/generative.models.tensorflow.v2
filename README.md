# Generative models with tensorflow version 2.0 style
* Final update: 2019. 05. 11.
* All right reserved @ Il Gu Yi 2019

This repository is a collection of various generative models (GAN, VAE, Normalizing flow, Autoregressive models, etc)
implemented by TensorFlow version 2.0 style


## Getting Started

### Prerequisites
* [`TensorFlow`](https://www.tensorflow.org) 2.0 or above 1.13
* Python 3.6
* Python libraries:
  * `numpy`, `matplotlib`, `PIL`, `imageio`
  * `urllib`, `zipfile`
* TensorFlow libraries & extensions:
  * [`tensorflow_probability`](https://www.tensorflow.org/probability/)
* Jupyter notebook
* OS X and Linux (Not validated on Windows OS)


## Contents

### Generative Adversarial Networks (GANs) [with MNIST and Fashion MNIST]

#### DCGAN (Deep Convolutional GAN)
* Unsupervised Representation Learning with Deep Convolutional
Generative Adversarial Networks paper [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
* [dcgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/dcgan.ipynb)

| *MNIST* | *Fashion MNIST* |
|---|---|
| <img src='https://user-images.githubusercontent.com/11681225/56466492-f3612480-644d-11e9-9e8f-0e3a63b5036d.gif'> | <img src='https://user-images.githubusercontent.com/11681225/56466506-0f64c600-644e-11e9-8ff9-4d003285a234.gif'> |


#### Conditional GAN
* Conditional Generative Adversarial Nets [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
* [cgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/cgan.ipynb)

| | |
|---|---|
| *MNIST* | <img src='https://user-images.githubusercontent.com/11681225/56466681-c7df3980-644f-11e9-9bba-334bdf73e496.gif'> |
| *Fashion MNIST* | <img src='https://user-images.githubusercontent.com/11681225/56466680-c746a300-644f-11e9-8b00-1907712f14d7.gif'> |


#### LSGAN
* Least Squares Generative Adversarial Networks [arXiv:1611.04076](https://arxiv.org/abs/1611.04076)
* [lsgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/lsgan.ipynb)

| *MNIST* | *Fashion MNIST* |
|---|---|
| <img src='https://user-images.githubusercontent.com/11681225/56466700-17be0080-6450-11e9-8b28-9338bbc2f632.gif'> | <img src='https://user-images.githubusercontent.com/11681225/56466699-17be0080-6450-11e9-920c-930a6e2f1b63.gif'> |


#### BiGAN
* Adversarial Feature Learning [arXiv:1605.09782](https://arxiv.org/abs/1605.09782)
* [bigan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/bigan.ipynb)

| | |
|---|---|
| *MNIST* | <img src='https://user-images.githubusercontent.com/11681225/56466717-4a67f900-6450-11e9-9f3c-64939d1c31b6.gif'> |
| *Fashion MNIST* | <img src='https://user-images.githubusercontent.com/11681225/56466716-4a67f900-6450-11e9-9035-1547ed4a9d59.gif'> |


#### Wasserstein GAN
* Wasserstein GAN [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
* [wgan.ipynb](https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/wgan.ipynb)

| *MNIST* | *Fashion MNIST* |
|---|---|
| <img src='https://user-images.githubusercontent.com/11681225/56466733-77b4a700-6450-11e9-860f-39c8f7acfb83.gif'> | <img src='https://user-images.githubusercontent.com/11681225/56466732-77b4a700-6450-11e9-804d-3c5154b68d89.gif'> |



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

| | |
|---|---|
| *MNIST* | <img src='https://user-images.githubusercontent.com/11681225/55270473-55f95180-52e2-11e9-8671-bd12983d53f4.gif'> |
| *Fashion MNIST* | <img src='https://user-images.githubusercontent.com/11681225/57566079-3f770780-7403-11e9-8aae-19d4df0dbb39.gif'> |



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
[Notion link](https://www.notion.so/Generative-models-620a774dc63143ddbe168fac4dbc423b)
