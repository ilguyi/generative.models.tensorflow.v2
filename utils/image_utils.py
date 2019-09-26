from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import PIL
import imageio


MNIST_SIZE = 28
CIFAR10_SIZE = 32
num_examples_to_generate = 25
checkpoint_dir = './train'


def print_or_save_sample_images(sample_images, max_print_size=num_examples_to_generate,
                                is_square=False, is_save=False, epoch=None,
                                checkpoint_dir=checkpoint_dir):
  available_print_size = list(range(1, 26))
  assert max_print_size in available_print_size
  if len(sample_images.shape) == 2:
    size = int(np.sqrt(sample_images.shape[1]))
  elif len(sample_images.shape) > 2:
    size = sample_images.shape[1]
    channel = sample_images.shape[3]
  else:
    ValueError('Not valid a shape of sample_images')
  
  if not is_square:
    print_images = sample_images[:max_print_size, ...]
    print_images = print_images.reshape([max_print_size, size, size, channel])
    print_images = print_images.swapaxes(0, 1)
    print_images = print_images.reshape([size, max_print_size * size, channel])
    if channel == 1:
      print_images = np.squeeze(print_images, axis=-1)

    fig = plt.figure(figsize=(max_print_size, 1))
    plt.imshow(print_images * 0.5 + 0.5)#, cmap='gray')
    plt.axis('off')
    
  else:
    num_columns = int(np.sqrt(max_print_size))
    max_print_size = int(num_columns**2)
    print_images = sample_images[:max_print_size, ...]
    print_images = print_images.reshape([max_print_size, size, size, channel])
    print_images = print_images.swapaxes(0, 1)
    print_images = print_images.reshape([size, max_print_size * size, channel])
    print_images = [print_images[:,i*size*num_columns:(i+1)*size*num_columns] for i in range(num_columns)]
    print_images = np.concatenate(tuple(print_images), axis=0)
    if channel == 1:
      print_images = np.squeeze(print_images, axis=-1)
    
    fig = plt.figure(figsize=(num_columns, num_columns))
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.imshow(print_images * 0.5 + 0.5)#, cmap='gray')
    plt.axis('off')
    
  if is_save and epoch is not None:
    filepath = os.path.join(checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(filepath)
  else:
    plt.show()



def print_or_save_sample_images_two(sample_images1, sample_images2, max_print_size=num_examples_to_generate,
                                    is_square=False, is_save=False, epoch=None,
                                    checkpoint_dir=checkpoint_dir):
  available_print_size = list(range(1, 26))
  assert max_print_size in available_print_size

  if len(sample_images1.shape) == 2:
    size = int(np.sqrt(sample_images1.shape[1]))
  elif len(sample_images1.shape) > 2:
    size = sample_images1.shape[1]
    channel = sample_images1.shape[3]
  else:
    ValueError('Not valid a shape of sample_images')
  
  if not is_square:
    print_images1 = sample_images1[:max_print_size, ...]
    print_images1 = print_images1.reshape([max_print_size, size, size, channel])
    print_images1 = print_images1.swapaxes(0, 1)
    print_images1 = print_images1.reshape([size, max_print_size * size, channel])

    print_images2 = sample_images2[:max_print_size, ...]
    print_images2 = print_images2.reshape([max_print_size, size, size, channel])
    print_images2 = print_images2.swapaxes(0, 1)
    print_images2 = print_images2.reshape([size, max_print_size * size, channel])

    print_images = np.concatenate((print_images1, print_images2), axis=0)
    if channel == 1:
      print_images = np.squeeze(print_images, axis=-1)
     
    plt.figure(figsize=(max_print_size, 2))
    plt.axis('off')
    plt.imshow(print_images)#, cmap='gray')
  else:
    print('This function is supported by `is_square=False` mode.')

  if is_save and epoch is not None:
    filepath = os.path.join(checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(filepath)
  else:
    plt.show()



def print_or_save_sample_images_pix2pix(x, y, z, model_name, name=None,
                                        is_save=False, epoch=None, checkpoint_dir=checkpoint_dir):
  #plt.figure(figsize=(15, 5))
  plt.figure(figsize=(12, 4))
  plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

  display_list = [x[0], y[0], z[0]]
  assert model_name in ['pix2pix', 'cyclegan', 'stargan']
  if model_name == 'pix2pix':
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
  elif model_name == 'cyclegan':
    assert name in ['X2Y2X', 'Y2X2Y']
    if name == 'X2Y2X':
      title = ['X domain', 'X -> Y', 'X -> Y -> X']
    else:
      title = ['Y domain', 'Y -> X', 'Y -> X -> Y']
  else:
    title = ['original domain', 'original -> target', 'original -> target -> original']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')

  if is_save and epoch is not None:
    if name is not None:
      filename = 'image_' + name + '_at_epoch_'
    else:
      filename = 'image_at_epoch_'
    filepath = os.path.join(checkpoint_dir, filename + '{:04d}.png'.format(epoch))
    plt.savefig(filepath)
  else:
    plt.show()




def display_image(epoch_no, name=None, checkpoint_dir=checkpoint_dir):
  if name is not None:
    filename = 'image_' + name + '_at_epoch_'
  else:
    filename = 'image_at_epoch_'
  filepath = os.path.join(checkpoint_dir, filename + '{:04d}.png'.format(epoch_no))
  return PIL.Image.open(filepath)



def generate_gif(gif_filename, checkpoint_dir=checkpoint_dir):
  with imageio.get_writer(gif_filename, mode='I') as writer:
    filenames = glob.glob(os.path.join(checkpoint_dir, 'image*.png'))
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
      frame = 2*(i**0.5)
      if round(frame) > round(last):
        last = frame
      else:
        continue
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
  
  # this is a hack to display the gif inside the notebook
  filename_to_copy = gif_filename + '.png'
  print('cp {} {}'.format(gif_filename, filename_to_copy))
  os.system('cp {} {}'.format(gif_filename, filename_to_copy))


