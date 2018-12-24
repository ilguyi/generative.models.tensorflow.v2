from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import PIL
import imageio


MNIST_SIZE = 28
num_examples_to_generate = 16
checkpoint_dir = './train'


def print_or_save_sample_images(sample_images, max_print_size=num_examples_to_generate,
                                is_square=False, is_save=False, epoch=None,
                                checkpoint_dir=checkpoint_dir):
  available_print_size = list(range(1, 26))
  assert max_print_size in available_print_size
  
  if not is_square:
    print_images = sample_images[:max_print_size, ...]
    print_images = print_images.reshape([max_print_size, MNIST_SIZE, MNIST_SIZE])
    print_images = print_images.swapaxes(0, 1)
    print_images = print_images.reshape([MNIST_SIZE, max_print_size * MNIST_SIZE])

    fig = plt.figure(figsize=(max_print_size, 1))
    plt.imshow(print_images, cmap='gray')
    plt.axis('off')
    
  else:
    num_columns = int(np.sqrt(max_print_size))
    max_print_size = int(num_columns**2)
    print_images = sample_images[:max_print_size, ...]
    print_images = print_images.reshape([max_print_size, MNIST_SIZE, MNIST_SIZE])
    print_images = print_images.swapaxes(0, 1)
    print_images = print_images.reshape([MNIST_SIZE, max_print_size * MNIST_SIZE])
    print_images = [print_images[:,i*MNIST_SIZE*num_columns:(i+1)*MNIST_SIZE*num_columns] for i in range(num_columns)]
    print_images = np.concatenate(tuple(print_images), axis=0)
    
    fig = plt.figure(figsize=(num_columns, num_columns))
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.imshow(print_images, cmap='gray')
    plt.axis('off')
    
  if is_save and epoch is not None:
    filepath = os.path.join(checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(filepath)

  plt.show()


def display_image(epoch_no, checkpoint_dir=checkpoint_dir):
  filepath = os.path.join(checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch_no))
  return PIL.Image.open(filepath)



def generate_gif(gif_filename, checkpoint_dir=checkpoint_dir):
  with imageio.get_writer(gif_filename, mode='I') as writer:
    filenames = glob.glob(os.path.join(checkpoint_dir, 'image*.png'))
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
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


