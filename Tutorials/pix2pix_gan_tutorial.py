import tensorflow as tf

import os
import pathlib
import time
import datetime
from os_utils import list_dir

from matplotlib import pyplot as plt
from IPython import display

# Implementation of the Pix2Pix GAN tutorial from the Tensorflow team, with some refactoring: https://www.tensorflow.org/tutorials/generative/pix2pix

# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100
show_samples = False
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
dataset_name = "facades"

_URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'

path_to_zip = tf.keras.utils.get_file(
    fname=f"{dataset_name}.tar.gz",
    origin=_URL,
    extract=True)

path_to_zip  = pathlib.Path(path_to_zip)
PATH = path_to_zip.parent/dataset_name
list(PATH.parent.iterdir())

def main(): 
  @tf.function
  def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = generator(input_image, training=True)

      disc_real_output = discriminator([input_image, target], training=True)
      disc_generated_output = discriminator([input_image, gen_output], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(gen_output, disc_generated_output, target)
      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
      tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
      tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
      tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
  
  def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
      if (step) % 1000 == 0:
        display.clear_output(wait=True)

        if step != 0:
          print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

        start = time.time()
        
        # Every 1000 steps, show an example of input, generated image, and target
        generate_images(generator, example_input, example_target)
        print(f"Step: {step//1000}k")

      train_step(input_image, target, step)

      # Training step
      if (step+1) % 10 == 0:
        print('.', end='', flush=True)

      # Save (checkpoint) the model every 5k steps
      if (step + 1) % 5000 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
  
  inp = load_sample_image(display=show_samples)
    
  train_dataset, test_dataset = load_preprocess_data()

  test_downsample_upsample(inp)

  generator = Generator()
  tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

  gen_output = generator(inp[tf.newaxis, ...], training=False) # Same as generator.predict(...)
  plt.imshow(gen_output[0, ...])

  discriminator = Discriminator()
  tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

  disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False) # Same as discriminator.predict(...)
  plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
  plt.colorbar()

  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator) 

  for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)
  log_dir="logs/"

  summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  
  fit(train_dataset, test_dataset, steps=40000)


def load_sample_image(display=True):     
    file = str(PATH / 'train/1.jpg')
    sample_image = tf.io.read_file(file)
    sample_image = tf.io.decode_jpeg(sample_image)
    print(sample_image.shape)
    
    inp, re = load_split_halves(file)
    if not display: 
      return inp
    plt.figure()
    plt.imshow(sample_image)
    plt.axis('off')

    
    # Casting to int for matplotlib to display the images
    plt.figure()
    plt.imshow(inp / 255.0)
    plt.figure()
    plt.imshow(re / 255.0)

    plt.figure(figsize=(6, 6))
    for i in range(4):
        rj_inp, rj_re = random_jitter(inp, re)
        plt.subplot(2, 2, i + 1)
        plt.imshow(rj_inp / 255.0)
        plt.axis('off')
    plt.show()
    return inp, re

def load_split_halves(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  w = w // 2
  label_image = image[:, w:, :]
  real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  label_image = tf.cast(label_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return label_image, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5: # Make a random number at runtime
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

# def load_image_train(image_file):
#   input_image, real_image = load_split_halves(image_file)
#   input_image, real_image = random_jitter(input_image, real_image)
#   input_image, real_image = normalize(input_image, real_image)

#   return input_image, real_image

# def load_image_test(image_file):
#   input_image, real_image = load_split_halves(image_file)
#   input_image, real_image = resize(input_image, real_image,
#                                    IMG_HEIGHT, IMG_WIDTH)
#   input_image, real_image = normalize(input_image, real_image)

#   return input_image, real_image

def load_preprocess_image(image_file,type):
  input_image, real_image = load_split_halves(image_file)
  if type == 'train': 
    input_image, real_image = random_jitter(input_image, real_image)
  else: 
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_preprocess_data(): 
    train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
    train_dataset = train_dataset.map(lambda x: load_preprocess_image(x, 'train'),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
    test_dataset = test_dataset.map(lambda x: load_preprocess_image(x,'test'))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset, test_dataset

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def test_downsample_upsample(inp): 
  down_model = downsample(3, 4)
  down_result = down_model(tf.expand_dims(inp, 0))
  print(down_result.shape)

  up_model = upsample(3, 4)
  up_result = up_model(down_result)
  print(up_result.shape)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, apply_batchnorm=False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  conv = tf.keras.layers.Conv2D(512, 4, strides=1, padding='same',
                                kernel_initializer=initializer,
                                use_bias=False)(down3)  # (batch_size, 32, 32, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  last = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same',
                                kernel_initializer=initializer)(leaky_relu)  # (batch_size, 32, 32, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
  # Try to get the generator to make an image (probabilty map of real-ness) for a real image
  # while making an array of zeros for a fake image, weighing True/False positives equally
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(gen_output, disc_generated_output, target):
  # Try to make the discriminator output an array of ones for a fake image (i.e. think it is very real)
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output) 

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) 
  # Additionally, force the generator to be close to original images. 
  # L2 or L1 can be used depending on whether spike features or blurring are less problematic
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

if __name__ == '__main__': 
    main()