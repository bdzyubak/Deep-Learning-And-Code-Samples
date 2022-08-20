import os
# Install pix2pix
# os.system('pip install git+https://github.com/tensorflow/examples.git')
# Based on tutorial from: https://www.tensorflow.org/tutorials/images/segmentation

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

# from IPython.display import clear_output # For Jupyter only
import matplotlib.pyplot as plt

TRAIN_LENGTH = ''
STEPS_PER_EPOCH = ''
BATCH_SIZE = 64
BUFFER_SIZE = 1000
OUTPUT_CLASSES = 3
DISPLAY_SAMPLES = False
SAVE_PREDICTION_DEMOS = True
SAVE_PREDICTIONS_PREFIX = 'predictions_sample_epcoh_'
EPOCHS = 10
VAL_SUBSPLITS = 5

def main():  
  dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
  TRAIN_LENGTH = info.splits['train'].num_examples
  STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
  VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
  
  train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
  test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

  train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

  test_batches = test_images.batch(BATCH_SIZE)

  
  for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
  if DISPLAY_SAMPLES:   
    display([sample_image, sample_mask])
  
  base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
  layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
  ]
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  # Make a model from outputs of each layer of the encoder side
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

  down_stack.trainable = False

  # Build layers with matched input sizes to the output sizes of the encoder
  up_stack = [
      pix2pix.upsample(512, 3),  # 4x4 -> 8x8
      pix2pix.upsample(256, 3),  # 8x8 -> 16x16
      pix2pix.upsample(128, 3),  # 16x16 -> 32x32
      pix2pix.upsample(64, 3),   # 32x32 -> 64x64
  ]

  # Add skip connections and combine the encoder and decoder into the same model
  model = add_skip_connections(down_stack,up_stack,output_channels=OUTPUT_CLASSES)
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  tf.keras.utils.plot_model(model, to_file='unet_pix2pix_fusion_tutorial.png',show_shapes=True)

  sample_data = (sample_image,sample_mask)
  
  display_callback = DisplayCallback(model,sample_data)
  model_history = model.fit(train_batches, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_batches,
                            callbacks=[display_callback])

  plot_loss_history(model_history)

  show_predictions(model,(sample_image,sample_mask),test_batches, 3)
  print('Done.')

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

def display(display_list,save_file):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  if SAVE_PREDICTION_DEMOS and save_file: 
    plt.savefig(save_file) 
  if DISPLAY_SAMPLES: 
    plt.show(block=False)

def add_skip_connections(down_stack,up_stack,output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(model,sample_data, num=1,save_file=''):
  if tf.is_tensor(sample_data):
    for image, mask in sample_data.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)],save_file)
  elif isinstance(sample_data,tuple):
    sample_image = sample_data[0]
    sample_mask = sample_data[1]
    display([sample_image, sample_mask,
              create_mask(model.predict(sample_image[tf.newaxis, ...]))],save_file)
  else: 
    raise(ValueError('Unknown type of sample data. Accepts tuple or tf dataset.'))
      
class DisplayCallback(tf.keras.callbacks.Callback):
  def __init__(self,model,sample_data): 
    self.model = model
    self.sample_data = sample_data
    show_predictions(model,sample_data,save_file=SAVE_PREDICTIONS_PREFIX+'_init_'+'.png')
  
  def on_epoch_end(self, epoch, logs=None):
    # clear_output(wait=True)
    show_predictions(self.model,self.sample_data,save_file=SAVE_PREDICTIONS_PREFIX+str(epoch)+'.png')

    print('\nSample Prediction after epoch {}\n'.format(epoch+1))

def plot_loss_history(model_history): 
  loss = model_history.history['loss']
  val_loss = model_history.history['val_loss']

  plt.figure()
  plt.plot(model_history.epoch, loss, 'r', label='Training loss')
  plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss Value')
  plt.ylim([0, 1])
  plt.legend()
  plt.show()

if __name__ == '__main__': 
    main()