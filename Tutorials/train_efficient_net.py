import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import * 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from os_utils import make_new_dirs

# Made for Kaggle Pathology Slide dataset: https://www.kaggle.com/code/taylorkern/tk-efficientnet

model_dir = os.path.join(os.path.dirname(__file__),"trained_model")
data_dir = os.path.join(top_path,'data')
make_new_dirs(model_dir,clean_subdirs=False)

""" Hyperparaqmeters """
batch_size = 8
lr = 1e-4   ## 0.0001
num_epochs = 10
model_path = os.path.join(model_dir,"model.h5")
csv_path = os.path.join(model_dir,"data.csv")
BATCH_SIZE = 64

def main(): 
    train = pd.read_csv(os.path.join(data_dir,'train_labels.csv'), dtype=str)
    print(train.shape)
    train.id = train.id + '.tif'

    # Visualize positive/negative label split
    train['label'].value_counts() / len(train)
    # Balanced set with slightly more negatives 

    train_path = os.path.join(data_dir,"train")

    ## Grab a random sample of the data
    # sample = train.sample(n=16).reset_index()
    # visualize_data(data_dir, sample)

    # Use straitify to match proportion of positives/negatives in population
    train_df, valid_df = train_test_split(train, test_size=0.2, random_state=1, stratify=train.label) 

    train_loader, TR_STEPS = make_random_data_generator(train_path, train_df)
    valid_loader, VA_STEPS = make_random_data_generator(train_path, valid_df)

    print(TR_STEPS)
    print(VA_STEPS)

    # Use transfer learning
    base_model = EfficientNetB0(input_shape=(96,96,3), include_top=False, weights='imagenet')

    # Try both options - not training the base model, just the prediction layer (transfer learning), and training all, just using imagenet weights as starting points
    base_model.trainable = False

    cnn = Sequential([
        base_model,
        
        Flatten(),
        
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(2, activation='softmax')
    ])

    cnn.summary()

    opt = tf.keras.optimizers.Adam(lr)
    cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC()])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
    ]

    # Use steps per epoch rather than batch size since random perumtations of the training data are used, rather than all training data
    h1 = cnn.fit(
        x = train_loader, 
        steps_per_epoch = TR_STEPS, 
        epochs = 40,
        validation_data = valid_loader, 
        validation_steps = VA_STEPS, 
        verbose = 1, 
        callbacks=callbacks
    )

def make_random_data_generator(path, df):
    datagen = ImageDataGenerator(rescale=1/255)
    # Make a training data loader which yields random perumtations of training data with a given batch size
    loader = datagen.flow_from_dataframe(
        dataframe = df,
        directory = path,
        x_col = 'id',
        y_col = 'label',
        batch_size = BATCH_SIZE,
        seed = 1,
        shuffle = True,
        class_mode = 'categorical',
        target_size = (96,96)
    )
    
    num_steps = len(loader)
    return loader, num_steps

def visualize_data(top_path, sample):
    plt.figure(figsize=(6,6))

    for i, row in sample.iterrows():
        img = plt.imread(os.path.join(top_path,'train',row['id']))
        label = row.label

        plt.subplot(4,4,i+1)
        plt.imshow(img)
        plt.text(0, -5, f'Class {label}', color='k')
        
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def merge_history(hlist):
    history = {}
    for k in hlist[0].history.keys():
        history[k] = sum([h.history[k] for h in hlist], [])
    return history

def vis_training(h, start=1):
    epoch_range = range(start, len(h['loss'])+1)
    s = slice(start-1, None)

    plt.figure(figsize=[14,4])

    n = int(len(h.keys()) / 2)

    for i in range(n):
        k = list(h.keys())[i]
        plt.subplot(1,n,i+1)
        plt.plot(epoch_range, h[k][s], label='Training')
        plt.plot(epoch_range, h['val_' + k][s], label='Validation')
        plt.xlabel('Epoch'); plt.ylabel(k); plt.title(k)
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 
    main()