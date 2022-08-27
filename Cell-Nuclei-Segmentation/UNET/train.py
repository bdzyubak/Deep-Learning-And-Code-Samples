import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_coef, iou
from shared_utils.os_utils import make_new_dirs
from shared_utils.prep_training_data import ImgMaskDataset

# Based on this tutorial: https://morioh.com/p/11231a697d59
top_path = os.path.dirname(os.path.dirname(__file__))
H = 256
W = 256

def main(): 
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    dataset = ImgMaskDataset(os.path.join(top_path,'data'))
    dataset.prep_data_img_labels()

    """ Directory to save files """
    model_dir = os.path.join(os.path.dirname(__file__),"trained_model")
    make_new_dirs(model_dir)

    """ Hyperparaqmeters """
    batch_size = 8
    lr = 1e-4   ## 0.0001
    num_epochs = 10
    model_path = os.path.join(model_dir,"model.h5")
    csv_path = os.path.join(model_dir,"data.csv")
    
    # train_dataset, valid_dataset, train_steps, valid_steps = prep_data_img_labels(images,masks,batch_size)

    """ Model """
    model = build_unet((H, W, 3))
    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
