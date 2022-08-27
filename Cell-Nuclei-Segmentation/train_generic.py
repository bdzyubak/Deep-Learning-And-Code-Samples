import os
import numpy as np
import tensorflow as tf
import sys
from shared_utils.prep_training_data import ImgMaskDataset
from shared_utils.model_initializer import InitializeModel
top_path = os.path.dirname(__file__)

H = 256
W = 256

def main(): 
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    dataset = ImgMaskDataset(os.path.join(top_path,'data'))
    dataset.prep_data_img_labels()
    dataset.set_max_batch_size()

    model_name = 'unet'
    model_path = os.path.join(os.path.dirname(__file__),"trained_model")
    model = InitializeModel(model_name,dataset,model_path)
    history = model.run_model()
    model.try_bypass_local_minimum(n_times=3)

if __name__ == "__main__":
    main()
