import os
import numpy as np
import tensorflow as tf
import sys
from shared_utils.prep_training_data import ImgMaskDataset
from shared_utils.model_initializer import InitializeModel
top = os.path.dirname(__file__)
H = 256
W = 256

def main(): 
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    dataset = ImgMaskDataset(os.path.join(top,'data'))
    dataset.prep_data_img_labels()

    model_name = 'unet'
    model_path = os.path.join(top,"trained_model")
    model = InitializeModel(model_name,dataset,model_path)
    model.run_model()

if __name__ == "__main__":
    main()
