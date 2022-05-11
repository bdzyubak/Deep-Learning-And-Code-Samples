import os
import numpy as np
import tensorflow as tf
import sys
top_path = os.path.dirname(__file__)
utils_path = os.path.join(os.path.dirname(top_path),'shared_utils')
sys.path.append(utils_path)
from prep_training_data import ImgMaskDataset
from model_initializer import InitializeModel

H = 256
W = 256

def main(): 
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    dataset = ImgMaskDataset(os.path.join(top_path,'data'))
    dataset.prep_data_img_labels()

    model_name = 'unet'
    model_path = os.path.join(os.path.dirname(__file__),"trained_model")
    model = InitializeModel(model_name,dataset,model_path)
    history = model.run_model()
    model.try_bypass_local_minimum(n_times=2)

if __name__ == "__main__":
    main()
