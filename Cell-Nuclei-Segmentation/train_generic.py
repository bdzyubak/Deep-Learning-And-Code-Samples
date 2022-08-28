import os
import numpy as np
import tensorflow as tf
import sys
from shared_utils.prep_training_data import ImgMaskDataset
from shared_utils.model_initializer import InitializeModel
from shared_utils.web_utils import download_if_not_exist
top_path = os.path.dirname(__file__)
data_path = os.path.join(top_path,'data')
url = ''
H = 256
W = 256

def main(): 
    # download_if_not_exist(data_path,
    #     url="https://www.kaggle.com/datasets/84613660e1f97d3b23a89deb1ae6199a0c795ec1f31e2934527a7f7aad7d8c37") 
    # Automated download does not work - download manually and place images in masks in ./data
    
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
