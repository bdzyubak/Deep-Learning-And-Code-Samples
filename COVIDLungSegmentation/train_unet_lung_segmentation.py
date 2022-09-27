import os
import numpy as np
import tensorflow as tf
from prep_training_data import ImgMaskDataset
from model_initializer import InitializeModel
from web_utils import download_if_not_exist 
from get_lung_segmentation_data import get_path_training_lung_segm_normal
top_path = os.path.dirname(__file__)
dataset_name = "covidqu"
data_path = os.path.join(top_path,'data')
H = 256
W = 256

def main(): 
    # tf.config.set_visible_devices([], 'GPU')
    download_if_not_exist(data_path,url="https://www.kaggle.com/datasets/anasmohammedtahir/covidqu") 
    training_data_paths = get_path_training_lung_segm_normal(data_path) # Do normal only for now; change variable name from global
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    dataset = ImgMaskDataset(training_data_paths)
    dataset.prep_data_img_labels()

    model_name = 'unet'
    model_path = os.path.join(os.path.dirname(__file__),"trained_model")
    model = InitializeModel(model_name,dataset,model_path)
    model.run_model()

if __name__ == "__main__":
    main()
