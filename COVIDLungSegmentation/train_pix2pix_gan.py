import os
import numpy as np
import tensorflow as tf
import shutil
from prep_training_data import ImgMaskDataset
from model_initializer import InitializeModel
from os_utils import list_dir, make_new_dirs, list_directories, enclose_in_quotes, add_dir_reenclose
from web_utils import download_if_not_exist 
top_path = os.path.dirname(__file__)
data_path = enclose_in_quotes(os.path.join(top_path,'data'))
dataset_name = "covidqu"

H = 256
W = 256

def main(): 
    download_if_not_exist(data_path,url="https://www.kaggle.com/datasets/anasmohammedtahir") 
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    dataset = ImgMaskDataset()
    dataset.prep_data_img_labels()

    model_name = 'unet'
    model_path = os.path.join(os.path.dirname(__file__),"trained_model")
    model = InitializeModel(model_name,dataset,model_path)
    model.run_model()

def get_path_lung_segm_normal(data_path): 
    # Function to dig down to unusual data locations for the covidqu dataset
    path = add_dir_reenclose(data_path, dirs=["Lung Segmentation Data","Lung Segmentation Data"
    ,"Train","Normal"])
    return path

if __name__ == "__main__":
    main()
