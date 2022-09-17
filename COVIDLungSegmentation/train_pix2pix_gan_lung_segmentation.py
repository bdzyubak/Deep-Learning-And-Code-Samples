import os
import numpy as np
import tensorflow as tf
import shutil
from prep_training_data import ImgMaskDataset
from model_initializer import InitializeModel
from os_utils import list_dir, make_new_dirs, list_directories, enclose_in_quotes, add_dir_reenclose
from web_utils import download_if_not_exist 
top_path = os.path.dirname(__file__)
dataset_name = "covidqu"
data_path = os.path.join(top_path,'data')


H = 256
W = 256

def main(): 
    download_if_not_exist(data_path,url="https://www.kaggle.com/datasets/anasmohammedtahir/covidqu") 
    dataset_path, path_images, path_masks = get_path_lung_segm_normal(data_path) # Do normal only for now; change variable name from global
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    dataset = ImgMaskDataset(dataset_path,path_images,path_masks)
    dataset.prep_data_img_labels()

    model_name = 'unet'
    model_path = os.path.join(os.path.dirname(__file__),"trained_model")
    model = InitializeModel(model_name,dataset,model_path)
    model.run_model()

def get_path_lung_segm_normal(data_path): 
    # Function to dig down to unusual data locations for the covidqu dataset
    dataset_path = os.path.join(data_path, "Lung Segmentation Data","Lung Segmentation Data"
    ,"Train","Normal")
    path_images = os.path.join(dataset_path,'images')
    path_masks = os.path.join(dataset_path,'lung masks')
    return dataset_path, path_images, path_masks

if __name__ == "__main__":
    main()
