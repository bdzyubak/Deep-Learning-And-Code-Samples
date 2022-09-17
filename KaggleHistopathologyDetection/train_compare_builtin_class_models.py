import os
import numpy as np
import tensorflow as tf
from prep_training_data import ImgLabelDataset
from model_initializer import InitializeModel
from web_utils import download_if_not_exist
top_path = os.path.dirname(__file__)
models_list = ['DenseNet121','efficientnetB0','efficientnet_v2B0','vgg16','inception','resnet50','resnet_v250',
'resnet_rs101','inception_resnet','regnetX002','regnetY002','mobilenet','mobilenet_v2','mobilenet_v3Small',
'mobilenet_v3Large','xception']
dataset_name = 'histopathologic-cancer-detection'
data_path = os.path.join(top_path,'data')

download_if_not_exist(os.path.join(top_path,'data')
    ,url='https://www.kaggle.com/competitions/histopathologic-cancer-detection/data'
    ,dataset_name=dataset_name)

def main(): 
    paths_training, paths_test = get_data_and_label_paths()

    """ Seeding """
    dataset = ImgLabelDataset(paths_training)

    for model_name in models_list: 
        print('Training ' + model_name)
        print('')
        print('')
        initialize_train_model(model_name,dataset)

    # ToDO: Read model and compare test set accuracy. Do this only at release when hyperparameters won't be tuned anymore

def get_data_and_label_paths(): 
    # Set up paths to images and labels
    paths_training = dict()
    paths_training['dataset_path'] = data_path
    paths_training['train']['path_images'] = os.path.join(data_path,'train')
    paths_training['train']['path_labels'] = os.path.join(data_path,'train_labels.csv')
    
    paths_test = dict()
    paths_test['dataset_path'] = data_path
    paths_test['test']['path_images'] = os.path.join(data_path,'test')
    paths_test['test']['path_labels'] = ''
    return paths_training, paths_test

def initialize_train_model(model_name,dataset): 
    model_path = os.path.join(os.path.dirname(__file__),"trained_model")
    model = InitializeModel(model_name,dataset,model_path,train_fresh=True)
    history = model.run_model()
    model.try_bypass_local_minimum(n_times=3)

if __name__ == "__main__":
    main()
