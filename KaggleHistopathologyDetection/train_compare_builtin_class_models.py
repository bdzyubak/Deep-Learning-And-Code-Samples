import os
import numpy as np
import tensorflow as tf
import sys
top_path = os.path.dirname(__file__)
utils_path = os.path.join(os.path.dirname(top_path),'shared_utils')
sys.path.append(utils_path)
from prep_training_data import ImgLabelDataset
from model_initializer import InitializeModel
# models_list = ['inception']
models_list = ['DenseNet121','efficientnetB0','efficientnet_v2B0','vgg16','inception','resnet50','resnet_v250',
'resnet_rs101','inception_resnet','regnetx002','regnety002','mobilenet','mobilenet_v2','mobilenet_v3Small',
'mobilenet_v3Large','xception']
dataset = ImgLabelDataset(os.path.join(top_path,'data','train'))

def main(): 
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    
    for model_name in models_list: 
        print('Training ' + model_name)
        print('')
        print('')
        initialize_train_model(model_name)

    # ToDO: Read model and compare test set accuracy. Do this only at release when hyperparameters won't be tuned anymore


def initialize_train_model(model_name): 
    model_path = os.path.join(os.path.dirname(__file__),"trained_model")
    model = InitializeModel(model_name,dataset,model_path,train_fresh=True)
    history = model.run_model()
    model.try_bypass_local_minimum(n_times=3)

if __name__ == "__main__":
    main()
