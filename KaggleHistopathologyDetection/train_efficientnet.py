import os
import numpy as np
import tensorflow as tf
import sys
top_path = os.path.dirname(__file__)
utils_path = os.path.join(os.path.dirname(top_path),'shared_utils')
sys.path.append(utils_path)
from prep_training_data import ImgLabelDataset
from model_initializer import InitializeModel

H = 256
W = 256

def main(): 
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    dataset = ImgLabelDataset(os.path.join(top_path,'data','train'))
    # dataset.set_max_batch_size()

    model_name = 'efficientnetB0'
    model_path = os.path.join(os.path.dirname(__file__),"trained_model")
    model = InitializeModel(model_name,dataset,model_path)
    history = model.run_model()
    model.try_bypass_local_minimum(n_times=3)

if __name__ == "__main__":
    main()
