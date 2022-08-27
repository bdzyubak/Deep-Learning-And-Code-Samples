import os
import numpy as np
import tensorflow as tf
from shared_utils.prep_training_data import ImgLabelDataset
from shared_utils.model_initializer import InitializeModel

top_path = os.path.dirname(__file__)
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
