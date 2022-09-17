import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # Override plt plotting styles with prettier ones. Still use plt code
import sys
from compare_trained_models import CompareModels

top_path = os.path.dirname(__file__)
models_list = ['DenseNet121','efficientnetB0','efficientnet_v2B0','vgg16','inception','resnet50','resnet_v250',
'resnet_rs101','inception_resnet','regnetX002','regnetY002','mobilenet','mobilenet_v2','mobilenet_v3Small',
'mobilenet_v3Large','xception']
# models_list = ['regnetY002','efficientnetB0']
trained_model_dir = os.path.join(top_path,'trained_model')
metrics_list = ['dice_coef','val_dice_coef']
column_names = ['model']+metrics_list


def main(): 
    compare_models = CompareModels(trained_model_dir,models_list)

if __name__ == "__main__":
    main()
