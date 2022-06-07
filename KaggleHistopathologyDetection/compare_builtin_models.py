import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # Override plt plotting styles with prettier ones. Still use plt code
import sys
top_path = os.path.dirname(__file__)
utils_path = os.path.join(os.path.dirname(top_path),'shared_utils')
sys.path.append(utils_path)
# models_list = ['DenseNet121','efficientnetB0','efficientnet_v2B0','vgg16','inception','resnet50','resnet_v250',
# 'resnet_rs101','inception_resnet','regnetX002','regnetY002','mobilenet','mobilenet_v2','mobilenet_v3Small',
# 'mobilenet_v3Large','xception']
models_list = ['regnetY002','efficientnetB0']
trained_model_dir = os.path.join(top_path,'trained_model')
metrics_list = ['dice_coef','val_dice_coef']
column_names = ['model']+metrics_list

def main(): 
    all_metrics = pd.DataFrame(columns=column_names)
    plt.close('all')
    for model_name in models_list: 
        print('Fetching history ' + model_name)
        row = get_final_metrics(model_name,metrics_list)
        all_metrics = all_metrics.append(row,ignore_index=True)
    plot_bar_chart(all_metrics)
    print('Done.')

def get_final_metrics(model_name,metrics_list): 
    history_file_name = os.path.join(trained_model_dir,model_name,'training_history_'+model_name.lower() + '.csv')
    df = pd.read_csv(history_file_name)
    metrics = df[metrics_list].iloc[-1]
    final_metrics = pd.concat([pd.Series({'model':model_name}),metrics])
    return final_metrics

def plot_bar_chart(all_metrics): 
    for metric in metrics_list: 
        plt.figure()
        sns.barplot(x=all_metrics['model'],y=all_metrics[metric], data = all_metrics)
    plt.show()

if __name__ == "__main__":
    main()
