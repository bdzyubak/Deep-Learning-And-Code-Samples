import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # Override plt plotting styles with prettier ones. Still use plt code
import sys
from os_utils import list_dir
metrics_list = ['dice_coef','val_dice_coef']
column_names = ['model']+metrics_list

class CompareModels(): 
    def __init__(self,models_path,models_list): 
        self.models_path = models_path
        self.measurements = pd.DataFrame(columns=column_names)
        self.check_results_present(models_list)
        self.compare_histories()

    def check_results_present(self,models_list): 
        subdirs = list_dir(self.models_path,file_name_only=True)
        subdirs_in_model_list = [name for name in subdirs if name in models_list]
        models_with_results = list()
        for subdir in subdirs_in_model_list: 
            training_history_files = list_dir(os.path.join(self.models_path,subdir),mask='*.csv')
            history_of_best_attempt = [name for name in training_history_files if 'backup' not in name]
            if history_of_best_attempt: 
                models_with_results.append(subdir)
            else: 
                print('No training result for: ' + models_list)
        self.models_list = models_with_results

    def compare_histories(self): 
        plt.close('all')
        for model_name in self.models_list: 
            print('Fetching history ' + model_name)
            row = self.get_final_metrics(model_name,metrics_list)
            self.measurements = self.measurements.append(row,ignore_index=True)
        self.plot_bar_chart()
        print('Done.')

    def get_final_metrics(self,model_name,metrics_list): 
        history_file_name = os.path.join(self.models_path,model_name,'training_history_'+model_name.lower() + '.csv')
        df = pd.read_csv(history_file_name)
        metrics = df[metrics_list].iloc[-1]
        final_metrics = pd.concat([pd.Series({'model':model_name}),metrics])
        return final_metrics

    def plot_bar_chart(self): 
        for metric in metrics_list: 
            plt.figure()
            sns.barplot(x=self.measurements['model'],y=self.measurements[metric], data = self.measurements)
        plt.show()
