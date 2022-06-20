import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # Override plt plotting styles with prettier ones. Still use plt code
import sys
from os_utils import list_dir
metrics_get_final = ['dice_coef','val_dice_coef','loss','val_loss']
metrics_get_total = ['epoch_time']
metrics_list = metrics_get_final + metrics_get_total
column_names = ['model']+metrics_get_final+metrics_get_total

class CompareModels(): 
    def __init__(self,models_path,models_list): 
        self.models_path = models_path
        self.plot_metrics = metrics_list # Default displayed metrrics to logged metrics, assuming no preprocessing
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
            row = self.get_final_metrics(model_name)
            self.measurements = self.measurements.append(row,ignore_index=True)
        self.replace_metric_names()
        self.plot_bar_chart()
        print('Done.')

    def replace_metric_names(self): 
        # For metrics that are preprocessed before being reported directly, change name
        # e.g. training reports epoch_time but plotting will show its total 'training_time'
        for metric in metrics_get_total: 
            plot_metric_name = metric.replace(metric.split('_')[0],'training')
            self.measurements.rename(columns={metric:plot_metric_name},inplace=True)

    def get_final_metrics(self,model_name): 
        history_file_name = os.path.join(self.models_path,model_name,'training_history_'+model_name.lower() + '.csv')
        df = pd.read_csv(history_file_name)      
        # Insert slicing of time-metrics and interpretationa as cumulative time, not last time point
        missing_metrics = [metric for metric in metrics_list if metric not in df.columns]
        if missing_metrics: 
            print(missing_metrics)
            raise(ValueError('Requested metrics missing in training history.'))
        
        metrics_final_value = df[metrics_get_final].iloc[-1]
        metrics_average_value = df[metrics_get_total].sum()
        metrics = pd.concat([metrics_final_value,metrics_average_value])
        final_metrics = pd.concat([pd.Series({'model':model_name}),metrics])
        return final_metrics

    def plot_bar_chart(self): 
        for metric in self.measurements.columns[1:]: 
            plt.figure()
            palette = self.set_palette(metric)
            fig = sns.barplot(x=self.measurements['model'],y=self.measurements[metric], data = self.measurements, dodge=False,palette=palette)
            plt.xticks(rotation=30)
            plt.get_current_fig_manager().full_screen_toggle()
            fig = plt.get_current_fig_manager()
            fig.window.showMaximized()
        plt.show()

    def set_palette(self, metric):
        if 'dice' in metric: 
            palette = sns.color_palette()
        elif 'time' in metric: 
            palette = sns.color_palette('bright')
        elif 'loss' in metric: 
            palette = sns.color_palette('dark')
        else: 
            raise(ValueError('Unsupported metric for plotting: ' + metric))
        sns.set_palette(palette) # Not working. Bypass by setting as part of plot outisde of this func
        return palette
