import os

def get_data_and_label_paths(data_path): 
    # Set up paths to images and labels
    paths_training = dict()
    paths_training['data_path_top'] = data_path
    paths_training['path_images'] = os.path.join(data_path,'train')
    paths_training['path_labels'] = os.path.join(data_path,'train_labels.csv')
    
    paths_test = dict()
    paths_test['data_path_top'] = data_path
    paths_test['path_images'] = os.path.join(data_path,'test')
    paths_test['path_labels'] = ''
    return paths_training, paths_test