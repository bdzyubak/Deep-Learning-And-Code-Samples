import os

def get_path_training_lung_segm_normal(data_path): 
    # Function to dig down to unusual data locations for the covidqu dataset
    dataset_path = os.path.join(data_path, "Lung Segmentation Data","Lung Segmentation Data"
    ,"Train","Normal")
    paths = dict()
    paths['data_path_top'] = dataset_path
    paths['path_images'] = os.path.join(dataset_path,'images')
    paths['path_labels'] = os.path.join(dataset_path,'lung masks')
    return paths