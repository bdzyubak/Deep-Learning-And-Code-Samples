import os
import shutil
import re
import glob

def make_new_dirs(folders,clean_exisitng_data=False):
    # If input is a string (a single directory), convert to list for common handling 
    if isinstance(folders,str): 
        folders = [folders]
    for folder in folders:
        # Also make parent folder 1 level up e.g. making /data/train will also make /data. Used for shared folders. 
        parent_folder = os.path.dirname(folder)
        if not os.path.exists(parent_folder): 
            os.mkdir(parent_folder)
        
        if os.path.exists(folder) and clean_exisitng_data: 
            shutil.rmtree(folder)
        if not os.path.exists(folder):  
            os.mkdir(folder)


def move_and_merge_dirs(origin_dir_name, target_dir_name):
    contents = glob.glob(os.path.join(origin_dir_name,'*'))
    for content in contents: 
        shutil.move(content,os.path.join(target_dir_name,os.path.basename(content)))
    shutil.rmtree(origin_dir_name)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)