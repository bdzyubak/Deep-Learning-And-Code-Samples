import os
import shutil
import re
import glob

def make_new_dirs(folders,clean_exisitng_data=False):
    if isinstance(folders,str): 
        folders = [folders]
    for folder in folders:
        if clean_exisitng_data and os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)


def move_and_merge_dirs(origin_dir_name, target_dir_name):
    for content in origin_dir_name: 
        if os.path.isfile(content): 
            shutil.move(content,os.path.join(target_dir_name,os.path.basename(content)))
        else: 
            subdirs = glob.glob(os.path.join(content,'*'))
            for subdir in subdirs: 
                shutil.move(subdir,os.path.join(target_dir_name,os.path.basename(subdir)))
        shutil.rmtree(content)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)