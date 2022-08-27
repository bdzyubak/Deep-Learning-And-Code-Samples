import os
import opendatasets as od
import shutil

def download_if_not_exist(data_path,url): 
    # Will require Kaggle account and API token
    if not os.path.exists(data_path) or not os.listdir(data_path): 
        
        od.download(url) # Url, not os call, so hardcoded slash is ok

        # opendatasets seems to have now way to download to target location
        # grab from ./ and drop into controlled location of this file
        move_data(origin=os.path.basename(url),target=data_path)

def move_data(origin,target): 
    if not os.path.isdir(origin): 
        raise(OSError('WARNING: Missing data in download location.'))
    shutil.move(origin,target)