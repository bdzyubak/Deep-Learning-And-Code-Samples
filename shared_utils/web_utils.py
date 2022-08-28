import os
import opendatasets as od
import shutil
from shared_utils.os_utils import move_contents

def download_if_not_exist(data_path,url,dataset_name=''): 
    if not dataset_name: 
        dataset_name = os.path.basename(url)
    # Will require Kaggle account and API token
    if not os.path.exists(data_path) or not os.listdir(data_path): 
        
        od.download(url) # Url, not os call, so hardcoded slash is ok

        # opendatasets seems to have now way to download to target location
        # grab from ./ and drop into controlled location of this file
        move_contents(dataset_name,data_path)

def move_data(origin,target): 
    if not os.path.isdir(origin): 
        raise(OSError('WARNING: Missing data in download location.'))
    shutil.move(origin,target)