import os
import hashlib
import gzip
from struct import unpack
import tarfile
import zipfile
import glob
from os_utils import make_new_dirs, move_and_merge_dirs, delete_directory
import tensorflow as tf
import shutil
this_script_location = os.path.dirname(__file__) # Derive from location of this tutorial 

def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def make_download_paths_default(url, data_dir_top):
    download_file_name = os.path.basename(url) # Derive file name from download address. Potential for mismatch? 
    path_to_archive = os.path.join(data_dir_top,download_file_name)
    unpacked_dir = path_to_archive.split('.')[0]
    return path_to_archive,unpacked_dir


def download_and_extract(url, path_to_archive, unpacked_dir):
    make_dirs = [os.path.dirname(unpacked_dir),unpacked_dir]
    make_new_dirs(make_dirs,clean_exisitng_data=True)
    tf.keras.utils.get_file(fname=path_to_archive, origin=url,
                                    extract=False, cache_dir='.',
                                    cache_subdir='') 
        # The unpacked dir name returned by get_file does not account for nested .tar.gz
        # Use custom utility to extract archive to controlled location
    extract_archive(path_to_archive,this_script_location)
    remove_dir = os.path.join(unpacked_dir, 'train', 'unsup')
    delete_directory(remove_dir) 


def extract_archive(path_to_zip, to_path=None, remove_finished=True):
    # A direct upgrade to my unpack util from https://stackoverflow.com/questions/31163668/how-do-i-extract-a-tar-file-using-python-2-4/31163747 
    if to_path is None:
        to_path = os.path.dirname(path_to_zip)
    
    control_name_for_layered_archive(path_to_zip)   

    if remove_finished:
        delete_directory(path_to_zip) 


def control_name_for_layered_archive(path_to_zip): 
    # When running on layered package - .tar.gz - unpacking tools do not yield or control the name of the final 
    # unpacked directory. Deal with it by finding what directory was generated during the run. 
    target_dir_name = path_to_zip.split('.')[0]

    top_path = os.path.dirname(target_dir_name)
    
    subdirs_before_unpack = [name for name in glob.glob(os.path.join(top_path,'*')) if os.path.isdir(name)]

    extract_type(path_to_zip)
    unpacked_dir_name = [name for name in glob.glob(os.path.join(top_path,'*')) if os.path.isdir(name)
     and name not in subdirs_before_unpack]
    if len(unpacked_dir_name) != 1: 
        raise(OSError('Too many unpacked dirs: ' + top_path))
    else: 
        unpacked_dir_name = unpacked_dir_name[0]

    if unpacked_dir_name != target_dir_name: 
        make_new_dirs(target_dir_name)

    move_and_merge_dirs(unpacked_dir_name, target_dir_name)


def unzip(path_to_zip_file,target_path=''): 
    if not target_path: 
        target_path = path_to_zip_file.split('.')[0] # Specify target path from archive name - tar.gz can have different folder name inside it
    if path_to_zip_file.endswith('.tar') or path_to_zip_file.endswith('.gz'): 
        import tarfile
        file = tarfile.open(path_to_zip_file)
        try: 
            file.extractall(target_path)
        except: 
            raise raise_unsupported_archive_error(path_to_zip_file)
        file.close()
    elif path_to_zip_file.endswith('.zip'): 
        import zipfile
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            try: 
                zip_ref.extractall(target_path)
            except: 
                raise raise_unsupported_archive_error(path_to_zip_file)
    else: 
        raise raise_unsupported_archive_error(path_to_zip_file)


def raise_unsupported_archive_error(path_to_zip_file): 
    TypeError('Unsupported archive type: ' + path_to_zip_file)


def extract_type(from_path):
    to_path = os.path.dirname(from_path)
    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))
    return to_path