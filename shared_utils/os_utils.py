import os
import shutil
import re
import glob
import subprocess
from pathlib import Path
from typing import Union

if os.name == 'nt': 
    os_name = 'win'
else: 
    os_name = 'lin'

"""A utilities module for making, finding, and interacting with directories and the OS. 

Import individual functions for smoother interactions e.g. 

from shared_utils.os_utils import list_dir

contents = list_files(directory)
# Instead of: contents = [os.path.join(directory,name) for name in os.listdir(directory)]
# or: contents = glob.glob(os.path.join(directory,'*'))

"""


def make_new_dirs(folders,clean_subdirs=True,max_nesting=1):
    if isinstance(folders,str): 
        folders = [folders]
    for folder in folders:
        # Also make parent folders n levels up. 
        make_parent_dirs(folder,max_nesting)
        if clean_subdirs and os.path.exists(folder): 
                delete_directory(folder)
        if not os.path.exists(folder): 
            os.mkdir(folder)

def make_parent_dirs(folder,max_nesting): 
    nesting = 0
    folder_level_not_exists = os.path.dirname(folder)
    folders_to_make = list()
    while not os.path.exists(folder_level_not_exists) and nesting <= max_nesting: 
        folders_to_make.append(folder_level_not_exists)
        folder_level_not_exists = os.path.dirname(folder_level_not_exists)
        nesting += 1
    for make_folder in folders_to_make[::-1]: 
        os.mkdir(make_folder)

def move_and_merge_dirs(origin_dir_name, target_dir_name):
    contents = glob.glob(os.path.join(origin_dir_name,'*'))
    for content in contents: 
        shutil.move(content,os.path.join(target_dir_name,os.path.basename(content)))
    delete_directory(origin_dir_name)

# This project does not currently use simlinks, but resolve symlink would require a win32com dependency
# def resolve_symlink(link): 
#     if os.name == 'nt':
#         # According to documentation, os.path.realpath shoudl work on windows but it has no effect
#         if os.name == 'nt':
#             import win32com.client
#         shell = win32com.client.Dispatch("WScript.Shell")
#         shortcut = shell.CreateShortCut(link)
#         path_digest = shortcut.Targetpath
#     else:
#         path_digest = os.path.realpath(link)
#     return path_digest

def natural_sort(dir_list):
    # Function for sorting files/directories/other lists in natural order and not 1, 10, 100, 2, 20... 
    # Add feature - first first or directories first
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(dir_list, key=alphanum_key)


def delete(target): 
    if os.path.isdir(target):
        delete_directory(target)
    elif os.path.exists(target):
        os.remove(target)


def delete_directory(target_directory): 
    # This module uses OS calls to speed up removal of large numbers of files compared to shutil's rmtree
    if os.path.exists(target_directory): 
        if os_name == 'win': 
            command = 'rmdir /s /q'
        else: 
            command = 'rm -rf'
        
        output = subprocess.check_output(command + add_argument(target_directory),shell=True)
        if output: 
            print('Cannot delete directory - check file lock: ' + target_directory)
    else: 
        print('Directory to be deleted does not exist: ' + target_directory)


def list_dir(directory,mask='*',target_type='',file_name_only=False): 
    if not isinstance(directory,str) or not isinstance(target_type,str) or not isinstance(mask,str): 
        raise(TypeError('Directory, type, and mask need to be strings.'))

    if not os.path.exists(directory) or not os.path.isdir(directory): 
        raise(OSError('Input directory not valid.'))

    contents = glob.glob(os.path.join(directory,mask))
        
    # Mask to only return files or folders
    if target_type == 'files': 
        if '.' in mask: 
            contents = [name for name in contents if name.endswith(target_type)]
        else: 
            contents = [name for name in contents if os.path.isfile(name)]
    elif target_type == 'folders': 
        contents = [name for name in contents if os.path.isdir(name)]
    
    if file_name_only: 
        contents = [os.path.basename(name) for name in contents]
    return natural_sort(contents)

def list_files(directory,mask='*',file_name_only=False): 
    files = list_dir(directory=directory,mask=mask,file_name_only=file_name_only,target_type='files')
    return files

def list_directories(directory,mask='*',file_name_only=False): 
    folders = list_dir(directory=directory,mask=mask,file_name_only=file_name_only,target_type='folders')
    return folders

def copy_dir(source,destination): 
    make_new_dirs(destination) # Attempt to make target directory (will recreate limited number of levels)
    if not source.endswith('*'): 
        contents = [source]
    else: 
        contents = list_dir(source[:-1]) 
    for content in contents: 
        copy(content,destination)

def copy(source,destination): 
    if not os.path.exists(source): 
        raise(OSError('Source to copy does not exist: ' + source))
    if not os.path.exists(destination) or not os.path.isdir(destination): 
        raise(OSError('Copy target directory does not exist: ' + destination))
    if os.path.isfile(source): 
        shutil.copy(source,os.path.join(destination,os.path.basename(source)))
    elif os.path.isdir(source): 
        shutil.copytree(source,os.path.join(destination,os.path.basename(source)))

def move_contents(source,destination): 
    contents = list_dir(source)
    if not os.path.exists(destination): 
        make_new_dirs(destination)
    for content in contents: 
        shutil.move(content,destination)
    os.remove(source)

def extract_file_from_container(exec,result_dir,file): 
    target_file = os.path.join(result_dir, file)
    if os.name == 'nt': 
        shutil.copyfile(os.path.join(exec,'mreplus_config',file),target_file)
    else: 
        os.system('docker exec ' + exec + ' /bin/bash -c "' + 'cat mreplus_config/' + file + ' > ' \
            + target_file + '"')
    return target_file

def parent_dir(path,level,dir_only=False): 
    path = str(Path(path).parents[level-1])
    if dir_only: 
        path = os.path.basename(path)
    return path

def parent_dir_name(path,level): 
    return parent_dir(path,level,dir_only=True)

def enclose_in_quotes(cmd): 
    """ Enclose a path in quotes for OS calls. Do not use internall for python arguments. """
    return '"' + cmd + '"'

def add_argument(arg): 
    if arg.startswith('-'): 
        return ' ' + arg
    else: 
        return ' ' + enclose_in_quotes(arg)

def add_dir_reenclose(path:str,dirs:Union[str,list]) -> str: 
    """ Add subdir(s) to path, keeping enclosed in quotes. 

    Takes a path which may or may not be enclosed in quotes and adds one or more subdirectories, 
      enclosing the result in quotes to deal with spaces in path names. Only run this if you are 
      using a system call. Python does not need and will object to quotes in paths e.g. os.listdir(r"C:\")

    Args:
        path (str): The original path. 
        dirs (str or list): Subdirectory(ies) to add to the path

    Returns:
        combined_path (str): The total combined path, enclosed in quotes
    """

    if path.startswith('"') or path.startswith('\''): 
        path = path[1:]
    if path.endswith('"') or path.endswith('\''): 
        path = path[:-1]
    if not isinstance(dirs,list): 
        dirs = [dirs]
    for dir in dirs: 
        path = os.path.join(path,dir)
    combined_path = enclose_in_quotes(path)
    return combined_path