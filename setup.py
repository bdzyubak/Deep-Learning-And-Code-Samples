"""setup.py
This script sets up the hepplus python development environment for an operating system.
It sets the system variable PYTHONPATH to include a list of desired FOLDERS, enabling clean imports across modules.
It also installs all hepplus python dependencies via pip and requirements.txt.
python >=3.9 needs to be installed first, added to system path, and aliased as python (e.g. alias python=python3.9).
After running, close terminals and IDE to refresh system variables.
Execution:
    python setup.py
Inputs:
    None
Outputs:
    Updates system variable PYTHONPATH
    Installs or upgrades pip and hepplus python dependencies
"""

import os
import subprocess

SET_FRESH_PYTHONPATH=True
SETUP_PATH=os.path.dirname(__file__)
FOLDERS=['shared_utils','common_models']
SYSTEM_VAR='PYTHONPATH'
if ' ' in SETUP_PATH:
    IMPORTABLE_FOLDERS=['"'+os.path.join(SETUP_PATH,folder)+'"' for folder in FOLDERS]
else:
    IMPORTABLE_FOLDERS=[os.path.join(SETUP_PATH,folder) for folder in FOLDERS]
if os.name=='nt': 
    SEPARATOR=';'
else: 
    SEPARATOR=':'

def check_all_importables_in_path(current_path):
    if current_path:
        all_paths_present=all((folder in current_path) for folder in IMPORTABLE_FOLDERS)
    else:
        all_paths_present=False
    return all_paths_present

def set_environ_windows(new_path):
    subprocess.call(['setx',SYSTEM_VAR,new_path],shell=True)
    return

def set_environ_linux(new_path):
    # with open(bashrc_path,'a+') as file:
    #     lines = file.readlines()
    #     lines = [line for line in lines if 'export PYTHONPATH' not in line]
    #     shutil.move(bashrc_path,bashrc_path+'_bu')
    #     lines.append(export_line)
    #     file.writelines(lines)
    user=str(subprocess.check_output('whoami',shell=True))[2:-3]
    bashrc_path='/home/'+user+'/.bashrc'
    export_line='export '+SYSTEM_VAR+'='+new_path
    with open(bashrc_path, 'a') as file:
        file.write(os.linesep)
        file.write('# Add Python system variables for imports')
        file.write(export_line+os.linesep)
    return

def set_path():
    current_path=os.getenv('PYTHONPATH')
    all_paths_present=check_all_importables_in_path(current_path)
    if all_paths_present:
        print('setup.py: all importable folders already present, not modifying PYTHONPATH')
    else:
        if SET_FRESH_PYTHONPATH:
            current_path=''
        new_path=current_path+(SEPARATOR.join(IMPORTABLE_FOLDERS))+SEPARATOR
        if os.name=='nt':
            set_environ_windows(new_path)
        else:
            set_environ_linux(new_path)
    return

def install_python_dependencies():
    requirements_file=os.path.join(SETUP_PATH,'requirements.txt')
    subprocess.check_output('python -m pip install --upgrade pip',shell=True)
    subprocess.check_output('python -m pip install -r '+requirements_file,shell=True)
    return

if __name__ == '__main__':
    set_path()
    install_python_dependencies()