""" setup.py
This module sets up python dependencies for the MREPlus (Hepatogram Plus) project. This must be run on any new 
system where the project workflows will be run. 
Python (tested on 3.10) needs to be installed first, added to system path, and aliased to run as python, if it isn't by default. 
Execution:
    python setup.py
Inputs:
    None.
Outputs: 
    Creates or appends to system variable PYTHONPATH. 
"""

import os
import subprocess

setup_path = os.path.dirname(__file__)
system_var = 'PYTHONPATH'

# NOTE: Should be OS agnostic, but not tested on Linux
def wrap_set(setup_path): 
    if not os.getenv("PYTHONPATH"): 
        subprocess.call(['setx', system_var,setup_path], shell=True)
    else: 
        if setup_path not in os.getenv("PYTHONPATH"): # Do not duplicate
            append_path = os.getenv("PYTHONPATH") + ';' + setup_path
            subprocess.call(['setx', system_var,append_path], shell=True)

wrap_set(setup_path)

python_ver = '3.10'
if os.name != 'nt': 
    os.system('apt install python' + python_ver + '-distutils')
    os.system('alias python=python3.10') 

os.system('python -m pip install --upgrade pip')

dependencies = ['pydicom','pandas','tensorflow','tensorflow_datasets'
            ,'scikit-image', 'opencv-python', 'matplotlib', 'IPython'
            ,'seaborn', 'pydot', 'sklearn','opendatasets', 'pytest'
            ,'psutil','git+https://github.com/tensorflow/examples.git']
 
for dep in dependencies: 
    os.system('python -m pip install --upgrade ' + dep)