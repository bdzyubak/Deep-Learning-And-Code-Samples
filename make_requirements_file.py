import subprocess
import os
project_path = os.path.dirname(__file__)

# Before use, move the common_models folder out of the package. Many of these are not 
# used at the momeny, and many have imports that are outdated and not used elsewhere.

subprocess.check_output('python -m pip install --upgrade pip')
subprocess.check_output('python -m pip install --upgrade pipreqs')
subprocess.check_output('pipreqs --mode no-pin --force ' + project_path)

with open(os.path.join(project_path,'requirements.txt'),'r') as file: 
    lines = file.readlines()    
# Remove incorrectly reported dependency. pip install scikit-image but import skimage
write_lines = []
for line in lines: 
    if 'skimage' in line: 
        continue
    # Exclude pywin32 dependency which cannot be installed on Linux
    if 'pywin32' in line: 
        continue
    write_lines.append(line)

with open(os.path.join(project_path,'requirements.txt'),'w') as file: 
    file.writelines(write_lines)
