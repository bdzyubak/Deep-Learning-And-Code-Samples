import pytest
import os
import glob
from  shared_utils.os_utils import make_new_dirs, make_parent_dirs, list_dir, natural_sort, delete_directory, copy_dir
import shutil
temp_subdir = os.path.join(os.path.dirname(__file__),'temp')
temp_target = os.path.join(os.path.dirname(temp_subdir),'temp_target')
test_dirs = ['SER0001','SERIES100','s1_Localizer','s99_newseries.5','s15_mDixon']
test_files = ['file1.dcm','file.png','SERIES0001.dcm']
test_contents = test_dirs + test_files
test_content_paths_dirs = [os.path.join(temp_subdir,name) for name in test_dirs]
test_content_paths_files = [os.path.join(temp_subdir,name) for name in test_files]
test_content_paths = [os.path.join(temp_subdir,name) for name in test_contents]

def make_test_dirs(): 
    make_new_dirs(test_content_paths_dirs)
    
    for file in test_content_paths_files: 
        with open(os.path.join(temp_subdir,file), 'w') as fp:
            pass

def test_make_new_dirs(): 
    created_dirs = [name for name in glob.glob(os.path.join(temp_subdir,'*')) if os.path.isdir(name)] 
    assert set(created_dirs) == set(test_content_paths_dirs), 'Failed to make target dirs.'

def test_make_parent_dirs(): 
    nesting = 5
    temp_nested = temp_subdir
    for i in range(nesting): 
        temp_nested = os.path.join(temp_nested,'temp'+str(i))

    make_parent_dirs(temp_nested,max_nesting=5)
    made_all_nested_dirs = os.path.exists(os.path.dirname(temp_nested)) # Expect parent dirs only 
    shutil.rmtree(os.path.join(temp_subdir,'temp0'))
    assert made_all_nested_dirs, 'Failed to make nested folders.'

def test_list_dir(): 
    assert list_dir(temp_subdir,target_type='') == natural_sort(test_content_paths), 'General sorting error.'

    assert list_dir(temp_subdir,target_type='files') == natural_sort(test_content_paths_files), 'File typing error.'

    assert list_dir(temp_subdir,target_type='folders') == natural_sort(test_content_paths_dirs), 'Directory typing error.' 

    found_content_names = [os.path.basename(name) for name in list_dir(temp_subdir,target_type='',mask='SER*')]
    assert found_content_names == natural_sort(['SER0001','SERIES100','SERIES0001.dcm']), 'Mask prefix error.' 

    found_content_names = [os.path.basename(name) for name in list_dir(temp_subdir,target_type='',mask='*.dcm')]
    assert found_content_names == ['file1.dcm','SERIES0001.dcm'], 'Ending mask error.' 

    found_content_names = [os.path.basename(name) for name in list_dir(temp_subdir,target_type='',mask='s*_mDixon')]
    assert found_content_names == ['s15_mDixon'], 'Middle mask Error.'

    found_content_names = [os.path.basename(name) for name in list_dir(temp_subdir,target_type='files',mask='SER*')]
    assert found_content_names == ['SERIES0001.dcm'], 'Files and mask error.' 

    found_content_names = [os.path.basename(name) for name in list_dir(temp_subdir,target_type='folders',mask='*.dcm')]
    assert found_content_names == [], 'Files found as folders.' 


def test_natural_sort(): 
    input = ['001.dcm','010.dcm','002.dcm','100.dcm']
    output = ['001.dcm','002.dcm','010.dcm','100.dcm']
    assert natural_sort(input) == output, 'Failed natural sorting'

    input = ['001.dcm','SER001','010.dcm','SER010','002.dcm','SER002','100.dcm','SER100']
    output = ['001.dcm','002.dcm','010.dcm','100.dcm','SER001','SER002','SER010','SER100']
    assert natural_sort(input) == output, 'Sorting failed to split files and dirs.'


def test_copy_dir():   
    make_new_dirs(temp_target,clean_subdirs=True)
    copy_dir(test_content_paths_dirs[0],temp_target)
    assert list_dir(temp_target,file_name_only=True)[0] == test_dirs[0], 'Failed to copy directory.' 

    make_new_dirs(temp_target,clean_subdirs=True)
    copy_dir(test_content_paths_files[0],temp_target)
    assert list_dir(temp_target,file_name_only=True)[0] == test_files[0], 'Failed to copy file.' 

    make_new_dirs(temp_target,clean_subdirs=True)
    copy_dir(temp_subdir,temp_target)
    assert list_dir(os.path.join(temp_target,os.path.basename(temp_subdir)),
            file_name_only=True) == list_dir(temp_subdir,file_name_only=True), 'Failed to copy dir with contents.' 

    make_new_dirs(temp_target,clean_subdirs=True)
    copy_dir(temp_subdir+'*',temp_target)
    assert list_dir(os.path.join(temp_target),
            file_name_only=True) == list_dir(temp_subdir,file_name_only=True), 'Failed to copy all contents.' 


def test_delete_directory(): 
    directory_exists_prior = os.path.exists(temp_subdir) and os.path.isdir(temp_subdir)
    delete_directory(temp_subdir)
    directory_removed = not os.path.exists(temp_subdir)
    assert directory_exists_prior and directory_removed, 'Delete_directory test failed.'
    delete_directory(temp_target)

if __name__ == '__main__': 
    make_test_dirs() # Run this first as test_make_new_dirs relies on this having been run. 
    retcode = pytest.main([os.path.basename(__file__)])
