import pytest
import os
import shutil
from  os_utils import make_new_dirs, list_dir, natural_sort

def test_list_dir(): 
    temp_subdir = os.path.join(os.path.dirname(__file__),'temp')
    test_contents = ['SER0001','SERIES100','s1_Localizer','s99_newseries.5','s15_mDixon','file1.dcm','file.png','SERIES0001.dcm']
    number_of_dirs = 5
    test_content_paths = [os.path.join(temp_subdir,name) for name in test_contents]
    test_content_paths_dirs = test_content_paths[:number_of_dirs]
    test_content_paths_files = test_content_paths[number_of_dirs:]
    make_new_dirs(test_content_paths_dirs)
    for file in test_content_paths_files: 
        with open(os.path.join(temp_subdir,file), 'w') as fp:
            pass

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
    assert found_content_names == ['SER0001','SERIES100'], 'Folders and mask error.' 

    shutil.rmtree(temp_subdir)

if __name__ == '__main__': 
    retcode = pytest.main()