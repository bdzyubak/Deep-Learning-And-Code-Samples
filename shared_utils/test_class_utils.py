import pytest
from shared_utils.class_utils import print_instance_attributes

def test_print_instance_attributes():
    test_class = TestClass()
    print_instance_attributes(test_class)

class TestClass(): 
    def __init__(self): 
        self.string_attribute = 'Hello World'
        self.num_attribute = 6 
    
    def add_string(self): 
        self.string_attribute += ' , Hello.'

if __name__ == '__main__': 
    message = pytest.main([__file__])