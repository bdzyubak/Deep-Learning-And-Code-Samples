import pytest
import model_initializer
import importlib

@pytest.fixture
def models_list(): 
    return ['DenseNet121','efficientnetB0','efficientnet_v2B0','vgg16','inception','resnet50','resnet_v250',
    'resnet_rs101','inception_resnet','regnetX002','regnetY002','mobilenet','mobilenet_v2','mobilenet_v3Small',
    'mobilenet_v3Large','xception']

def test_valid_model_names(models_list): 
    supported_models = sorted(model_initializer.valid_model_names_all.keys())
    assert supported_models == models_list, 'Mismatch in supported model lists.'

def test_InitializeModel(models_list): 
    try: 
        model_name = models_list[0]
        model = model_initializer.InitializeModel(model_name,'','')
    except: 
        raise(ValueError('Failed to initialize model'))
    
@pytest.fixture
def example_initialized_model(models_list): 
    return model_initializer.InitializeModel(models_list[0],'','')


# def test_make_model(): 
#     assert model == , 'Failed to import: ' + model_name

if __name__ == '__main__': 
    # models_list = 

    pytest.main([__file__])