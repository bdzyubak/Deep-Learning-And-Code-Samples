import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
import importlib
import os
from os_utils import make_new_dirs
from metrics import dice_coef
import sys

valid_model_names_builtin = ['efficientnet','vgg','resnet']
valid_model_names_custom = ['unet']
valid_model_names_all = valid_model_names_builtin + valid_model_names_custom
valid_opt_params = ['learning_rate','num_epochs','batch_size']

class InitializeModel(): 
    def __init__(self,model_name,dataset,model_path,base_trainable=True,params=None): 
        self.model_name = model_name.lower()
        self.params = params
        self.model_path = model_path
        self.dataset = dataset
        make_new_dirs(model_path)
        
        self.set_default_optimization_params() 

        # Model top layer will always be excluded at this time as the goal is transfer learning 
        base_model = self.make_model() 

        # The target is mostly medical applications, so by default, enable training of all layers 
        # i.e. trained ImageNet weights are only used as starting guesses
        base_model.trainable = base_trainable 
        # TODO: Split off segmentation context. In this case, final layer needs to have the same H and W
        # as the pre-trained one, but should have a flexible number of output channels. 
        self.add_final_layers(base_model)

        self.make_callbacks()
        self.set_loss_function()
        self.set_optimizer()
        self.set_metrics()
        

    def check_valid_opt_params(self): 
        print('Supported optimization parameters are: ')
        print(valid_opt_params)
        return valid_opt_params
    
    def set_optimization_params(self,opt_params:dict): 
        for key in opt_params: 
            if key in valid_opt_params: 
                setattr(self,key,opt_params[key])
            else: 
                self.check_valid_opt_params()
                print('Unrecognized optimization parameter: ' + key)

    def check_valid_model_names(self): 
        print('Valid model names are: ')
        print(valid_model_names_all)
        return valid_model_names_all

    def make_model(self): 
        import_statement, method_name = self.make_import_statement()
        # Import is being bound to returned variable name. 
        if self.model_name_base in valid_model_names_builtin: 
            module = importlib.import_module(import_statement)
            model = getattr(module,method_name)
            base_model = model(input_shape=self.dataset.image_dims_original,include_top=False)
        elif self.model_name_base in valid_model_names_custom: 
            spec = importlib.util.spec_from_file_location(os.path.basename(import_statement), import_statement)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = getattr(module,method_name)
            base_model = model(input_shape=self.dataset.image_dims_original)
        return base_model

    def make_import_statement(self): 
        model_name = self.model_name
        import_statement = ''
        method_name = ''
        self.model_name_base, complexity = self.get_complexity_from_model_name()
        if model_name.startswith('efficientnet'): 
            if not model_name[-1].isdigit(): 
                raise(ValueError('Must specify complexity: EfficientNetB0 - B7'))
            import_statement = 'tensorflow.keras.applications.efficientnet' 
            method_name = 'EfficientNetB' + complexity
        elif model_name.startswith('vgg'): 
            import_statement = 'tensorflow.keras.applications.vgg' + complexity
            method_name =  'VGG' + complexity
        elif model_name in valid_model_names_builtin: 
            # Generic importer - doesn't hurt to try but generally specific support will need to be enabled
            import_statement = 'tensorflow.keras.applications.' + model_name
            method_name = model_name
        elif model_name.startswith('unet'): 
            import_statement = self.make_path_to_custom_models()
            method_name = 'build_unet'
        
        if not import_statement or not method_name: 
            self.check_valid_model_names()
            raise(ValueError('Failed to make ' + model_name + ' model with generalized initializer.'))
        return import_statement, method_name
    
    def make_path_to_custom_models(self): 
        package_top = os.path.dirname(os.path.dirname(__file__))
        model_module_path = os.path.join(package_top,'common_models',self.model_name,self.model_name+'_model.py')
        return model_module_path

    def get_complexity_from_model_name(self): 
        model_name_base = [name for name in valid_model_names_all if self.model_name.startswith(name)][0]
        complexity = self.model_name.split(model_name_base)[-1] # Ending digits, if any, indicate complexity
        if len(complexity) > 5: 
            raise(ValueError('Model complexity spec too long.')) # Security limiter on arbitrary inputs
        return model_name_base, complexity

    def add_final_layers(self,base_model): 
        # TODO: Add ability to accept dictionary of layers 
        self.model = Sequential([
            base_model,
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(2, activation='softmax')
        ])

    def set_default_optimization_params(self): 
        self.batch_size = 32
        self.learning_rate = 1e-4   ## 0.0001
        self.num_epochs = 40

    def make_callbacks(self):
        model_path = self.model_path
        model_path = os.path.join(model_path,"model.h5")
        csv_path = os.path.join(model_path,"data.csv") 
        self.callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1),
            CSVLogger(csv_path),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
        ]

    def set_loss_function(self,loss='binary_crossentropy'): 
        # For now, use binary cross entropy appropriate for mutually exclusive classes
        # Categorical_crossentry could be used for non-mutually exclusive classes
        self.loss = loss

    def set_optimizer(self): 
        # For now, always use Adam
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)

    def set_metrics(self): 
        # Dice is appropriate for segmentation tasks, accuracy can be used for classification tasks
        self.metrics =[dice_coef,'accuracy']

    def run_model(self,): 
        # Use steps per epoch rather than batch size since random perumtations of the training data are used, rather than all training data
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=self.metrics)
        
        self.history = self.model.fit(
            x = self.dataset.train_dataset, 
            steps_per_epoch = self.dataset.train_steps, 
            epochs = 40,
            validation_data = self.dataset.valid_dataset, 
            validation_steps = self.dataset.valid_steps, 
            verbose = 1, 
            callbacks=self.callbacks
        )
