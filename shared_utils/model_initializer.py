import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import * 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
import importlib
import os

valid_model_names = ['efficientnet','unet','vgg16','vgg24','resnet50']
valid_opt_params = ['learning_rate','num_epochs','batch_size']

class InitializeModel(): 
    def __init__(self,model_name,params,model_path,base_trainable=True): 
        self.model_name = model_name
        self.params = params
        self.model_path = model_path
        
        self.set_default_optimization_params() 

        self.pick_model()

        self.base_model.trainable = base_trainable

        self.make_callbacks()

    def check_valid_opt_params(self): 
        print('Supported optimization parameters are: ')
        print(valid_opt_params)
    
    def set_optimization_params(self,opt_params:dict): 
        for key in opt_params: 
            if key in valid_opt_params: 
                setattr(self,key,opt_params[key])
            else: 
                self.check_valid_opt_params()
                print('Unrecognized optimization parameter: ' + key)

    def check_valid_model_names(self): 
        print('Valid model names are: ')
        print(valid_model_names)

    def pick_model(self): 
        model_name = self.model_name
        try: 
            if model_name.lower().startswith('efficientnet'): 
                if not model_name[-1].isdigit(): 
                    raise(ValueError('Must specify complexity: EfficientNetB0 - B7'))
                complexity = 'B' + int(model_name[-1])
                import_statement = 'tf.keras.applications.efficientnet.EfficientNetB' + complexity
                importlib.import_module(import_statement)
            elif model_name in valid_model_names: 
                import_statement = 'tf.keras.applications.' + model_name
                importlib.import_module(import_statement)
            else: 
                self.check_valid_model_names()
                raise(ValueError('Unsupported model name: ' + model_name))
        except: 
            raise(ValueError('Model ' + model_name + ' unsupported by generalized import.'))
        
        self.add_parameters()
        base_model = eval(model_name+self.params_string)
        self.base_model = base_model

    def add_parameters(self,params:dict): 
        if not type(params) == 'dict': 
            raise(ValueError('Parameters must be specified as a dictionary.'))
        args_string = '('
        first = True
        for key in params: 
            if key not in ['input_shape', 'include_top', 'weights']: 
                print('Unsupported parameter: ' + key + '. Ignoring.')
            else: 
                if not first: 
                    args_string += ','
                else: 
                    first = False
                args_string += key + '=' + params.key 
        self.args_string = args_string

    def add_final_layers(self): 
        # TODO: Add ability to accept dictionary of layers 
        self.model = Sequential([
            self.base_model,
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

    def main(self,model_path,base_trainable=True,lr=0.0001): 


        opt = tf.keras.optimizers.Adam(lr)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC()])
        callbacks = make_callbacks(model_path)
        run_model(model,callbacks)

    def make_callbacks(model_path):
        model_path = os.path.join(model_path,"model.h5")
        csv_path = os.path.join(model_path,"data.csv") 
        callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1),
            CSVLogger(csv_path),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
        ]

    def run_model(model,callbacks): 
        # Use steps per epoch rather than batch size since random perumtations of the training data are used, rather than all training data
        h1 = model.fit(
            x = train_loader, 
            steps_per_epoch = TR_STEPS, 
            epochs = 40,
            validation_data = valid_loader, 
            validation_steps = VA_STEPS, 
            verbose = 1, 
            callbacks=callbacks
        )

# TODO: Define loader and steps calculator
# TODO: Make into class
# TODO: Write unite tests