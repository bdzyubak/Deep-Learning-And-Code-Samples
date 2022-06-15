import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
import importlib
import os
from os_utils import make_new_dirs
from string_utils import get_trailing_digits
from metrics import dice_coef
from copy import deepcopy
import shutil
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pandas as pd

# NOTE: Tested on Tensorflow 2.9.1 - earlier versions do not support all of these models
valid_models_builtin = {'efficientnet':'class','efficientnet_v2':'class','densenet':'class', 'vgg':'class',
 'inception':'class', 'inception_resnet': 'class','resnet': 'class', 'resnet_v2': 'class', 'resnet_rs': 'class',
 'regnet': 'class', 'mobilenet': 'class', 'mobilenet_v2': 'class', 'mobilenet_v3': 'class', 'xception': 'class'} 
valid_models_custom = {'unet':'segm'}

valid_model_names_all = deepcopy(valid_models_builtin)
valid_model_names_all.update(valid_models_custom)
valid_opt_params = ['learning_rate','num_epochs','batch_size']

class InitializeModel(): 
    def __init__(self,model_name,dataset,model_path_top,base_trainable=True,train_fresh=False): 
        self.model_name = model_name.lower()
        self.model_path = os.path.join(model_path_top,model_name)
        self.train_fresh = train_fresh # Alternative is to train the model fresh, ignoring saved trained model and csv log
        self.get_complexity_from_model_name()
        self.set_model_type()
        make_new_dirs(self.model_path,clean_subdirs=train_fresh)
        self.dataset = dataset
        
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

    def set_model_type(self):
        if self.model_name_base not in valid_model_names_all: 
            raise(ValueError('Unsupported model: ' + self.model_name_base))
        if valid_model_names_all[self.model_name_base] == 'class': 
            self.model_type = 'class'
        elif valid_model_names_all[self.model_name_base] == 'segm': 
            self.model_type = 'segm'
        else: 
            raise(ValueError('Unsupported model type: ' + valid_model_names_all[self.model_name_base]))

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
        model_args = {'input_shape':self.dataset.image_dims_original}
        if self.model_name_base in valid_models_builtin: 
            module = importlib.import_module(import_statement)
            model = getattr(module,method_name)
            if valid_models_builtin[self.model_name_base] == 'class': 
                model_args['include_top'] = False
        elif self.model_name_base in valid_models_custom: 
            spec = importlib.util.spec_from_file_location(os.path.basename(import_statement), import_statement)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = getattr(module,method_name)
            # For now, includ_top is not supported for segmentatin - train model from scratch
        base_model = model(**model_args)
        
        return base_model

    def make_import_statement(self): 
        model_name = self.model_name
        import_statement = ''
        method_name = ''
        if self.model_name_base in valid_models_builtin: 
            import_statement, method_name = self.make_import_statement_builtin()
        elif model_name.startswith('unet'): 
            import_statement = self.make_path_to_custom_models()
            method_name = 'build_unet'
        
        if not import_statement or not method_name: 
            self.check_valid_model_names()
            raise(ValueError('Failed to make ' + model_name + ' model with generalized initializer.'))
        return import_statement, method_name

    def make_import_statement_builtin(self):
        # Default name/method name
        import_statement = 'tensorflow.keras.applications.' + self.model_name_base
        
        # Modify default for some models
        if self.model_name_base.startswith('efficientnet'): 
            method_name = 'EfficientNet'
            method_name = self.append_version(method_name)
            self.model_complexity = self.model_complexity.upper()
        elif self.model_name_base == 'densenet': 
            method_name = 'DenseNet'
            self.check_model_complexity_spec(method_name)
        elif self.model_name_base == 'vgg': 
            method_name = 'VGG'
            import_statement += self.model_complexity
            self.check_model_complexity_spec(method_name)
        elif self.model_name_base == 'inception': 
            method_name = 'InceptionV3'
            import_statement += '_v3'
        elif self.model_name_base == 'inception_resnet': 
            method_name = 'InceptionResNetV2'
            import_statement += '_v2'
        elif self.model_name_base.startswith('resnet'): 
            method_name = 'ResNet'
            self.check_model_complexity_spec(method_name)
            if '_rs' in self.model_name_base: 
                method_name += 'RS'
        elif self.model_name_base.startswith('regnet'): 
            method_name = 'RegNet'
            self.model_complexity = self.model_complexity.upper()
            self.check_model_complexity_spec(method_name)
        elif self.model_name_base.startswith('mobilenet'): 
            method_name = 'MobileNet'
            method_name = self.append_version(method_name)
            import_statement = 'tensorflow.keras.applications'
        elif self.model_name_base.startswith('xception'): 
            method_name = 'Xception'
            method_name = self.append_version(method_name)
        else: 
            raise(ValueError('Failed to split model name and complexity: ' + self.model_name_base))
        # elif self.model_name_base == 'nasnet': 
        #     method_name = 'NASNetLarge'
        
        method_name += self.model_complexity
        if self.model_name_base.startswith('resnet'): 
            method_name = self.append_version(method_name)
        
        return import_statement, method_name
    
    def append_version(self,method_name): 
        if '_v2' in self.model_name_base: 
                method_name += 'V2'
        elif '_v3' in self.model_name_base: 
                method_name += 'V3'
        return method_name

    def check_model_complexity_spec(self,method_name): 
        if not self.model_complexity: 
            print('Refer to : https://www.tensorflow.org/api_docs/python/tf/keras/applications. For model complexity options.')
            raise(ValueError('Must specify complexity: ' + method_name + ' followed by a valid architecture size.'))

    def make_path_to_custom_models(self): 
        package_top = os.path.dirname(os.path.dirname(__file__))
        model_module_path = os.path.join(package_top,'common_models',self.model_name,self.model_name+'_model.py')
        return model_module_path

    def get_complexity_from_model_name(self): 
        model_name_base = [name for name in valid_model_names_all if self.model_name.startswith(name)]
        if not model_name_base: 
            raise(ValueError('Requested base model not supported: ' + self.model_name))
        else:
            self.model_name_base = max(model_name_base,key=len)
            self.model_complexity = self.model_name.split(self.model_name_base)[-1]
            self.model_complexity = self.model_complexity.replace('small','Small')
            self.model_complexity = self.model_complexity.replace('large','Large')
        if len(self.model_complexity) > 12: 
            raise(ValueError('Model complexity spec too long.')) # Security limiter on arbitrary inputs

    def add_final_layers(self,base_model): 
        # TODO: Add ability to accept dictionary of layers 
        if self.model_type == 'segm': 
            # For now, don't do transfer learning with segmentation
            self.add_final_layers_segm(base_model)
        else: 

            self.add_final_layers_class(base_model)
    
    def add_final_layers_segm(self,base_model): 
        self.model = base_model
    
    def add_final_layers_class(self,base_model): 
            self.model = Sequential([
            base_model,
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(2, activation='softmax') # TODO: Use softmax for multiclass and sigmoid for single class. 
        ])

    def set_default_optimization_params(self): 
        self.batch_size = 32
        self.learning_rate_base = 1e-3   # 0.0001
        self.num_epochs = 40 # Set to 1 to debug

    def make_callbacks(self):
        trained_model_file = os.path.join(self.model_path,"trained_model_" + self.model_name + ".h5")
        csv_path = os.path.join(self.model_path,"training_history_"+self.model_name+".csv") 
        if (os.path.exists(trained_model_file) and os.path.exists(csv_path)) and not self.train_fresh: 
            continue_training = not self.train_fresh
            self.model.load_weights(trained_model_file)
        else: 
            continue_training = False # Leave the self.continue_training alone for potential exentions to running as a service
            print('Starting training fresh. No model or history in: ' + trained_model_file)

        self.timing_callback = TimingCallback()
        self.callbacks = [
            ModelCheckpoint(trained_model_file, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1),
            CSVLogger(csv_path,append=continue_training),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            self.timing_callback
        ]

        self.trained_model_file = trained_model_file
        self.history_file = csv_path
    
    def log_timings(self): 
        df = pd.read_csv(self.history_file)
        df['epoch_time'] = self.timing_callback.logs
        df.to_csv(self.history_file)

    def set_loss_function(self,loss='binary_crossentropy'): 
        # For now, use binary cross entropy appropriate for mutually exclusive classes
        # Categorical_crossentry could be used for non-mutually exclusive classes
        self.loss = loss

    def set_optimizer(self,learning_rate=''): 
        # For now, always use Adam
        if not learning_rate: 
            learning_rate = self.learning_rate_base
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def set_metrics(self): 
        # Dice is appropriate for segmentation tasks, accuracy can be used for classification tasks
        self.metrics =[dice_coef,'accuracy']

    def run_model(self): 
        # Use steps per epoch rather than batch size since random perumtations of the training data are used, rather than all training data
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.timing_callback.clear_logs()

        self.history = self.model.fit(
            x = self.dataset.train_dataset, 
            steps_per_epoch = self.dataset.train_steps, 
            epochs = self.num_epochs, 
            validation_data = self.dataset.valid_dataset, 
            validation_steps = self.dataset.valid_steps, 
            verbose = 1, 
            callbacks=self.callbacks
        )

        self.log_timings()
        return self.history

    def try_bypass_local_minimum(self,n_times=1): 
        # This method is a way to bypass local minima without using stochastic small-batch training. 
        # Large batches are faster to train but their optimization is smoother and more prone to getting stuck. 
        # Using large batches and a reasonably high initial starting learning rate, the model should hit a decent accuracy. 
        # This function can than be used to look for an even better minimum. 
        # TODO: Consider reducing batch size as well. Initial training can be done with batches as large as memory will allow
        # but local minima avoidence is better with mini-batches e.g. 32 or 64 examples
        backup_models = dict() # Model name: csv file
        backup_models['jump_iter0'] = [self.trained_model_file,self.history_file,self.get_final_epoch_coeff('dice_coef')]
        
        for i in range(1,n_times+1): 
            backup_file_model = self.make_backup_file(self.trained_model_file,i)
            backup_file_csv = self.make_backup_file(self.history_file,i)
            self.set_optimizer()
            hist = self.run_model() # The optimizer is re-compiled inside to reset individual weights adjusted by Adam
            backup_models['jump_iter'+str(i)] = [backup_file_model,backup_file_csv,self.get_final_epoch_coeff('dice_coef')]
        
        print('Done trying to bypass minima.')
        self.pick_best_model(backup_models)
        
    def make_backup_file(self, filename, i):
        extension = '.'+filename.split('.')[-1]
        backup_file_name = filename.split(extension)[-2] + '_backup'+str(i) + extension
        shutil.copy(filename,backup_file_name)
        return backup_file_name
    
    def get_final_epoch_coeff(self,type='dice_coef'): 
        last_iter_coeff = round(self.history.history[type][-1],3)
        return last_iter_coeff

    def pick_best_model(self,backup_models): 
        # The default is to compare a value like dice or accuracy where higher is better
        # If optimal metric is low (e.g. comparing loss) - define a flip scenario in get_final_epoch_coeff and use that as input
        best_acc = 0
        best_model = ''
        for model in backup_models: 
            accuracy = backup_models[model][2]
            if best_acc>0: 
                best_acc = accuracy
            elif accuracy > best_acc: 
                best_model = model

        if best_model: 
            print('Updating best model with: ' + best_model)
            shutil.copy(backup_models[best_model][0],self.trained_model_file) 
            shutil.copy(backup_models[best_model][1],self.history_file) 
        else: 
            print('Original model was the best.')

    def vis_training(self, start=1):
        history = self.history
        epoch_range = range(start, len(history['loss'])+1)
        s = slice(start-1, None)

        plt.figure(figsize=[14,4])

        n = int(len(history.keys()) / 2)

        for i in range(n):
            k = list(h.keys())[i]
            plt.subplot(1,n,i+1)
            plt.plot(epoch_range, history[k][s], label='Training')
            plt.plot(epoch_range, history['val_' + k][s], label='Validation')
            plt.xlabel('Epoch'); plt.ylabel(k); plt.title(k)
            plt.grid()
            plt.legend()

        plt.tight_layout()
        plt.show()

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)
    def clear_logs(self): 
        self.logs=[]