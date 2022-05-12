from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import numpy as np
import os
from os_utils import list_dir

class Dataset: 
    def __init__(self,data_path): 
        # This class is used to initialize a dataset for image segmentation. 
        # It expects parallel image and mask folders to be located in a common data_path
        # There should be one mask for every image, with identical names 
        # Some preprocessing tools are also defined. 
        self.data_path = data_path
        self.path_images, self.images = self.use_default_subdirs(img_type='images')
        self.set_image_dims_original()
        self.image_dims_target = self.image_dims_original
        self.set_batch_size(32) 

    def use_default_subdirs(self,img_type): 
        path = os.path.join(self.data_path,img_type)
        self.set_subidrs(path)
        images = self.get_image_and_mask_list(path)
        self.num_examples = len(images)
        return path, images

    def get_max_batch_set(self): 
        self.num_examples

    def set_subidrs(self,path:str): 
        images = self.get_image_and_mask_list(path)
        return images
    
    def get_image_and_mask_list(self,path): 
        # Reset locations of images and masks folders if named in a non-standard way
        images = list_dir(path,target_type='files')
        if not images: 
            raise(FileNotFoundError('Cannot find image files in: ' + images))
        return images
        
    def set_image_target_dims(self,dims:tuple): 
        if len(dims) != 3: 
            raise(ValueError('Specify image dimensions as a tuple of (Height,Width,Channels).'))

        if min(dims[0:2]) < 100 and max(dims[0:2]) >10000: 
            print('Unexpected image size: ' + str(dims[0]) + 'x' + str(dims[1]))
            self.image_dims_target = dims
        self.image_dims_target = dims
    
    def set_image_dims_original(self): 
        image_path = self.images[0]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        dims = image.shape
        if len(dims) <2 or len(dims)>3: 
            raise(ValueError('Unsupported image shape of ' + str(dims) + 'in ' + image_path))

        if len(dims) == 2: 
            self.image_dims_original = (dims[0],dims[1],1)
        else: 
            self.image_dims_original= dims
    
    def set_batch_size(self,batch_size): 
        self.batch_size = batch_size

    def prep_data_img_labels(self,batch_size=''):
        if not batch_size: 
            batch_size = self.batch_size
        """ Dataset """
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = self.load_data(self.images,self.masks)
        train_x, train_y = shuffle(train_x, train_y)

        print(f"Train: {len(train_x)} - {len(train_y)}")
        print(f"Valid: {len(valid_x)} - {len(valid_y)}")
        print(f"Test: {len(test_x)} - {len(test_y)}")

        self.train_dataset = self.tf_dataset(train_x, train_y)
        self.valid_dataset = self.tf_dataset(valid_x, valid_y)

        self.train_steps = (len(train_x)//batch_size)
        self.valid_steps = (len(valid_x)//batch_size)

        if len(train_x) % batch_size != 0:
            self.train_steps += 1

        if len(valid_x) % batch_size != 0:
            self.valid_steps += 1

    def shuffling(self,x, y):
        x, y = shuffle(x, y, random_state=42)

    def read_image(self,path,image_type='image'):
        if not isinstance(path,str): 
            path = path.decode()
        if image_type == 'image': 
            x = cv2.imread(path, cv2.IMREAD_COLOR)
        elif image_type == 'mask': 
            x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else: 
            raise(ValueError("Trying to read unrecognized image type. Should be 'image' or 'mask'"))
        
        if self.image_dims_original != self.image_dims_target: 
            x = cv2.resize(x, (self.dims[0], self.dims[1]))
        x = x/255.0
        x = x.astype(np.float32)
        if len(x.shape)==2: 
            x = np.expand_dims(x, axis=-1)
        return x

    def tf_parse(self, x, y):
        def _parse(x, y):
            x = self.read_image(x,image_type='image')
            y = self.read_image(y,image_type='mask')
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
        x.set_shape([self.image_dims_original[0], self.image_dims_original[1], 3])
        y.set_shape([self.image_dims_original[0], self.image_dims_original[1], 1])
        return x, y

    def tf_dataset(self, train_x, train_y):
        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        # Preprocess images and masks
        dataset = dataset.map(self.tf_parse)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_data(self, images, masks, split=0.2):
        size = int(len(images) * split)

        # Split data into train and val sets
        train_x, valid_x = train_test_split(images, test_size=size, random_state=42)
        train_y, valid_y = train_test_split(masks, test_size=size, random_state=42)

        # Further split off a test set from train
        train_x, test_x = train_test_split(train_x, test_size=size, random_state=42)
        train_y, test_y = train_test_split(train_y, test_size=size, random_state=42)

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


class ImgMaskDataset(Dataset): 
    def __init__(self,data_path): 
        Dataset.__init__(self,data_path)
    
        self.path_masks, self.masks = self.use_default_subdirs(img_type='masks')
        if len(self.images) != len(self.masks): 
            raise(FileNotFoundError('Mismatched number of image and mask files in: ' + os.path.dirname(self.path_images)))

# class ImgLabelDataset(Dataset): 



# class Augment(tf.keras.layers.Layer):
#   # Use train_batches = (train_images.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().map(Augment()).prefetch(buffer_size=tf.data.AUTOTUNE))
#   # test_batches = test_images.batch(BATCH_SIZE)
#   def __init__(self, seed=42):
#     super().__init__()
#     # both use the same seed, so they'll make the same random changes.
#     self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
#     self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

#   def call(self, inputs, labels):
#     inputs = self.augment_inputs(inputs)
#     labels = self.augment_labels(labels)
#     return inputs, labels

# if __name__ == "__main__":
#     main()
