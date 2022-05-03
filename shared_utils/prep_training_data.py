from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import numpy as np
import os
from os_utils import list_dir

class ImgMaskDataset: 
    def __init__(self,data_path): 
        # This class is used to initialize a dataset for image segmentation. 
        # It expects parallel image and mask folders to be located in a common data_path
        # There should be one mask for every image, with identical names 
        # Some preprocessing tools are also defined. 
        self.data_path = data_path
        self.use_default_subdirs()
        self.set_height_width((256,256))
        self.set_batch_size(32)


    def use_default_subdirs(self): 
        path = os.path.join(self.data_path,'images')
        self.images = list_dir(path,target_type='files')
        path = os.path.join(self.data_path,'masks')
        self.masks = list_dir(path,target_type='files')
    
    def set_subdirs(self,path_images,path_masks): 
        # Reset locations of images and masks folders if named in a non-standard way
        self.images = list_dir(path_images,target_type='files')
        self.masks = list_dir(path_masks,target_type='files')

    def set_height_width(self,dims:tuple): 
        if min(dims) > 50 and max(dims) <10001: 
            self.dims = dims
        else: 
            raise(ValueError('Check image size: ' + str(dims[0]) + 'x' + str(dims[1])))
    
    def set_batch_size(self,batch_size): 
        self.batch_size = batch_size

    def prep_data_img_labels(self):
        batch_size = self.batch_size
        """ Dataset """
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(self.images,self.masks)
        train_x, train_y = shuffle(train_x, train_y)

        print(f"Train: {len(train_x)} - {len(train_y)}")
        print(f"Valid: {len(valid_x)} - {len(valid_y)}")
        print(f"Test: {len(test_x)} - {len(test_y)}")

        self.train_dataset = tf_dataset(train_x, train_y, batch_size, self.dims)
        self.valid_dataset = tf_dataset(valid_x, valid_y, batch_size, self.dims)

        self.train_steps = (len(train_x)//batch_size)
        self.valid_steps = (len(valid_x)//batch_size)

        if len(train_x) % batch_size != 0:
            self.train_steps += 1

        if len(valid_x) % batch_size != 0:
            self.valid_steps += 1

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y, dims):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([dims[0], dims[1], 3])
    y.set_shape([dims[0], dims[1], 1])
    return x, y

def tf_dataset(train_x, train_y, batch=8,dims=(256,256)):
    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    # Preprocess images and masks
    dataset = dataset.map(lambda x, y: tf_parse(x,y,dims))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def load_data(images, masks, split=0.2):
    size = int(len(images) * split)

    # Split data into train and val sets
    train_x, valid_x = train_test_split(images, test_size=size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=size, random_state=42)

    # Further split off a test set from train
    train_x, test_x = train_test_split(train_x, test_size=size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


# if __name__ == "__main__":
#     main()
