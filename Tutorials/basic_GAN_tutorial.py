import tensorflow as tf
# Ignore import warning - tensorflow.keras.layers does exist
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Activation 
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Reshape, Flatten

DROPOUT_RATE = 0.1
MAX_CHANNELS = 256
NUM_LAYERS = MAX_CHANNELS // 64
ACTIVATION = 'LeakyReLU'
USE_BATCHNORM = True

def main(): 
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    input_shape = (x_train.shape[1],x_train.shape[2],1)
    dcgan = DCGAN(input_shape)

class DCGAN(): 
    def __init__(self,input_shape): 
        disc_input = Input(shape=input_shape)
        self.build_discriminator(disc_input)

        gen_input = Input(shape=(100))
        self.build_generator(gen_input)
        # self.adversarial = tf.keras.Sequential([generator,discriminator])
        # self.adversarial.summary()

    def build_discriminator(self,disc_input):     
        filters = int(MAX_CHANNELS / NUM_LAYERS)
        layers = disc_input
        for i in range(NUM_LAYERS-1): 
            layers = self.build_dowsample(layers,filters)
            filters = filters * 2
        layers = self.build_CNN_block(layers,filters)
        layers = Flatten()(layers)
        # Classification
        layers = Dense(1)(layers)
        layers = Activation('sigmoid')(layers)
        self.discrimintor = tf.keras.Model(inputs=disc_input, outputs=layers)
        self.discrimintor.summary()
        
    
    def build_generator(self,gen_input): 
        disc = self.build_dense_block(starting_size=7*7*256,prev_layer=gen_input)
        disc = Reshape((7,7,256))(disc)
        filters = MAX_CHANNELS
        for i in range(NUM_LAYERS-2): 
            disc = self.build_upsample(disc,filters)
            filters = filters // 2
        # Single-channel image generation
        disc = self.build_CNN_block(disc,1,activation='sigmoid')
        self.discrimintor = tf.keras.Model(inputs=gen_input, outputs=disc)
        self.discrimintor.summary()


    def build_upsample(self,prev_layer,filters,activation=ACTIVATION): 
        # Convolution/Upsampling layer
        layer = self.build_CNN_block(prev_layer,filters,rescale='up'
                ,strides=2,activation=activation)
        return layer

    def build_dowsample(self,prev_layer,filters,activation=ACTIVATION): 
        # Convolution/downsampling layer
        layers = self.build_CNN_block(prev_layer,filters,rescale='down'
                ,strides=2,activation=activation)
        return layers

    def build_CNN_block(self,prev_layer,filters,rescale='',strides=1,activation=''): 
        use_bias=True
        if USE_BATCHNORM: 
            use_bias = False
        if not rescale or rescale == 'down': 
            # Downsample
            layers = Conv2D(filters,3,strides=strides,padding='same',use_bias=use_bias)(prev_layer)
            layers = Conv2D(filters,3,padding='same',use_bias=use_bias)(layers)
        elif rescale == 'up':  
            # Upsample
            layers = Conv2DTranspose(filters,3,strides=strides,padding='same',use_bias=use_bias)(prev_layer)
            layers = Conv2D(filters,3,padding='same',use_bias=use_bias)(layers)
        else: 
            raise(ValueError('Unrecognized rescale type.'))

        if USE_BATCHNORM: 
            layers = BatchNormalization()(layers)
        if not activation: 
            activation = ACTIVATION
        layers = Activation(activation)(layers)
        return layers

    # def build_CNN_block(self,prev_layer,filters): 
    #     conv = tf.keras.layers.Conv2D(filters,3)(prev_layer)
    #     conv = tf.keras.layers.Conv2D(filters,3)(conv)
    #     conv = tf.keras.layers.BatchNormalization()
    #     return conv

    def build_dense_block(self,starting_size,prev_layer): 
        dense = Dense(starting_size)(prev_layer)
        dense = BatchNormalization()(dense)
        dense = Dropout(DROPOUT_RATE)(dense)
        return dense

if __name__ == '__main__': 
    main()