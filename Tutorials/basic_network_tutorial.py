import tensorflow as tf

DROPOUT_RATE = 0.1
STARTING_CHANNELS = 256
NUM_LAYERS = 3
ACTIVATION = 'LeakyReLU'

def main(): 
    dcgan = DCGAN()


class DCGAN(): 
    def __init__(self): 
        self.build_generator()
        self.build_discriminator()

    def build_discriminator(self):     
        input = tf.keras.layers.Input(input_shape=(100))
        disc = self.build_dense_block(input)
        
        for i in range(NUM_LAYERS+1): 
            disc = self.build_upsample(disc)
        disc = self.build_CNN_block(disc)
        disc = tf.keras.layers.Activation('sigmoid')
        self.discriminator = tf.keras.Model(inputs=input, outputs=disc)
    
    def build_generator(self): 
        input = tf.keras.layers.Input(input_shape=[100,1])
        disc = self.build_dense_block(input)
        filters = STARTING_CHANNELS
        for i in range(NUM_LAYERS+1): 
            disc = self.build_upsample(disc,filters)
            filters = filters // 2
        disc = self.build_upsample(disc,filters,activation='sigmoid')
        self.discriminator = tf.keras.Model(inputs=input, outputs=disc)

    def build_upsample(self,prev_layer,filters,activation=ACTIVATION): 
        layer = tf.keras.layers.Conv2D(filters,3)(prev_layer)
        layer = tf.keras.layers.UpSampling2D()(layer)
        layer = tf.keras.layers.Activation(activation)(layer)
        return layer

    def build_dowsample(self,prev_layer,filters,activation=ACTIVATION): 
        layer = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2,
                                    padding='same',
                                    use_bias=False)(prev_layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(activation)(layer)
        return layer

    # def build_CNN_block(self,prev_layer,filters): 
    #     conv = tf.keras.layers.Conv2D(filters,3)(prev_layer)
    #     conv = tf.keras.layers.Conv2D(filters,3)(conv)
    #     conv = tf.keras.layers.BatchNormalization()
    #     return conv

    def build_dense_block(self,prev_layer): 
        dense = tf.keras.layer.Dense()(prev_layer)
        dense = tf.keras.layer.BatchNormalization(dense)
        dense = tf.keras.layer.Dropout(DROPOUT_RATE)(dense)
        return dense

if __name__ == '__main__': 
    main()