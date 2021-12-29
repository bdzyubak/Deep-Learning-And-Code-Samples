import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization 
import tensorboard 


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
download_fresh = True
script_dir = os.path.dirname(__file__)

def main():     
    train_ds, val_ds = fetch_data(url,download_fresh)

    # Debug interactively to study the embedding layer
    # study_embedding_layer() 

    # Define model
    vocab_size = 10000
    sequence_length = 100
    model = define_model(train_ds,vocab_size, sequence_length)

    # Define tensorboard callback which will monitor a log directory and construct a training chart
    logdir = os.path.join(script_dir,"logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    # Train model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.summary()
    
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard_callback])
    print('Done training model.')

    # To view training history in Tensorboard run the following in debug mode and click the url to open it in default web browser: 
    # tensorboard --logdir=logdir

def fetch_data(url,download_fresh):
    # Download to the location of this script file 
    # Automatically unzips 
    this_script_location = os.path.dirname(__file__) # Derive from location of this tutorial
    download_file_name = os.path.basename(url).split('.')[0] # Derive from download link address
    dataset_dir = os.path.join(this_script_location,download_file_name)
    if download_fresh: 
        unpack_dir = tf.keras.utils.get_file(fname=dataset_dir, origin=url,
                                    extract=True, cache_dir='.',
                                    cache_subdir='')
    else: 
        unpack_dir = r'C:\Source\DL_Tensorflow\aclImdb' # TempFix: Unknown why unpacking happens to a place different than the specified fname

    train_dir = os.path.join(unpack_dir, 'train')
    os.listdir(train_dir) 

    remove_dir = os.path.join(train_dir, 'unsup')
    if os.path.exists(remove_dir): # Expected only when download_fresh=True but may be retained from partial run
        shutil.rmtree(remove_dir) 

    batch_size = 1024
    seed = 123
    train_ds = tf.keras.utils.text_dataset_from_directory(os.path.join(unpack_dir,'train'), 
        batch_size=batch_size, validation_split=0.2,
        subset='training', seed=seed)
    val_ds = tf.keras.utils.text_dataset_from_directory(os.path.join(unpack_dir,'train'),
        batch_size=batch_size, validation_split=0.2,
        subset='validation', seed=seed)

    for text_batch, label_batch in train_ds.take(1):
        for i in range(5):
            print(label_batch[i].numpy(), text_batch.numpy()[i])

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def study_embedding_layer(embedding_layer):
    # Embed a 1,000 word vocabulary into 5 dimensions.
    embedding_layer = tf.keras.layers.Embedding(1000, 5) 
    
    # Test by embedding three integers into this 5-channel layer
    # The weights have not been trained, yet - they are initialized randomly
    result = embedding_layer(tf.constant([1, 2, 3]))
    result.numpy()

    # For text problems pass 2D tensors of shape (samples, sequence_lenght) 
    # The number of channels will be the last dimension 
    result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
    result.shape # 2 samples, of length 3 each, being put into a 5-channel layer 

    result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
    result.shape # 2 samples, of length 3 each, being put into a 5-channel layer


def custom_standardization(input_data): 
    lowercase = tf.strings.lower(input_data)
    stripped_html_newlines = tf.strings.regex_replace(lowercase, '<br />', ' ')
    stripped_html_punctuation = tf.strings.regex_replace(stripped_html_newlines,
                                  '[%s]' % re.escape(string.punctuation), '')
    return stripped_html_punctuation


def vectorize_inputs(train_ds,vocab_size=10000,sequence_length=100): 
    # Use TextVectoriztion imported from tensorflow.keras.layers. 
    # Pass custom_standardization as a function reference to be applied to every data instance
    vectorize_layer = TextVectorization(standardize = custom_standardization, max_tokens=vocab_size, 
    output_mode = 'int', output_sequence_length=sequence_length)
    # Return text only by iteratively splitting train_ds and leave out the labels 
    text_ds = train_ds.map(lambda x, y: x)
    # Learn the word-integer vocabulary 
    vectorize_layer.adapt(text_ds)
    return vectorize_layer


def define_model(train_ds, vocab_size, sequence_length):   
    # Convert text reviews (strings) into arrays of vocabulary indeces. Each index corresponds to an embedded layer made of embedding_dim numbers
    vectorize_layer = vectorize_inputs(train_ds,vocab_size,sequence_length)
    
    # Construct a Sequential keras model. Each layer uses the number of channels of previous layer as its input size 
    # To my current knowledge, only one connection between layers is allowed i.e. U-NET cannot be constructed using sequential API
    embedding_dim=16
    model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name="embedding"),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)]) 
    return model


if __name__ == "__main__": 
    # Call script as a function so that it can be at the top (for readability) rather than following every function that it uses
    main()