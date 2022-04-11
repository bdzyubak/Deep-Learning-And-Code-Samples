import io
import os
import shutil
import sys

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization 
# import tensorboard 

# Import shared utilities part of this repository by adding the utilities path for the current python session
package_location = os.path.dirname(os.path.dirname(__file__))
utils_location = os.path.join(package_location,'shared_utils')
sys.path.append(utils_location)
from archive_utils import download_and_extract, make_download_paths_default
from string_tools import custom_standardization
from os_utils import make_new_dirs
this_script_location = os.path.dirname(__file__) # Derive from location of this tutorial 
data_dir_top = os.path.join(this_script_location,'data_word_embeddings')


# This is a modified Tutorial based on https://www.tensorflow.org/text/guide/word_embeddings. 
# The intent is to firstly reorganize code into easily maintanable production-like rather than tutorial-like modules, 
# and make some quality-of-life improvements. Subsequently, the code would be expanded with better models, prediction 
# on new text-based datasets, and other functionality. Any functions useable for other projects (e.g. general text 
# sentiment recognition or video classification) will be pulled out into a shared utilities module. 

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
download_fresh = False
script_dir = os.path.dirname(__file__)

def main():     
    train_ds, val_ds = fetch_data()

    # Debug interactively to study the embedding layer
    # study_embedding_layer() 

    # Convert text reviews (strings) into arrays of vocabulary indeces. Each index corresponds to an embedded layer made of embedding_dim numbers
    vocab_size = 10000
    sequence_length = 100
    vectorize_layer = vectorize_inputs(train_ds,vocab_size,sequence_length)

    # Define model architecture 
    model = define_model(vectorize_layer,vocab_size)

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

    weights = model.get_layer('embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()
    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()
    

def fetch_data():
    # Download and unzip data
    path_to_archive, unpacked_dir = make_download_paths_default(url, data_dir_top)
    if download_fresh: 
        download_and_extract(url, path_to_archive, unpacked_dir) 

    # Fetch files as tensorflow datasets
    batch_size = 1024
    seed = 123
    train_ds = tf.keras.utils.text_dataset_from_directory(os.path.join(unpacked_dir,'train'), 
        batch_size=batch_size, validation_split=0.2,
        subset='training', seed=seed)
    val_ds = tf.keras.utils.text_dataset_from_directory(os.path.join(unpacked_dir,'train'),
        batch_size=batch_size, validation_split=0.2,
        subset='validation', seed=seed)

    for text_batch, label_batch in train_ds.take(1):
        for i in range(5):
            print(label_batch[i].numpy(), text_batch.numpy()[i])

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
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


def define_model(vectorize_layer,vocab_size):       
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