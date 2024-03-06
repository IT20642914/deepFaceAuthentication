import os
import tensorflow as tf
import cv2
import os
import uuid
import random
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy 
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Function to preprocess images for the dataset
def preprocess(file_path):
     # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

# Function to preprocess images for training
def preprocess_twin(input_img, validation_img, label):
   return(preprocess(input_img), preprocess(validation_img), label)

# Function to build the embedding model
def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

# Function to calculate L1 distance
class L1Dist(Layer):  
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Function to build the siamese model
def make_siamese_model():
    inp = Input(name='input_img', shape=(100,100,3))
    validation_img = Input(name='validation_img', shape=(100,100,3))

    embedding = make_embedding()

    inp_embedding = embedding(inp)
    val_embedding = embedding(validation_img)

    siamese_layer = L1Dist()
    distances = siamese_layer(inp_embedding, val_embedding)

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[inp, validation_img], outputs=classifier, name='SiameseNetwork')

# Function for training steps
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:     
        X = batch[:2]
        y = batch[2]
        
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)

    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    return loss

# Function to train the model
def train(data, EPOCHS):
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        r = Recall()
        p = Precision()
        
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)

# Load dataset and train the model
def main():
    anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(3000)
    positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(3000)
    negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(3000)

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    train_data = data.take(round(len(data)*.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    test_data = data.skip(round(len(data)*.7))
    test_data = test_data.take(round(len(data)*.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    global binary_cross_loss
    binary_cross_loss = tf.losses.BinaryCrossentropy() 
    global opt
    opt = tf.keras.optimizers.Adam(1e-4)

    global checkpoint_dir
    checkpoint_dir = './training_checkpoints'
    global checkpoint_prefix
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    # Build the siamese model
    global siamese_model
    siamese_model = make_siamese_model()

    EPOCHS = 1
    train(train_data, EPOCHS)

    r = Recall()
    p = Precision()

    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = siamese_model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat) 

    print(r.result().numpy(), p.result().numpy())
  # Save the trained model
    model_save_path = 'siamese_model.h5'
    siamese_model.save(model_save_path)

if __name__ == "__main__":
    main()
