import cv2
import os
import uuid
import random
import numpy as np
from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy 

# Function to preprocess images
def preprocess(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (100, 100))
    img = img / 255.0  # Normalize pixel values
    return img
# Function to calculate L1 distance
class L1Dist(Layer):  
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
# Function to verify
def verify(model, input_image_path, verification_images_folder, detection_threshold=0.9, verification_threshold=0.7):
    input_img = preprocess(input_image_path)
    results = []

    for image_name in os.listdir(verification_images_folder):
        validation_img = preprocess(os.path.join(verification_images_folder, image_name))
        result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(verification_images_folder))
    verified = verification > verification_threshold

    return verified

def main():

    siamese_model = tf.keras.models.load_model('siamese_model.h5', 
                                    custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


    # Paths
    input_image_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
    verification_images_folder = os.path.join('application_data', 'verification_images')

    # Verify
    verified = verify(siamese_model, input_image_path, verification_images_folder)

    if verified:
        print("Verification Successful!")
    else:
        print("Verification Failed!")

    # Capture video from webcam for real-time verification
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        # frame = frame[120:120+250, 200:200+250, :]
        cv2.imshow('Verification', frame)

        if cv2.waitKey(10) & 0xFF == ord('v'):
            cv2.imwrite(input_image_path, frame)
            verified = verify(siamese_model, input_image_path, verification_images_folder)
            if verified:
                print("Verification Successful!")
            else:
                print("Verification Failed!")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
