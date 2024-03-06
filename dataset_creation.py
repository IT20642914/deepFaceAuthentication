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

IMAGES_PER_FOLDER = 400  
POS_PATH = os.path.join('data', 'positive')
ANC_PATH = os.path.join('data', 'anchor')
# Make the directories
# Make the directories
os.makedirs(POS_PATH)
os.makedirs(ANC_PATH)
 #Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
# Function to process video and save images
def process_video(video_path, pos_path, anc_path, images_per_folder):
    cap = cv2.VideoCapture(video_path)

    pos_count = 0
    anc_count = 0

    while cap.isOpened() and (pos_count < images_per_folder or anc_count < images_per_folder):
        ret, frame = cap.read()

        if not ret:  # End of the video
            break
        
        if random.random() < 0.5 and pos_count < images_per_folder:  
            save_path = pos_path
            pos_count += 1
        elif anc_count < images_per_folder: 
            save_path = anc_path
            anc_count += 1
        else:
            continue  # Skip if both folders are near their limit
        
        resized_frame = cv2.resize(frame, (250, 250)) 
        imgname = os.path.join(save_path, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, resized_frame)

    cap.release()

# Function to perform data augmentation
def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
        pass
    return data

# Main function for dataset creation
def create_dataset():
    video_path = 'feem_295E1_IMG_3187.mp4'
    process_video(video_path, POS_PATH, ANC_PATH, IMAGES_PER_FOLDER)

    for file_name in os.listdir(os.path.join(POS_PATH)):
        img_path = os.path.join(POS_PATH, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img) 
        for image in augmented_images:
            cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())
            


if __name__ == "__main__":
    create_dataset()
