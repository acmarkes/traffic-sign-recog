#%%
import os
import csv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from glob import glob
from random import sample, seed
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def getPic(img_path, size):
    '''
    Arguments:
        image_path: string of image path
        size: 2-tuple for image resizing
    Returns: image as an array ready for tf
    '''

    return np.array(Image.open(img_path).convert('RGB').resize(size,Image.BILINEAR))


#%%
def readTrafficSigns(rootpath, split, as_size):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: 
        rootpath: path to the traffic sign data, for example './GTSRB/'
        split: training or test
        as_size: 2-tuple to set resizing of imgs
    Returns:   list of images, list of corresponding labels'''

    split = split.capitalize()
    images = [] # images
    labels = [] # corresponding labels
    path = rootpath + '/' + split + '/' + 'Images/'
    folders = os.listdir(path)

    if split.lower() == 'test':
        file = open(path + 'GT-final_test.csv') # annotations file
        reader = csv.reader(file, delimiter=';') # csv parser for annotations file
        next(reader) # skip header
        # loop over all images in current annotations file
        for row in reader:
            images.append(getPic(path + row[0], as_size)) # the 1th column is the filename
            labels.append(int(row[7])) # the 8th column is the label
        file.close()
        
    elif split.lower() == 'training':
        for folder in folders:
            data_dir = Path(path+folder)
            for file in list(data_dir.glob('*.ppm')):
                images.append(getPic(file, as_size)) 
                labels.append(int(folder)) 

    images= np.array(images)
    labels = np.array(labels)

    return images, labels


#%%
def sample_images(dataset, seed_num=None):
    #utility function for plotting of six random images in a dataset
    #dataset = numpy array of images

    if seed_num:
        seed(seed_num)

    imgs = sample(list(dataset), 6) 

    fig, axes = plt.subplots(1, 6, figsize=(15, 10))
    axs = axes.ravel()

    for i,ax in enumerate(axs):
        ax.imshow(np.squeeze(imgs[i]), cmap=plt.cm.gray)

    fig.tight_layout()
    plt.show()
    seed(None)

#%%
def get_dataset_partitions_tf(data, labels, val_split=0.15, seed=42, train_batch_size = 32, val_batch_size = 8, gen_kws={}):

    #function for creation of Training and Validation Tensors with stratified classes and support for data augmentation
    """ input:
            data: numpy array of images
            labels: numpy array of labels
            val_split: (float) [0,1] representing size of validation set
            seed: random state seed for the shuffling of the data
            train_batch_size, val_batch_size: (int) training and validation batch sizes
            gen_kws: (dict) arguments for data augmentation compatible with tensorflow.keras.preprocessing.image.ImageDataGenerator
    """

    #creating stratified splits
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=val_split, stratify=labels, random_state=seed)

    #putting them back together
    data = np.concatenate((X_train, X_val))
    labels = np.concatenate((y_train, y_val))

    #splitting train and validation sets with data augmentation support
    datagen = ImageDataGenerator(validation_split=val_split, **gen_kws)

    datagen.fit(data)

    #split dataset into train and validation sets
    Xy_train = datagen.flow(data, labels, batch_size=train_batch_size, subset='training')
    Xy_val = datagen.flow(data, labels, batch_size=val_batch_size, subset='validation')

    return Xy_train, Xy_val
