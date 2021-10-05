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
        rootpath: path to the traffic sign data, for example './GTSRB/Training'
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
        
    elif split.lowercase() == 'training':
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
""" 
def DEPRECATED_get_dataset_partitions_tf(data, labels, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=5000):
    assert (train_split + test_split + val_split) == 1
    
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()


    if shuffle:
        # Specify seed to always have the same split distribution between runs
        dataset = dataset.shuffle(shuffle_size, seed=42)
   
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    
    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size).skip(val_size)
    
    X_train, y_train = np.array([x.numpy() for x, y in train]), np.array([y.numpy() for x, y in train])
    X_val, y_val = np.array([x.numpy() for x, y in val]), np.array([y.numpy() for x, y in val])
    X_test, y_test = np.array([x.numpy() for x, y in test]), np.array([y.numpy() for x, y in test])

    return X_train, y_train, X_val, y_val, X_test, y_test
    #train, val, test
 """
#%%
def get_dataset_partitions_tf(data, labels, val_split=0.15, seed=42, train_batch_size = 32, val_batch_size = 8, gen_kws={}):

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=val_split, stratify=labels, random_state=seed)

    data = np.concatenate((X_train, X_val))
    labels = np.concatenate((y_train, y_val))

    datagen = ImageDataGenerator(validation_split=val_split, **gen_kws)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(data)

    #split dataset into train and validation sets
    Xy_train = datagen.flow(data, labels, batch_size=train_batch_size, subset='training')
    Xy_val = datagen.flow(data, labels, batch_size=val_batch_size, subset='validation')

    return Xy_train, Xy_val
