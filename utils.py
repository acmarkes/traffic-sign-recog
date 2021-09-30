#%%
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image
from random import sample, seed
import tensorflow as tf

def getPic(img_path, size):
    '''
    Arguments:
        image_path: string of image path
        size: 2-tuple for image resizing
    Returns: image as an array ready for tf
    '''

    return np.array(Image.open(img_path).convert('RGB').resize(size,Image.BILINEAR))


#%%
def readTrafficSigns(rootpath, as_size):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: 
        rootpath: path to the traffic sign data, for example './GTSRB/Training'
        as_size: 2-tuple to set resizing of imgs
    Returns:   list of images, list of corresponding labels'''

    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(getPic(prefix + row[0], as_size)) # the 1th column is the filename
            labels.append(int(row[7])) # the 8th column is the label
        gtFile.close()
    
    images= np.array(images)
    labels = np.array(labels)

    return images, labels

#%%
def sample_images(dataset, seed_num=42):

    seed(seed)
    imgs = sample(list(dataset), 6) 

    fig, axes = plt.subplots(1, 6, figsize=(15, 10))
    axs = axes.ravel()

    for i,ax in enumerate(axs):
        ax.imshow(np.squeeze(imgs[i]),  cmap=plt.cm.gray)

    fig.tight_layout()
    plt.show()


#%%
def get_dataset_partitions_tf(data, labels, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=5000):
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


