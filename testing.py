#%%
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import utils
from preprocessing import preprocessor
from resnet import ResidualBlock


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#%%
model34 = keras.models.Sequential()
model34.add(keras.layers.Conv2D(64, 5, strides=1, input_shape=[32, 32, 1],
                              padding="same", use_bias=False))
model34.add(keras.layers.BatchNormalization())
model34.add(keras.layers.Activation("relu"))
model34.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

filter_list = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3

prev_filters = []
#conv_block + (n-1) * id_blocks
for filters in filter_list:        #list of filters to be used
    strides = 1 if filters == prev_filters else 2                   #making images smaller at every change of number of filters
    model34.add(ResidualBlock(filters, strides=strides))
    prev_filters = filters

model34.add(keras.layers.GlobalAvgPool2D())
model34.add(keras.layers.Flatten())
model34.add(keras.layers.Dense(43, activation="softmax"))             #fully connected layer outputting 43 classes

#%%
model18 = keras.models.Sequential()
model18.add(keras.layers.Conv2D(64, 5, strides=1, input_shape=[32, 32, 1],
                              padding="same", use_bias=False))
model18.add(keras.layers.BatchNormalization())
model18.add(keras.layers.Activation("relu"))
model18.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

filter_list = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2

prev_filters = []
#conv_block + (n-1) * id_blocks
for filters in filter_list:        #list of filters to be used
    strides = 1 if filters == prev_filters else 2                   #making images smaller at every change of number of filters
    model18.add(ResidualBlock(filters, strides=strides))
    prev_filters = filters

model18.add(keras.layers.GlobalAvgPool2D())
model18.add(keras.layers.Flatten())
model18.add(keras.layers.Dense(43, activation="softmax"))             #fully connected layer outputting 43 classes

# %%
model50 = keras.models.Sequential()
model50.add(keras.layers.Conv2D(64, 5, strides=1, input_shape=[32, 32, 1],
                              padding="same", use_bias=False))
model50.add(keras.layers.BatchNormalization())
model50.add(keras.layers.Activation("relu"))
model50.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

filter_list = [[64,256]] * 3 + [[128,512]] * 4 + [[256,1024]] * 6 + [[512,2048]] * 3

prev_filters = []
#conv_block + (n-1) * id_blocks
for filters in filter_list:        #list of filters to be used
    strides = 1 if filters == prev_filters else 2                   #making images smaller at every change of number of filters
    model50.add(ResidualBlock(filters, strides=strides))
    prev_filters = filters

model50.add(keras.layers.GlobalAvgPool2D())
model50.add(keras.layers.Flatten())
model50.add(keras.layers.Dense(43, activation="softmax"))             #fully connected layer outputting 43 classes


#%%
#loading processed test images and test labels
procTestImages = joblib.load('procTestImages.joblib')
utils.sample_images(procTestImages, seed_num=42)
testLabels = joblib.load('test_labels.joblib')

#%%
img_tweaks = {
    'featurewise_center':True,
    'featurewise_std_normalization':True,
    'rotation_range':20,
    'width_shift_range':0.2,
    'height_shift_range':0.2,
}

datagen = ImageDataGenerator(**img_tweaks)

datagen.fit(procTestImages)
Xy_test = datagen.flow(procTestImages, testLabels, batch_size=32) #creating data augmented tensor of test images

#%%
folders = ['adam 0.001', 'adam 0.0001','nadam 0.001', 'nadam 0.0001']
optimizers = [keras.optimizers.Adam(learning_rate = 0.001), keras.optimizers.Adam(learning_rate = 0.0001), keras.optimizers.Nadam(learning_rate = 0.001), keras.optimizers.Nadam(learning_rate = 0.0001)]

#%%

for folder, optimizer in zip(folders,optimizers):
    print(f'Testing with parameters {folder}')
    #loading weights from trained model
    model18.load_weights(f'./runs/{folder}/best_resnet18_weights.hdf5')
    model34.load_weights(f'./runs/{folder}/best_resnet34_weights.hdf5')
    model50.load_weights(f'./runs/{folder}/best_resnet50_weights.hdf5')

    print('Testing ResNet18 on new data')
    model18.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model18.evaluate(Xy_test)

    print('Testing ResNet34 on new data')
    model34.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model34.evaluate(Xy_test)

    print('Testing ResNet50 on new data')
    model50.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model50.evaluate(Xy_test)

# %%
