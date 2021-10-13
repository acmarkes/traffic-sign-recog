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
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 5, strides=1, input_shape=[32, 32, 1],
                              padding="same", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

filter_list = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3

prev_filters = []
#conv_block + (n-1) * id_blocks
for filters in filter_list:        #list of filters to be used
    strides = 1 if filters == prev_filters else 2                   #making images smaller at every change of number of filters
    model.add(ResidualBlock(filters, strides=strides))
    prev_filters = filters

model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(43, activation="softmax"))             #fully connected layer outputting 43 classes


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
#loading weights from trained model
model.load_weights('./runs/best_resnet_weights.hdf5')
model50.load_weights('./runs/best_resnet50_weights.hdf5')


#%%
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.evaluate(Xy_test)

# %%
model50.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
model50.evaluate(Xy_test)

# %%
