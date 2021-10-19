#%%
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import utils
from preprocessing import preprocessor
from resnet import ResidualBlock, resnet18, resnet34, resnet50


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#%%
#loading processed test images and test labels
procTestImages = joblib.load('./data/procTestImages.joblib')
utils.sample_images(procTestImages, seed_num=42)
testLabels = joblib.load('./data/test_labels.joblib')

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
model18 = resnet18(43)
model34 = resnet34(43)
model50 = resnet50(43)


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
