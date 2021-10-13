#%%
import joblib

import tensorflow as tf
from tensorflow import keras

import utils
from preprocessing import preprocessor
from resnet import resnet34, resnet50, ResidualBlock
from datetime import datetime as dt

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#%%
#loading processed training images and labels
procTrainImages = joblib.load('procTrainImages.joblib')
utils.sample_images(procTrainImages, seed_num=42)
trainLabels = joblib.load('train_labels.joblib')

#%%
img_tweaks = {
    'featurewise_center':True,
    'featurewise_std_normalization':True,
    'rotation_range':20,
    'width_shift_range':0.2,
    'height_shift_range':0.2,
}
#%%
#splitting dataset and doing data augmentation
Xy_train, Xy_val = utils.get_dataset_partitions_tf(procTrainImages, trainLabels, gen_kws=img_tweaks)

#%%
model = resnet34(43)
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) #early stopping to avoid overfitting and wasting time
filepath = 'best_resnet_weights.hdf5'
checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')    #keeping saved weights of best result only
csv_logger = keras.callbacks.CSVLogger(f'log {dt.now().strftime("%Y-%m-%d %H:%M")} resnet.csv', append=True, separator=';')


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(Xy_train, batch_size=32 ,epochs=30, validation_data=Xy_val, callbacks=[early_stop, checkpoint, csv_logger])



# %%
model50 = resnet50(43)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) #early stopping to avoid overfitting and wasting time
filepath = 'best_resnet50_weights.hdf5'
checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')    #keeping saved weights of best result only
csv_logger = keras.callbacks.CSVLogger(f'log {dt.now().strftime("%Y-%m-%d %H:%M")} resnet50.csv', append=True, separator=';')


model50.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history50 = model50.fit(Xy_train, batch_size=32 ,epochs=30, validation_data=Xy_val, callbacks=[early_stop, checkpoint, csv_logger])
# %%
