'''
DL Model for training Fruits360 dataset
Author: Saikiran Tharimena
30 September 2021
dr.saikirant@gmail.com
'''

# Import libraries
import os
import csv
from os.path import join
import numpy as np
from time import time
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Add, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D

import horovod.tensorflow.keras as hvd
from horovod.tensorflow.keras.callbacks import LearningRateWarmupCallback, BroadcastGlobalVariablesCallback, MetricAverageCallback

# Initialize Horovod
hvd.init()

# Pin to a GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Parameters
batch_size = 64
epochs     = 80

warmup_epochs = 5    # No of epochs for which base learning rate will be used
momentum = 0.9       # Momentum for Stochastic Gradient Descent

base_learning_rate = 0.1 # Learning rate for 1 GPU

train_target = 0.8 # Training target accuracy
val_target   = 0.8 # Validation target accuracy

datapath = '/scratch1/05802/tharimen/data/fruits360'

savemodel = '/scratch1/05802/tharimen/dlmodels/model_fruits360'

num_classes = 131

# Set verbose based on GPU rank
if hvd.rank() == 0:
    verbose = 1
else:
    verbose = 0 # prints total time

# Dataset is in organized in folders
# ImageGenerator class will be used
# to read data, resize, and rescale images

# Resize image dimensions
img_rows, img_cols = 100, 100

# Rescale images to 0 - 1
rescale = 1./255.

# Reformat to Keras friendly format
input_shape = (3, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, 3)

# Image Data Generator for training
# This will be used to perform additional processing on the images that
# will help with training
train_datagen = image.ImageDataGenerator(
                featurewise_center=False,            # set input mean to 0 over the dataset
                samplewise_center=False,             # set each sample mean to 0
                featurewise_std_normalization=False, # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,                 # apply ZCA whitening
                zca_epsilon=1e-06,                   # epsilon for ZCA whitening
                rotation_range=0,                    # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)
                shear_range=0.,                      # set range for random shear
                zoom_range=0.,                       # set range for random zoom
                channel_shift_range=0.,              # set range for random channel shifts
                fill_mode='nearest',                 # set mode for filling points outside the input boundaries
                cval=0.,                             # value used for fill_mode = "constant"
                horizontal_flip=True,                # randomly flip images
                vertical_flip=False,                 # randomly flip images
                rescale=rescale,                     # set rescaling factor (applied before any other transformation)
                preprocessing_function=None,         # set function that will be applied on each input
                data_format=None,                    # image data format, either "channels_first" or "channels_last"
                validation_split=0.0)                # fraction of images reserved for validation (strictly between 0 and 1)

# Create training iterator
train_iter = train_datagen.flow_from_directory(join(datapath, 'train'),
                                               target_size=(img_rows, img_cols),
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=False,
                                               seed=100)

# Image Data Generator for test and validation
# This will be used to perform additional processing on the images that
# will help with training
test_datagen = image.ImageDataGenerator(
                featurewise_center=False,            # set input mean to 0 over the dataset
                samplewise_center=False,             # set each sample mean to 0
                featurewise_std_normalization=False, # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                rescale=rescale)                     # set rescaling factor (applied before any other transformation)

# Test iterator
test_iter = test_datagen.flow_from_directory(join(datapath, 'test'),
                                             target_size=(img_rows, img_cols),
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=False,
                                             seed=100)

#%% Generate Model
def create_model(input_shape, baselr, momentum):
    
    # Helper functions - convolution with Batch Normalization
    def conv_batchnorm(x, conv_size, channel_axis):
        
        x = Conv2D(filters=conv_size, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        return x

    # Convolution block - calls conv_batchnorm and adds drop out
    def conv_block(x, conv_size, channel_axis, scale_input = False):
        x_0 = x
        if scale_input:
            x_0 = Conv2D(conv_size, (1, 1), activation='linear', padding='same')(x_0)

        x = conv_batchnorm(x, conv_size, channel_axis)
        x = Dropout(0.01)(x)
        x = conv_batchnorm(x, conv_size, channel_axis)
        x = Add()([x_0, x])

        return x
    
    # Input
    inputs = Input(shape=input_shape)
    
    # Channel axis
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    # Model - 1st layer
    x = conv_batchnorm(inputs, 16, channel_axis)
    
    # 1st conv block with and without scaling
    x = conv_block(x, 160, channel_axis, True)
    x = conv_block(x, 160, channel_axis)
    x = MaxPooling2D((2, 2))(x)
    
    # 2nd conv block
    x = conv_block(x, 320, channel_axis, True)
    x = conv_block(x, 320, channel_axis)
    x = MaxPooling2D((2, 2))(x)
    
    # 3rd conv block
    x = conv_block(x, 640, channel_axis, True)
    x = conv_block(x, 640, channel_axis)
    x = MaxPooling2D((2, 2))(x)
    
    # 4th conv block
    x = conv_block(x, 1280, channel_axis, True)
    x = conv_block(x, 1280, channel_axis)
    x = GlobalAveragePooling2D()(x)
  
    # Output dense layer of num classes with softmax activation
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    opt = SGD(lr=baselr, momentum=momentum)

    # Wrap optimizer in Horovod distributed optimizer
    opt = hvd.DistributedOptimizer(opt)
    
    # Compile
    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model

#%% Callbacks
# Callbacks for printing total time and for early stopping
# For early stopping, train and validation targets are used
# For the first run, training will be run for all epochs
# and for the subsequent run, training will be stopped at a
# set value of validation and training target accuracy

# Total time
class PrintTotalTime(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = round(time() - self.start_time, 2)
        print("Elapsed training time through epoch {}: {}".format(epoch+1, elapsed_time))

    def on_train_end(self, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Total training time: {}".format(total_time)) 

class PrintThroughput(Callback):
    def __init__(self, total_images=0):
        self.total_images = total_images

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.epoch_start_time
        images_per_sec = round(self.total_images / epoch_time, 2)
        print('Images/sec: {}'.format(images_per_sec))

# Stop accuracy
class StopAtAccuracy(Callback):
    def __init__(self, train_target=0.75, val_target=0.25, patience=2, verbose=0):
        self.train_target = train_target
        self.val_target = val_target
        self.patience = patience
        self.verbose = verbose
        self.stopped_epoch = 0
        self.met_train_target = 0
        self.met_val_target = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > self.train_target:
            self.met_train_target += 1
        else:
            self.met_train_target = 0
            
        if logs.get('val_accuracy') > self.val_target:
            self.met_val_target += 1
        else:
            self.met_val_target = 0

        if self.met_train_target >= self.patience and self.met_val_target >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and verbose == 1:
            print('Early stopping after epoch {}. Training accuracy target ({}) and validation accuracy target ({}) met.'.format(self.stopped_epoch + 1, self.train_target, self.val_target))

# Save training data
class SaveTrainingData(Callback):
    def __init__(self, data_filepath=''):
        self.data_filepath = data_filepath

    def on_train_begin(self, logs=None):       
        file = open(self.data_filepath, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(['time', 'val_accuracy'])
        writer.writerow([0.0, 0.0])
        file.close()  

        self.train_start_time = time()

    def on_epoch_end(self, epoch, logs={}):
        total_time = time() - self.train_start_time
        file = open(self.data_filepath, 'a')
        writer = csv.writer(file)
        writer.writerow([round(total_time,1), round(logs['val_accuracy'], 4)])
        file.close()

# Learning rate scheduler
# Base learning rate is set in parameters
# based on the number of epochs, the learning
# rate will be changed. Initial epochs will use
# base learning rate which will be exponentially
# reduced for further epochs
def lr_schedule(epoch):
    if epoch < 15:
        return base_learning_rate
    if epoch < 25:
        return 1e-1 * base_learning_rate
    if epoch < 35:
        return 1e-2 * base_learning_rate
    return 1e-3 * base_learning_rate


#%% Model with stop callback
# Callbacks
callbacks = [LearningRateScheduler(lr_schedule), 
             LearningRateWarmupCallback(initial_lr=base_learning_rate,
                                        warmup_epochs=warmup_epochs,
                                        verbose=verbose),
             BroadcastGlobalVariablesCallback(0),
             MetricAverageCallback(),
             StopAtAccuracy(train_target=train_target, val_target=val_target, verbose=verbose)]

# Append total time and save training data on local node
if verbose:
    callbacks.append(PrintTotalTime())

    data_filepath = "{}ranks-{}bs-{}lr-{}m-{}w-stop.csv".format(hvd.size(), batch_size, base_learning_rate, momentum, warmup_epochs)

    callbacks.append(SaveTrainingData(data_filepath=data_filepath))

# Create model
model = create_model(input_shape, base_learning_rate, momentum)
model.summary()

# Fit the model on the batches generated by datagen
model.fit(train_iter,
          callbacks=callbacks,
          epochs=epochs,
          verbose=verbose,
          initial_epoch=0,
          steps_per_epoch=len(train_iter) // hvd.size(),
          validation_data=test_iter,
          validation_steps=3 * len(test_iter) // hvd.size())

# Score trained model.
scores = model.evaluate(test_iter,
                        steps=len(test_iter),
                        verbose=verbose)

# Save model
model.save(savemodel)

if verbose:
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])