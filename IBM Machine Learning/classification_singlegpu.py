# Import libraries
import os
import numpy as np
from time import time
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Add, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions
img_rows, img_cols = 32, 32

# Reformat to Keras friendly format
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test  = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test  = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)


# Convert classes to categorical values
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

# Convert train / test to float
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# Scale data to 0 - 1
x_train /= 255
x_test  /= 255

# Image Data Generator
# This will be used to perform additional processing on the images that
# will help with training
datagen = image.ImageDataGenerator(
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
            rescale=None,                        # set rescaling factor (applied before any other transformation)
            preprocessing_function=None,         # set function that will be applied on each input
            data_format=None,                    # image data format, either "channels_first" or "channels_last"
            validation_split=0.0)                # fraction of images reserved for validation (strictly between 0 and 1)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# Create model
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
    inputs = Input(shape=input_shape) # 32 x 32 RGB

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
    x = GlobalAveragePooling2D()(x)

    # Output dense layer of 10 classes with softmax activation
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)

    opt = SGD(lr=baselr, momentum=momentum)

    # Compile
    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


# Parameters
batch_size = 128
epochs     = 50

warmup_epochs = 5    # No of epochs for which base learning rate will be used
momentum = 0.9       # Momentum for Stochastic Gradient Descent

base_learning_rate = 0.01 # Learning rate for 1 GPU

verbose = 1 # prints total time

# Create model
model = create_model(input_shape, base_learning_rate, momentum)
model.summary()

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

# Stop accuracy
class StopAtAccuracy(tf.keras.callbacks.Callback):
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


# Run model
callbacks = [StopAtAccuracy(verbose=verbose),LearningRateScheduler(lr_schedule)]

# Append total time
if verbose: callbacks.append(PrintTotalTime())

# Fit the model on the batches generated by datagen
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
        callbacks=callbacks,
        epochs=epochs,
        verbose=verbose,
        # avoid shuffling for reproducible training
        shuffle=False,
        steps_per_epoch=int(len(y_train)/batch_size),
        validation_data=(x_test, y_test),
        workers=4)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=verbose)
if verbose:
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


# Create model
model = create_model(input_shape, base_learning_rate, momentum)
model.summary()

# Run model
callbacks = [LearningRateScheduler(lr_schedule)]

if verbose:
    callbacks.append(PrintTotalTime())

# Fit the model on the batches generated by datagen.flow().
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
        callbacks=callbacks,
        epochs=epochs,
        verbose=verbose,
        # avoid shuffling for reproducible training
        shuffle=False,
        steps_per_epoch=int(len(y_train)/batch_size),
        validation_data=(x_test, y_test),
        workers=4)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=verbose)
if verbose:
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])