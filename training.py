import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, \
    Dense, Activation, BatchNormalization
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# setting constants
FAST_RUN = False
WIDTH=64
HEIGHT=64
SIZE=(WIDTH, HEIGHT)
IMAGE_CHANNELS=3
epochs = 2
batch_size = 12
train_num = 196
validate_num = 58

train_data_dir = './data/train'
validation_data_dir = './data/test'

# set image shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, WIDTH, HEIGHT)
else:
    input_shape = (WIDTH, HEIGHT, 3)

# setting hyperparameters
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization()) # Normalize data to increase efficiency
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # reduce size to one fourth
model.add(Dropout(0.25)) # increase generalizability

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)) # filter increases
model.add(BatchNormalization()) # Normalize data to increase efficiency
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # reduce size to one fourth
model.add(Dropout(0.25)) # increase generalizability


# dense layer
model.add(Flatten())
model.add(Dense(64, activation='relu')) # 512 neurons
model.add(BatchNormalization())
model.add(Dropout(0.5)) # generalizability
model.add(Dense(2, activation='softmax')) # 2 since binary classificaiton

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# training stuff
earlystop = EarlyStopping(patience=15) # prevent overfitting
rate_reduce = ReduceLROnPlateau(minotor='val_acc',
                                patience=2,
                                verbose=1,
                                factor=0.5,
                                min_lr=0.00001)
callbacks = [earlystop, rate_reduce]

# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1./ 255
)

# generate training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(WIDTH, HEIGHT),
    batch_size=batch_size,
    class_mode='binary'
)

# generate validation data
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(WIDTH, HEIGHT),
    batch_size=batch_size,
    class_mode='binary')

# fit
model.fit_generator(
    train_generator,
    steps_per_epoch=train_num // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validate_num // batch_size
)

# last step
model.save('distraction_model.hdf5')