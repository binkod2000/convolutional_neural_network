#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network CNN
Created on Mon Apr  8 14:10:46 2019

@author: Liam Sullivan
"""
# import libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initalize the CNN
# Create the classifer object
classifier = Sequential()

# Step One: Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))# theano back end input_shape(3, 64, 64)

# Step Two: Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add Second Layer
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add thrid layer
classifier.add(Conv2D(64, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step Three: Flatten
classifier.add(Flatten())

# Step Four: Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# classifier.add(Dense(output_dim = 2, activation = 'softmax'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit to images
# rescale/augment training_set folder photos.
train_image_data_generator = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2,
                                   horizontal_flip = True)
# Rescale test set
test_image_data_generator = ImageDataGenerator(rescale = 1./255)

# Connect training_set to image gererator
training_set = train_image_data_generator.flow_from_directory('dataset/training_set', target_size = (64, 64), batch_size = 32,
                                                    class_mode = 'binary')

# Connect test set to image generator
test_set = test_image_data_generator.flow_from_directory('dataset/test_set',target_size = (64, 64), batch_size = 32,
                                                        class_mode = 'binary')
# TRAIN THE MODEL!!!
classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 25, validation_data = test_set,
                    validation_steps = 2000)
