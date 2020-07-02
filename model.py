from typing import Any, Generator

import numpy as np
import cv2
import csv
import os
import math
from random import choice

# Note: Simulator App is in the file called beta_simulator_mac

lines =[]

print(os.getcwd())
from  scipy.stats import bernoulli
train_data_dir = os.getcwd() + "/train_data/"

def keep_zero_angle(p = 0.2):
    # return choices([True, False], [p, 1 - p])
    # returns True or False based on the
    # probability parameter - p
    c = choice(range(10))
    if c < p * 10:
        return True
    else:
        return False

with open(train_data_dir + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_angle = float(line[3])

        # Only use 20% of observations where wheel is at
        # zero angle
        if steering_angle != 0 or keep_zero_angle(.2):
            lines.append(line)

print("Number of lines {lines}".format(lines = len(lines)))

def correct_driving_measrument(measurement, camera):
    # caluclates adjustment for the wheel angle measurment
    # based on the side on which camera is installed
    side_camera_correction = 0.3
    if camera == 0: # center camera
        res = measurement
    elif camera == 1: # left camera
        res = measurement + side_camera_correction
    elif camera == 2: # right camera
        res = measurement - side_camera_correction

    return res

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Split data into training and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Generator function returns images and measurment values
def driving_batch_generator(samples, batch_size = 20):
    num_observations = len(samples)
    while True:
        samples = shuffle(samples)
        for batch_offset in range(0, num_observations, batch_size):
            batch = samples[batch_offset:batch_offset + batch_size]

            images = []
            measurements = []

            for observation in batch:
                for i in range(3):
                    source_path = observation[i]
                    file_name = source_path.split("/")[-1]
                    image_path = train_data_dir + "IMG/" + file_name
                    # print("Reading image: {image_path}".format(image_path = image_path))
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # this is required because drive.py uses RGB
                    if image is None or not image.shape == (160, 320, 3):
                        print(image_path)
                        break

                    images.append(image)
                    measurement = float(line[3])  # Steering Angle
                    measurement = correct_driving_measrument(measurement, i)
                    measurements.append(measurement)

                    image_flip = cv2.flip(image, 1)
                    measurement_flip = -measurement
                    images.append(image_flip)
                    measurements.append(measurement_flip)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)

# Size of the batch for the image generation
batch_size = 10

# Data generators to lazy-load images from the drive
train_generator = driving_batch_generator(train_samples, batch_size=batch_size)
validation_generator = driving_batch_generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout

model = Sequential()
# Here we crop the top and bottom of image, keeping the road
model.add(Cropping2D(((75, 25), (0, 0)), input_shape = (160, 320, 3)))

# normalization of the image
model.add(Lambda(lambda x: x / 255 - 0.5))

model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation="relu"))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation="relu"))
model.add(Conv2D(filters=48, kernel_size=5, strides=(1, 1), activation="relu"))


model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(42))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")

# Fit the model to the training data
history_object = model.fit_generator(
    generator = train_generator,
    steps_per_epoch = len(train_samples) // batch_size,
    validation_data=validation_generator,
    validation_steps= len(validation_samples) // batch_size,
    verbose=1,
    epochs=8
)

model.save('model.h5')
