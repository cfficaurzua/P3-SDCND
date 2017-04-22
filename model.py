## import necessary libraries
import csv
import cv2
import os
import random
import numpy as np
import pickle
from random import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.initializers import TruncatedNormal
from keras.optimizers import Adam
from keras.models import load_model
from PIL import ImageEnhance
from PIL import Image

import sklearn
from sklearn.model_selection import train_test_split

%matplotlib inline

def balance_data(samples, n_bins = 101, width = (-1,1), cutoff=500):
    """This functions receives as an input a set of samples in csv format,
    Then randomly selects data until per angle range there is no more than
    samples than the threshold designated.
    """
    output = []
    shuffle(samples)
    raw_measurements = (np.array(samples)[:,3]).astype(float)
    bins = np.linspace(width[0],width[1],n_bins)
    bin_values = np.zeros(n_bins-1)
    for i in range(len(raw_measurements)):
        measurement = raw_measurements[i]
        for j in range(1,len(bins)):
            if (( bins[j-1] <= measurement )  and  ( measurement <= bins[j] ) ) :
                if type(cutoff) == int:
                    if bin_values[j-1] < cutoff:
                        bin_values[j-1] += 1
                        output.append(samples[i])
                else:
                    assert (len(cutoff) == (n_bins-1))
                    if bin_values[j-1] < cutoff[j-1]:
                        bin_values[j-1] += 1
                        output.append(samples[i])
                break
    return output

def generator(samples, batch_size = 32,  factor = 0.25, augmentation = True):
    """generator choose randomly between the center, left and right images
    with the associated measurement multiply by its correction factor,
    also applies augmentation if desired, and flip half of the returned images.
    """
    num_samples = len(samples)
    factors = [0, factor, -factor]
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                i = np.random.choice([0,1,2], p = [0.5, 0.25, 0.25])
                filename = batch_sample[i].split('\\')[-3:]
                filename.insert(0,os.getcwd())
                image = cv2.imread(os.path.join(*filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                measurement = float(batch_sample[3]) + factors[i]
                if augmentation:
                    if random.random() > 0.5:
                        M = np.float32([[1,0,0],[0,1,random.randint(-25,25)]])
                        image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]), borderMode = cv2.BORDER_REFLECT)
                    if random.random() > 0.5:
                        image[:,:,2] = random.choice([(image[:,:,2]* random.uniform(0.4,0.8)).astype('uint8'), (255 -(255-image[:,:,2])*random.uniform(0.6,0.8)).astype('uint8')])
                images.append(image)
                measurements.append(measurement)
                X = np.array(images)
                y = np.array(measurements)
                ind = random.sample(range(len(X)), int(len(X) / 2))
                X[ind] = X[ind, :, ::-1, :]
                y[ind] = -1.0 * y[ind]
                yield sklearn.utils.shuffle(X, y)


def load_data(input_paths):
    """load csv files from a list of input directories"""
    outdata = []
    for input_path in input_paths:
        with open(os.path.join(os.getcwd(),input_path, 'driving_log.csv')) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    if len(line)==7:
                        outdata.append(line)
    return outdata


### input directories
input_paths = ['Track_1_fantastic_2_lap_back_and_forth', 'Track_1_fastest_2_lap_back_and_forth', 'Track_1_simple_2_lap_back_and_forth','Track_1_bridge', 'Track_1_dirt_curve','one_more_lap','Track_1_dirt_curve', 'Track_1_bridge', 'Track_2_fastest_2_lap_back_and_forth', 'Track_2_fastest_2_lap_back_and_forth_english','Track_2_simple_2_lap_back_and_forth',]

###load the data

raw_samples = load_data(input_paths)


### plot the data distribution
samples_distribution = np.array(raw_samples)[:,3].astype(float)
hist, bin_edges = np.histogram(samples_distribution, bins = 100, range = (-1,1))
plt.bar(bin_edges[:-1], hist,width=0.01)


### balace the data and plotted
balanced_samples = balance_data(raw_samples, cutoff =500)
len(balanced_samples)
test = np.array(balanced_samples)[:,3].astype(float)
hist, bin_edges = np.histogram(test, bins = 100, range = (-1,1))
plt.bar(bin_edges[:-1], hist,width=0.01)

### important parameters
input_shape = (160, 320, 3)
batch_size = 128


### split the dataset accordingly
train_samples, validation_samples = train_test_split(balanced_samples, test_size= 0.3)

### generators that will be used
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size, augmentation = False)


### Model

initializer = TruncatedNormal(mean = 0.0, stddev = 1e-4)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = input_shape))
model.add(Cropping2D(cropping=((60,25),(0,0))))
model.add(Conv2D(3,(1,1), activation = 'elu',kernel_regularizer=l2(1e-4)))
model.add(Conv2D(16,(8,8), activation = 'elu',kernel_regularizer=l2(1e-4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32,(5,5), activation = 'elu',kernel_regularizer=l2(1e-4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48,(3,3), activation = 'elu',kernel_regularizer=l2(1e-4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'elu',kernel_regularizer=l2(1e-4), kernel_initializer = initializer))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'elu',kernel_regularizer=l2(1e-4), kernel_initializer = initializer))
model.add(Dense(1))
#model.summary()


### Compiling training, and saving the model

for i in range(3):
    balanced_samples = balance_data(raw_samples, cutoff = 500)
    train_samples, validation_samples = train_test_split(balanced_samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size = batch_size)
    validation_generator = generator(validation_samples, batch_size = batch_size, augmentation = False)
    model_path = './model_dday_nnn_500_n_' +str(i) + '.h5'
    model.compile(loss= 'mae', optimizer = Adam(lr=1e-6))
    model.fit_generator(train_generator,
                        steps_per_epoch = len(train_samples)/batch_size,
                        validation_data = validation_generator,
                        validation_steps = len(validation_samples)/batch_size,
                        epochs = 10,
                        verbose = 2)
    print(i)
    model.save(model_path)
