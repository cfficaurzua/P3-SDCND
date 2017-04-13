import csv
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
import sklearn
from sklearn.model_selection import train_test_split

%matplotlib inline

def balance_data(samples, bins = 101, width = (-1,1), cutoff=500):
    output = []
    shuffle(samples)
    raw_measurements = (np.array(samples)[:,3]).astype(float)
    bins = np.linspace(width[0],width[1],bins)
    bin_values = np.zeros(100)
    for i in range(len(raw_measurements)):
        measurement = raw_measurements[i]
        for j in range(1,len(bins)):
            if (( bins[j-1] <= measurement )  and  ( measurement <= bins[j] ) ) :
                if bin_values[[j-1]] < cutoff:
                    bin_values[j-1] +=1
                    bin_values[len(bin_values)-j] +=1
                    output.append(samples[i])
                break
    return output

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('\\')[-1]
                current_path = './data/IMG/' + filename
                image = cv2.imread(current_path)
                flipped_image = cv2.flip(image,1)
                measurement = float(batch_sample[3])
                flipped_measurement = -1.0 * measurement
                images.append(image)
                measurements.append(measurement)
                images.append(flipped_image)
                measurements.append(flipped_measurement)
                X_train = np.array(images)
                y_train = np.array(measurements)
                yield sklearn.utils.shuffle(X_train, y_train)

raw_samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        raw_samples.append(line)
print(len(raw_samples))
samples_distribution = np.array(raw_samples)[:,3].astype(float)
hist, bin_edges = np.histogram(samples_distribution, bins = 100, range = (-1,1))
plt.bar(bin_edges[:-1], hist,width=0.01)
balanced_samples = balance_data(raw_samples)
test = np.array(balanced_samples)[:,3].astype(float)
test = np.append(test,-test)
hist, bin_edges = np.histogram(test, bins = 100, range = (-1,1))
plt.bar(bin_edges[:-1], hist,width=0.01)




train_samples, validation_samples = train_test_split(balanced_samples, test_size=0.3)
input_shape = (160, 320, 3)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = input_shape))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(6,(5,5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6,(5,5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100,kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(50,kernel_regularizer=l2(0.01)))
model.add(Dense(10,kernel_regularizer=l2(0.01)))
model.add(Dense(1))
model.compile(loss= 'mae', optimizer = 'adam')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples), validation_data = validation_generator, validation_steps = len(validation_samples), epochs = 3, verbose = 2)
model.save('model.h5')
