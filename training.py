import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Convolution2D(6,5,5, activation = 'relu'))
model.add(MaxPooling2D(pool_size=2,2))
model.add(Convolution2D(6,5,5, activation = 'relu'))
model.add(MaxPooling2D(pool_size=2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss= 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split= 0.3, shuffle= True, epochs= 4)
model.save('model.h5')
