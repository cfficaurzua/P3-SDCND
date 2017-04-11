import csv
import cv2
import random
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D
from keras.layers.convolutional import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
lines = []
output_size = 100

def gaussmf(x,mean,sigma):
    return np.exp(-((x-mean)**2.)/float(sigma)**2.)

def distribute(x, sigma = 0.1, n = 10, width = (-0.75,0.75)):
    output = []
    means = np.linspace(width[0],width[1], n)
    for i in range(n):
        output.append(gaussmf(x,means[i],sigma)-0.5)
    return np.array(output)

def generate_order(l):
    order = list(range(l))
    random.shuffle(order)
    return order


with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
order = generate_order(output_size)
pickle.dump(order, open('order.p','wb'))
for line in lines:
    measurement = distribute(float(line[3]), n = output_size)[order]
    flipped_measurement = distribute(-float(line[3]), n = output_size)[order]
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(measurement)
    # flipped images
    flipped_image = cv2.flip(image,1)
    images.append(flipped_image)
    measurements.append(flipped_measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print(y_train.shape)




model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5), strides = (2,2),  activation = 'relu'))
model.add(Conv2D(36,(5,5), strides = (2,2),  activation = 'relu'))
model.add(Conv2D(48,(5,5), strides = (2,2),  activation = 'relu'))
model.add(Conv2D(64,(3,3),  activation = 'relu'))
model.add(Conv2D(64,(3,3),  activation = 'relu'))
model.add(Flatten())
model.add(Dense(512,kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(256,kernel_regularizer=l2(0.01)))
model.add(Dense(output_size,kernel_regularizer=l2(0.01)))

model.compile(loss= 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.3, shuffle = True, epochs = 4, verbose = 2)
model.save('model.h5')
