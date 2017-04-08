import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

def preprocessing(data):
    output = (data/255.0 -0.5)
    return output

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
model.add(Lambda(lambda x: preprocessing(x), input_shape = (160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss= 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split= 0.3, shuffle= True, epochs= 7)
model.save('model.h5')
