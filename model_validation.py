import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


from keras.models import load_model
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

  
input_shape = (160, 320, 3)
measurements = []
predictions = []
model_path = './model.h5'
model = load_model(model_path)
for line in samples:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './data/IMG/' + filename
    image_array = cv2.imread(current_path)
    #    image_array[:,:,i] = (current_image[:,:,0]+current_image[:,:,1]+current_image[:,:,2])*1/3
    prediction = model.predict(image_array[None, :, :, :], batch_size=1)
    steering_angle = float(prediction)
    #predictions.append(steering_angle)
    #measurements.append(float(line[3]))
    print(steering_angle, float(line[3]))
