from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import backend as K
from random import shuffle
import os
import cv2
import csv
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from random import shuffle
%matplotlib inline
from keras.models import Model

def balance_data(samples, n_bins = 101, width = (-1,1), cutoff=500):
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

model_path = 'model_nvidia.h5'
model = load_model(os.path.join(os.getcwd(),model_path))


model.get_config()
model.summary()
intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer('dense_7').output)
n = 10
def load_data(input_paths):
    outdata = []
    for input_path in input_paths:
        with open(os.path.join(os.getcwd(),input_path, 'driving_log.csv')) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    if len(line)==7:
                        outdata.append(line)
    return outdata
input_paths = ['Track_1_fantastic_2_lap_back_and_forth', 'Track_1_fastest_2_lap_back_and_forth', 'Track_1_simple_2_lap_back_and_forth','Track_1_bridge', 'Track_1_dirt_curve','one_more_lap','Track_1_dirt_curve', 'Track_1_bridge', 'Track_2_fastest_2_lap_back_and_forth', 'Track_2_fastest_2_lap_back_and_forth_english','Track_2_simple_2_lap_back_and_forth',]
samples = load_data(input_paths)

samples_by_angles = {}
for i in range(n):
    cutoff_array  = np.zeros(n)
    cutoff_array[i] = 500
    samples_by_angles[i] = balance_data(samples, n_bins= n+1, cutoff = cutoff_array)

for i in range(n):
    y =  np.array(samples_by_angles[i])[:,3].astype(float)
    hist, bin_edges = np.histogram(y, bins = n, range = (-1,1))
    plt.bar(bin_edges[:-1],hist, width = 0.1)

output_by_angles = {}
for i in range(n):
    output = []
    for line in samples_by_angles[i]:
        source_path = line[0]
        filename = source_path.split('\\')[-3:]
        filename.insert(0,os.getcwd())
        image_array = cv2.imread(os.path.join(*filename))
        intermediate_output =  intermediate_layer_model.predict([image_array[None, :, :, :]])[0]
        output.append(intermediate_output)
    output_by_angles[i] = output

np.mean(output_by_angles[5],axis = 0)
for i in range(n):
    plt.figure()
    plt.bar(np.linspace(-1,1,n), np.mean(output_by_angles[i],axis = 0) , width = 0.1)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(elev=40, azim=110)
for i in range(n):
    x = np.linspace(0,1,n)
    z = np.linspace(-1,1,n)
    y = np.mean(output_by_angles[i],axis = 0)
    variance = np.var(output_by_angles[i])
    ax.bar(x, y, zs=z[i], zdir = 'y', alpha=1, width = 0.1)

ax.set_xlabel('Neurons')
ax.set_ylabel('Angles')
ax.set_zlabel('Activation')

plt.show()
