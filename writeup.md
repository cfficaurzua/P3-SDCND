#**Behavioral Cloning**
---

[//]: # (Image References)

[image1]: ./examples/raw_distribution_example.PNG "Raw Distribution"
[image2]: ./examples/balanced_data_distribution.PNG "Balanced Distribution"
[image3]: ./examples/activations_nvidia_model.PNG "Activations nvidia"
[image4]: ./examples/Activations_final_model.PNG "Activations final model"

[TOC]

## Project structure

| File                       |Description                  |
|----------------------------|---------------------------- |
| model.py | containing the script to create and train the model|
| drive.py | for driving the car in autonomous mode|
| model.h5 | containing a trained convolution neural network|
| writeup_report.md|  summarizing the results|
|visualization.py| code needed to visualize the activations |
|drive.bat | bash file to run the drive script automatically |

## Introduction
The aim of this project is to provide a model that can drive by itself, maneuvering the steering wheel with only camera inputs, which is known as end to end learning, to obtain the trained model it is necessary to collect a lot of data from the simulator provided by Udacity.
As the quantity of data collected grows, smart memory management is needed, therefore the use of generators is required.

## Goals
The goals  of this project were the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Data collection

In order to collect data, I used the simulator provided by Udacity with a mouse as an input, I realized that there were different quality settings that range from fastest to fantastic, when I inspected the output of the images taken, I notice immediately that they were differences, for example in the fantastic settigs there were shadows and sharp edges, not present in the fastest settings, therefore, with the intention of generalize the model,  I recorded 2 laps back and forth in both tracks with each setting, furthermore I recorded an additional set of laps driving in the left lane as is driven in England or India.

The model had a hard time trying to go through the turn after the bridge, the one that access to the dirt shortcut, in order to overcome this challenge, I added more data of this section (from the bridge onward), I also park the car looking towards the intersection and recorded images with the correct angles plus a noise, without without moving the car.

After driving around the tracks for hours improving the technique I recorder almost 2 Gb of data. and organized it in different folders.

## Balance samples

![alt text][image1]

The data retrieved by the simulator was naturallly unbalanced as shown above, there were many samples with steering angles near zero, and few samples of closed curves near  -1 and 1, so I created a function that selects randomly samples untill no more than a threshold per angle has been completed.

After balancing the data distribution changes as shown below

![alt text][image2]

## Recovering Data

To obtain recovering data I selected randomly image from the left, right or center image, and multiply the steering angle with the corresponing factor, Within the training, I use a magnitud of .25 for the correction factor.

## Data augmentation
In order to generalize even more the model, I randomly tweak the brightness up and down, and also randomly shifted the image vertically

## Splitting

To build a validation an train samples, the train_test_split function from keras was used upon the balanced samples with a ratio of 0.3

## Normalization and cropping.
The input image to the model was first normalized, shrinked to a range between -.5 and .5, then the environment such as trees and sky were cropped out along with the hood of the car, using the embedded function Cropping2d from the keras library.

## Model

After trying with different architectures, including leNet and nVidia end-to-end architecture, I chose the architecture shared by comma.ai, and tweak it a little bit, adding a 1x1 convolution layer at the top, to help the model understand the color space, adding a Dense layer of 10, before the last layer, an changing some parameters as is shown in the following table.

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_2 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_2 (Cropping2D)    (None, 75, 320, 3)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 75, 320, 3)        12        
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 68, 313, 16)       3088      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 34, 156, 16)       0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 30, 152, 32)       12832     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 15, 76, 32)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 13, 74, 48)        13872     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 6, 37, 48)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 10656)             0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 10656)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               5456384   
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11        
=================================================================
Total params: 5,491,329
Trainable params: 5,491,329
Non-trainable params: 0
_________________________________________________________________
## generator
To properly works with great amount of images, a generator was essential. the generator built return the required images that could be augmented, along with the corresponding measurements.

## Parameters

In this project, I encountered that the batch size has an enormous impact in the training result. At first I set it to 32, but the model failed to learn how to drive, but then, I modified the batch size to 128, which did the trick, the model succeed to learn. Without data augmentation, I noticed that a lower batch size would be fine, but as I inserted more random variations of the samples, a larger batch size is necessary.

To compile the model an Adam optimizer was used, with a learning rate of 1e-4, but then was change to 1e-5, the loss function chosen was the mean absolute value, as was well spread in the slack channel to have a much better result.

## L2 Regularization

At first using the n-vidia model the car drove pretty well in the second track, but run dreadfully in the first track, in order to examine what was going on, I decided to take a look inside the activations on the second-last and third-last layers.
To perform the analysis I insert as an input a collection of images with more or less the same steering angle, going from left to right, and then taking the median per neuron of all activations. Looking at the result I immediately notice, surprisingly, that only 2 of the 10 neurons were relevant for the second-last layer, and no more tahn 4 for the 50 neurons in the third-last layer.

In the following image is shown the activations for every angle estimulation for the nvidia model

![alt text][image3]

My first thought to fix this issue was to reduce the number of neurons from 10 to 2, and from 50 to 4 respectively, but this will make the model less robust and dependent of only a few neurons.

My second attempt was to teach the network right from the second-last layer, to achieve this, I scattered the steering angle with a guassian distribution, that returned 10 values. then shuffle this values to removed any sort of bias and save the order in which the values were shuffled to later unshuffled them.

Afterwards I trained the model with mean square error, comparing directly this values with the second-last layer. When the model is trained in the drive script I unshuffled the values and retrieve the steering angle.

Unfortunately this plan failed to succeed, the car drove only straight ahead without turns and brake every a couple of seconds, probably because retrieving the steering angles was a bottleneck in the system.

Lastly my third and final effort was to implement L2 Regularization, this overcame the problem of a few meanful neurons per layer and also provided a solution to overfitting.

With L2 regularization a more consistent activations were obtained as shown below

![alt text][image4]


## Final Approach

To finally obtain a valid self driving car I had to use all the data I recorded, and do an iterative process than consists in balance the data with a cutoff of 500 samples per angle range. Then split the data to get train and validation samples, from that build a generator compile and train 10 epochs. this process is repeated 3 times.

## Results
The results of the autnomous driving on the first track can be seen in the following [link](https://youtu.be/BuKoBeq9uAQ). the car completed the track successfully but wobbles a lot. The model is not yet ready for the hard turns of the second track.

## Conclusion

This project was really difficult, I first underestimated it, thought that wouldn't take that much time to completed, but overpassing the closed curves and dirt zones was really tough, and most of times the model get stuck and refuse to learn, or drove completely straight, other times it swerve randomly before encountering with the bridge, but the most annoying thing was that the model sometimes unlearned the things that has already master.

Troughout the project I realized that there were many random factors that made the outcome totally unpredictable, but at the end the car managed to complete the first track, meanwhile the second track remains pending, and I will go on improving the model until it could drive autonomously and flawlessly in both tracks.
