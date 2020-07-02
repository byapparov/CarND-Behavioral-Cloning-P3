# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_model]: ./writeup_images/model_architecture.png "Model Visualization"
[image_crop]: ./writeup_images/cropped_image.jpg "Cropping (75, 25)"
[image_flip]: ./writeup_images/flipped_image.jpg "Recovery Image"
[image_original]: ./writeup_images/car_sharp_angle.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
 The file shows the pipeline I used for training and validating the model, 
 and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), 
and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 120 & 122). 

There are also stride (2x2) parameters in the convolution layers that reduce the number of the
model parameters.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 57 & 98-99). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Observations where steering was at zero angle were sampled (line 35) to reduce the bias towards zero. 

Left and right cameras were used with correction which would eliminate the bias towards zeo in the 
observations further. Correction parameter was selected through the tuning described below.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 130).

#### 4. Appropriate training data

I have done following recordings of the simulation to collect the data:
- driving on the first track in both directions
- driving on the second track in one direction
- driving on the first track getting car on the road multiple times
in different locations on the track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I have used [model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) used by NVIDIA team
as the starting point for my implementation. 

I have added two dropout layers between the dense layers of the model to reduce overfitting. 

To reduce the number of parameters in the model, I used (2x2) strides in the first two convolution layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Selected model architecture consists of three convolution (5x5) layers,
followed by two convolution (3x3) layers and and four dense layers. 

Total number of model parameters is **1,902,531**.

Here is a visualisation of the final model:

![model architecture][image_model]

Summary of the Keras model is:

Layer (type)       |  Output Shape, height| width| layers   |   # Params   
-------------------|:-----------:|:----------:|:-----------:|:-------------------:
Cropping2D | 60 | 320| 3 |   0         
Lambda    |  60| 320|3  |  0         
Conv2D    |  28| 158| 24 |  1824      
Conv2D    |  12| 77| 36  |  21636     
Conv2D    |  8| 73| 48   |  43248     
Conv2D    |  6| 71| 64   |  27712     
Conv2D    |  4| 69| 64   |  36928     
Flatten   |  17664 | | |  0         
Dense     | 100  | | | 1766500   
Dropout   | 100 | | | 0         
Dense     | 42  | | |  4242      
Dropout   | 42  | | |  0         
Dense     | 10  | | |  430       
Dense     |  1  | | |  11 
       
#### 3. Creation of the Training Set & Training Process

I have done following recordings of the simulation to collect the data:
- driving on the first track in both directions
- driving on the second track in one direction
- driving on the first track getting car on the road multiple times
in different locations on the track. 


To augment the data and increase the number of data points, I have flipped the images 
within the generator:

Original Image|Horisontal Flip
------------|----------
![alt text][image_original]|![alt text][image_flip]


In total I have collected 14622 data points (43866 images from 3 cameras).

After re-sampling to reduce number of observations where control was zero, ~9876 lines were used.

Data was randomly shuffled with 80/20 split into training and validation sets. 


I have used Adam optimiser and manually selected 8 epochs for training as loss was growing afterwards on the validation set.

## Model parameters selection

### Number of Epochs

Loss values progression for the model with batch Size 50:

Epoch | Loss | Val Loss
:---:|:------:|:-------:
1| 0.0387 | 0.0308
2|0.0295 |0.0271
3|0.0263|0.0238
4|0.0239 | 0.0221
5|0.0223 | 0.0218
6|0.0197 | 0.0196
7|0.0185 | 0.0177
8|0.0177 | 0.0176
9|0.0158 | 0.0156
10|0.0150 | 0.0147 *
11|0.0135 | 0.0153
12|0.0131 | 0.0136
13|0.0119 | 0.0137
14|0.0113 | 0.0129
15|0.0109 | 0.0130

## Camera correction selection 

Having images from three cameras allowed me to select driving angle correction in a way that would 
enforce correct position of the car in the road. 

To select correct adjustment I have ran the model for several values and compared value loss keeping other 
parameters of the training the same.

Batch size: 10
Epochs: 10
Number of observations (first track only): 3434

Correction|Loss|Val Loss|Car makes the full circle|Training post 1st Epoch|Comment
:----:|:------:|:---:|:-----:|:-----:|:----
0.15| 0.0150 |  0.0150|No|Yes|Unable to make turns
0.2| 0.0117 |  0.0120|No|Yes|Drives off in to the off-road track.
0.25| 0.0163|  0.0186|No|Yes|Driving of towards the end of the track. 
0.3| 0.0217 | 0.0234|No|Yes|Drives off at the end of the first turn

Based on the analysis above, I have chosen the model with 0.25 angle adjustment
for images from the left and right cameras. 

# Model Tuning Summary

Parameter | Value
-------:|:---------:
Keep Zero Angle Observations | 20%
Wheel angle adjustment| 0.25
Batch size | 10 
Epochs | 8
Observations | 9876
Validation Loss | 0.0186
