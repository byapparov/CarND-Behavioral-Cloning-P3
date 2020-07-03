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
[image_flip]: ./writeup_images/flipped_image.jpg "Flipped Image"
[image_original]: ./writeup_images/car_sharp_angle.jpg   "Original Image"
[image_sharp_angle]: ./writeup_images/car_sharp_angle.png "Sharp Angle Screenshot"

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

As a model from NVIDIA's paper produced some results from the start, 
I have focused on improving the data collection process with the following features:

- Sampling of images in way that proportion of the observations with zero wheel angle is reduced (reduces bias for zero).
- Using all three cameras with correction for side cameras that allows the cart to make sharper turns.
- Cropping of the images was added to the model to leave the relevant part. 
- Flipped images were added to the data set to increase the sample size and also remove the direction bias.


#### 2. Final Model Architecture

First layer of the sequential model crops the top and bottom of the image.

Original Image|Cropped Image
------------|--------------
![Original Image][image_original]|![Cropped image][image_crop]

This removes area above the road that is not relevant for driving and
area that contains the boot of the car.

Following layers of selected model architecture consist of:

  - three convolution (5x5) layers,
  - two convolution (3x3) layers 
  - four dense layers. 

To avoid overfitting, first two layers have strides (2,2) 
and dense layers are linked through dropout layers.

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

Here a screenshot of an example where car is in a position where it is
entering or crossing the road:

 ![Sharp Angle][image_sharp_angle]
 
This position is very rear while driving but important for the car
to learn to steer sharp turns. 

To augment the data and increase the number of data points, I have flipped the images 
within the generator function:

Original Image|Horisontal Flip
------------|----------
![alt text][image_original]|![alt text][image_flip]

Each observation for a flipped image label got assigned the negative of the original measurement.

In total I have collected 14622 data points (43866 images from 3 cameras). 

After re-sampling to reduce number of observations where wheel angle was zero, 6992 lines were used.
Due to the random nature of re-sampling procedure this number will be different for each execution of `model.py`.

Data was randomly shuffled with 80/20 split into training and validation sets. 

*Camera correction selection*

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

Based on the analysis above, I have trained the model values over 0.2 angle adjustment
for images from the left and right cameras and 0.3 was used in the final version of the model. 

Higher correction value allows the car to apply sharper angles in the turns, which means it is likely to 
drive on the side of the road. 


# Appendix

# Model Tuning Summary


Parameter | Value
-------:|:---------:
Keep Zero Angle Observations | 20%
Wheel angle adjustment| 0.3
Batch size | 10 
Epochs | 8
Observations | 9876
Validation Loss | 0.0186

## Final model fit history

I have used Adam optimiser and manually selected 8 epochs for training as loss was growing afterwards on the validation set.

Number of observations: 6992 * 2 (flip) * 3 (cameras)

Epoch | Loss | Validation Loss
:-------:|:-----------:|:---------:
1 | 0.0475 | 0.0406
2 | 0.0417 | 0.0391
3 | 0.0383 | 0.0358
4 | 0.0362 | 0.0332
5 | 0.0338 | 0.0314
6 | 0.0314 | 0.0290
7 | 0.0297 | 0.0284
8 | 0.0274 | 0.0283
