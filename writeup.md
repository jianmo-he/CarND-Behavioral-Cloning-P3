# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_cnn_image]: ./writeup_images/nvidia_cnn_architecture.png "Nvidia Model Visualization"
[center]: ./writeup_images/center.jpg "Center Driving"
[left]: ./writeup_images/recovery_left.png "Left Recovery Image"
[right]: ./writeup_images/recovery_right.png "Right Recovery Image"
[normal]: ./writeup_images/normal.jpg "Normal Image"
[flipped]: ./writeup_images/center_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based of the Nvidia self-driving car model and consists of a convolution neural network with both 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py starting from line 90) 

The model includes RELU layers to introduce nonlinearity, the data is normalized in the model using a Keras lambda layer (code line 77), and is cropped using Keras cropping layer, removing the top 70 pixels and the bottom 25 pixels (code line 78). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 104). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 103).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. And I drove the car as best as I can. Especially during the turns, I tried to drive it as smooth as I can. And I also reduced the speed during the turns. Because it the speed is set in autonomous mode.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a proven model. In this project I used the Nvidia self-driving car model mentioned in the course videos.

Firstly I pre-processed the image data with normalization, corpping out not usefull parts for images.

Then I used a series of cnn models as the Nvidia model architecture. And I connected flattened the data to pass it through 4 dense layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. about 20% of the training set is used for validation.

To combat the overfitting, I only ran 3 epochs. More then 3 epochs displayed an oscillating behavior of the rms error in the validation set, as well as an eventual increase in rms error in the training model. 3 epochs seems to work well.

The final step was to run the simulator to see how well the car was driving around track one. There was one spot that the car almost run out of the road. But overall it works fine.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

* Normalize, between -0.5 and 0.5
* Cropping, 70 pixels off the top and 25 pixels off the bottom
* Apply a 5x5 convolution with 24 output filters, 2x2 subsample, RELU activation
* Apply a 5x5 convolution with 36 output filters, 2x2 subsample, RELU activation
* Apply a 5x5 convolution with 48 output filters, 2x2 subsample, RELU activation
* Apply a 3x3 convolution with 64 output filters, RELU activation
* Apply a 3x3 convolution with 64 output filters, RELU activation
* Flatten
* Dense, output size 100
* Dense, output size 50
* Dense, output size 10
* Dense, output size 1

Here is a visualization of the architecture:

![alt text][nvidia_cnn_image]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 1 lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back towards the center when it finds itself off-course. I recorded two examples on each side. These images show what a recovery looks like starting from the left and the right :

![alt text][left]
![alt text][right]

To augment the data sat, I also flipped images and angles thinking that this would double the amoung of data to train on, as they generate the same scenario but turning the opposite way. This will provide a perfect balance of left and right turning training. For example, here is an image that has then been flipped:

![alt text][normal]
![alt text][flipped]

After the collection process, I had 34,698 number of data points. I then preprocessed this data by normalizing the image and cropping the top and bottom, as described earlier.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the training and evaluation rms error, as described above. I used an adam optimizer so that manually training the learning rate wasn't necessary.