# **Behavioral Cloning** 

## Writeup Report

Link for the screen recording of the car driving autonomously :
[![Watch the video](http://img.youtube.com/vi/rWRq7b76zkM/0.jpg)](https://www.youtube.com/watch?v=rWRq7b76zkM)
https://youtu.be/rWRq7b76zkM

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model_vis.png "Model Visualization"
[image2]: ./images/center_930.jpg "Sample of the Track1 Center Camera"
[image3]: ./images/flipped.png "Driving opposite way, Image Flipped"
[image4]: ./images/shadow.png " Random shadow on Image"
[image5]: ./images/warped.png "warped Image"
[image6]: ./images/left_854.jpg "left camera Image"
[image7]: ./images/_948.jpg "right camera Image"
[image8]: ./images/brightness.png "brightness Image"
[image9]: ./images/brightness2.png "brightness Image"
[image10]: ./images/brightness3.png "brightness Image"
[image11]: ./output/Epoch-6.png "Training and Validation loss"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model2.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing model2.h5.

```sh
python drive.py model2.h5
```
The video of the drive is availible as run1.mp4 and the screen recording is :
https://www.youtube.com/watch?v=rWRq7b76zkM

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. As suggested in the project description, a generator is used to shuffle the samples and generate a batch to be processed before feeding it to the Network. For every batch the generator performs image augmentation after shuffling the samples. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The data is normalized in the model using a Keras lambda layer (code line 104) after cropping the image from top and buttom (code line 103). 
The model consists of a 3 consequitive convolution neural network with 5x5 filter sizes and depths 24, 36 and 48 (model.py lines 106-116), followed by 2 consequitive convolution neural network with 3x3 filter sizes and both depths 64. Model includes Relu activation after each convolution neural network. The last stages are 4 fully connected layers of 100, 50, 10, 1. A single neuron at the output is to predict the steering angle. After each fully connected layer before activation layer, dropout is used to ensure that model doesnt overfit and memorize the patern instead of generalizing it. All the activation layers are Relu introducing nonlinearity.

The final model also included 2 layes of maxpool, which improved the final results.

The final model visualization is presented by the graph: (plot_model, line 145)

![alt text][image6]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 115, 126, 129, 132, 135). 
The model was trained and validated on different data sets to ensure that the model was not overfitting, 20% of the dataset is partitioned for validation (code line 51). The model was tested by running it through the simulator autonomously (using drive.py script: "python drive.py model2.h5") and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). A batch_size of 32 is selected, however the generator considers 2 other images from right and left camera in addition to the center image.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving recovering from the left and right sides of the road whill the car is directed off the centre. Given my bad simulator driving skills, the network trained well enough that the car can autonomously drive without getting off the road on track 1. (note, there were not any data taken from track 2.) However, with this training, the car doesn't get that far on the more complicated track, possibly because there are not enough data for the model to be trained for the level of complexity of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have enough layers that the model can extract larger number of features.

My first step was to use a convolution neural network model similar to the one that was used in the keras notes. The next model was implemented (without the maxpool(M_no_MaxPool.h5)) after looking into the paper introduced\recomended in the project notes, "End to End Learning for Self-Driving Cars" by NVIDIA. Then added two Maxpool (line 114, and 122) to see if the model performs better. To my surprise with adding the maxpool the results became much better. I was under impression that if we have layers of maxpool it can impact the center driving negatively, but it actually didn't.

The data that was inputed to the model is split of the images and steering angle data, into a training and validation set. The images are used as the input to the convolutional layers and stearing angels as the Y_data, to be predicted by the model. On the very first model both validation and and training mean square error converged, thus adding more data wouldn't help the model and the error for both training and validation seemed to be low. However, once tried the model on the simulator autonomously, after sometime the car got out of the road and into the greens!! So it appeared that the model didn't generalize well enough. Have to mention that the data collected at that time was only for one lap of driving. 
The second attempt was with the model presented in this report, given data for one lap of driving, the mean square error for validation and training were not much different but not convereged so to address this issue collected another lap of driving in additional to add the images from both left and right camera to the data set. To compensate for the steer angel, lets say the camera is off by x meters and after dz distance is desired to get back to center; based on this we can estimate the correction angel and convert it to degrees. This helps to train for the conditions other than steering angel zero, since on track 1 there are higher number of training data collected that has a zero steering angel. 

The images below shows the right and left camera views that used in the training data:

![alt text][image6]
![alt text][image7]
 

To further improve the driving behavior when there are shadows a function that randomly generate polygon on the image (Img_process.py, called "rand_shadow2(img)") is used.
The main idea is to generate any random region and create a shading for only those pixels. For that reason generated a random two set of points on the image and produced a line using polyfit, using two parallel lines, found the regions between the lines to change the shading of those pixels by modifying channel H of HLS.  

![alt text][image4]

Also there has been some online recomendations to use augmentation of brightness, to ensure the change of the lighting wont impact the results, but didn't seem to change the result for this particular track. However, on the second track there is a lot of darker and brighter areas that this feature would help.

The images below show some of the examples of random brightness changes: 

![alt text][image8]
![alt text][image9]
![alt text][image10]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. 
One interesting observation is that, when collected data from car driving on track 2, even for half a track, the model could generalize so much better on track 1 with even a smaller dataset, although that model (M.h5) never seen track 1. The same model eventhough saw more for the track 2 could only drive so far on the track 2 before running off the road, which I beleive could be improved by collecting more datasets for the second track or augment further the track one data so it could generalize for the complexity of the road better. 

Video run1.mp4 : model2.h5 (trained only on track 1, tested on track 1)
video run2.mp4: M.h5 (trained only on track 2 for less than half a lap, tested on track 2)
video run3.mp4: M.h5 (trained only on track 2 for less than half a lap, tested on track 1) 

#### 2. Final Model Architecture

The final model architecture (model.py lines 96-137) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture :

________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 38, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 17, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
_________________________________________________________________
activation_3 (Activation)    (None, 7, 37, 48)         0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 36, 48)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6, 36, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 34, 64)         27712     
_________________________________________________________________
activation_4 (Activation)    (None, 4, 34, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 32, 64)         36928     
_________________________________________________________________
activation_5 (Activation)    (None, 2, 32, 64)         0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 1, 31, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1984)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1984)              0         
_________________________________________________________________
activation_6 (Activation)    (None, 1984)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               198500    
_________________________________________________________________
dropout_3 (Dropout)          (None, 100)               0         
_________________________________________________________________
activation_7 (Activation)    (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_4 (Dropout)          (None, 50)                0         
_________________________________________________________________
activation_8 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_5 (Dropout)          (None, 10)                0         
_________________________________________________________________
activation_9 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 335,419
Trainable params: 335,419
Non-trainable params: 0
_________________________________________________________________

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To teach the network for negative stearing angle, randomly the images are selected to be flipped and the stearing angle is also multiplied with (-1). 

![alt text][image3]

Also it has been recomended to use shearing the images, in away that the buttom of the image is held still and the top part is distoted, since it creates distortion around the top of the road it help introducing random shap turns, thus training for the kind of turns that the simulated data didn't capture. To do that first estimate a random delta x (dx) which will be added to the mid point on the center of the image at the top, given two set of triagnles (1st: mid-point of the image, and bottom_right and bottom_left; 2nd: mid-point+dx and bottom_right and bottom_left). First using cv2.getAffineTransform will find a transform matrix then using cv2.warpAffine to warp the image.

![alt text][image5]


After the collection process, I had about 8037 number of data points. I then preprocessed this data by normalizing the images and cropping the top and buttom before random augmentation(using generator, line 54). The data is randomly shuffled and put 20% of the data into a validation set. The training data is used for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 which tested by driving the model on the track autonomusly without running off the road. Model3.h5 is the model trained with 7 epoch and although the car wont get off the road but is not driving as center as the model trained for 5 or 6 epoch (Model2.h5).
I used an adam optimizer so that manually training the learning rate wasn't necessary.

As an exmple it shown in this plot after epoch 6 the training doesn't improve. (this plot is for track 2 run3.mp4 )

![alt text][image11]


This video was trained on track 1 for 2 laps:
![alt text](Video/run1.mp4)

This video is for a model that only seen track 2 partially:

![alt text](Video/run3.mp4)
