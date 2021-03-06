# **Behavioral Cloning** 

## Third project for Udacity's Self Driving Car Nanodegree - Term 1

### This is the project writeup. [Instructions](Instructions.md) are here. Project's expectations are called rubic points and they are [here](https://review.udacity.com/#!/rubrics/432/view).

See how model-4.2 drives on [track 2](https://youtu.be/Fz2WwQxrifo) or see [driver's view](https://youtu.be/4UF54O0kL4s) created by drive.py

See how model-3.4a drives on [track 1](https://youtu.be/wIPkZwhLO9o) or see [driver's view](https://youtu.be/kE68FkYNxZk) created by drive.py

Both of the models are available [here](https://drive.google.com/drive/folders/0B2wt3poVYbAYTUZFcURKOEVuMms?usp=sharing).

**Things I changed after getting review comments**

* Added dropout of 0.7 just before fully connected layer of 10
* Slightly changed architecture by removing fully connected layer of 1164 neurons
* Switched back and forth between keras versions to see if it makes any difference
* model-3.4a is on lower version of keras and model-4.2 is on higher
* I know the project requirement was to drive on only track 1... and in order to make it outstanding, same model should be driving car on track 2.
* I couldn't achive outstanding requirement but I was able to produce model that drives flawlessly on track 2 only (model-4.2) 
* Tried converting all images to YUV color space, couldn't see any improvements
* Tried various probabilities of dropout, and placed drouput at many different places. Finally, settled on dropout with 0.7 keep probability just before fully connected layer of 10
* Added a few more data points and new number is 26794, up from old number of 20203 data points. 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_2017_10_07_00_30_50_845.jpg "Normal Image"
[image3]: ./examples/center_2017_10_07_00_30_32_338.jpg "Recovery Image"
[image4]: ./examples/center_2017_10_07_00_32_53_483.jpg "Recovery Image"
[image5]: ./examples/center_2017_10_07_00_33_16_514.jpg "Recovery Image"
[image6]: ./examples/center_2017_10_07_00_33_16_409.jpg "Normal Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* lib.py containing functions to read log file & generate test and validation data
* drive.py for driving the car in autonomous mode
* model-3.4a.h5 containing a trained convolution neural network; it drives perfectly on track 1
* model-4.2.h5 containing a trained convolution neural network; it drives perfectly on track 2
* README.md (this file) summarizing the results
* Video of the simultations are here [track 2 aerial view](https://youtu.be/Fz2WwQxrifo) or [track 2 driver's view](https://youtu.be/4UF54O0kL4s) & [track 1 aerial view](https://youtu.be/wIPkZwhLO9o) or [track 1 driver's view](https://youtu.be/kE68FkYNxZk) 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model-3.4a.h5

python drive.py model-4.2.h5
```

#### 3. Submission code is usable and readable

Pipeline was built with "trial & error" process. I first built the neural network based on LeNet. It created a good working model but car wouldn't finish whole lap. Model built with LeNet was about 450MB big.

I then switched to [NVIDIA's architecture](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-768x1095.png) and I was able to train network faster with the same data. Model generated was about 4MB... less than 1% of the size compared to LeNet.

I also learned that training network with "more" data is not always a good idea. I kept on adding more and more data to generate better model. Finally, model was able to drive car multiple laps (actually indefinetely) with about 80,000 images including center, left, right and flipped images (20,000 original) but when I added more data, model actually performed worse. So, I came back to the optimal size of the training data and it now works like a charm. *EDIT* - Current data count is 26794; See comments on the top of the page for edits done after review.

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As explained above, network was with multiple well-known models. The final model chosen (due to reasons described above) was [NVIDIA's self driving car model](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-768x1095.png). This is how it looks like - 

```sh
num_classes = 1
epochs = 3

# input image dimensions
img_rows, img_cols, img_channels = 160, 320, 3
input_shape = (img_rows, img_cols, img_channels)

# NVIDIA-like implementation
model = Sequential()
model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0)))) # 70 rows pixels from the top, 25 from bottom, 0 from left & right 
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(num_classes))
```

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (lib.py line 25-33). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 84).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and clock-wise driving. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to - 

* Create a small and simple sequential model to do sanity testing. I tested out that all moving parts are connected and I can indeed simulate driving with network trained on AWS.
* Add positive training data. I added 4 laps on center driving data.
* Create Le-Net like network. I thought this model would work as this is a well-known model. But, I didn't realize that I shouldn't be using classification model for regression problem. 
* Add negative data. I added a few recovery and reverse track data.
* As I explained above, models created from Le-Net network was not efficient. Car won't drive and size of the model was too big.
* Create NVIDIA like network. As I explained before, this worked. Size of the model was 1% of model created before and car would drive indefientely. 
* It took me about 17 attempts to come to the final model (model10.h5) that I am submitting.
* In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
* I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
* To combat the overfitting, I modified the model and gathered more data. I also learned that training network with "more" data is not always a good idea. I kept on adding more and more data to generate better model. Finally, model was able to drive car multiple laps (actually indefinetely) with about 80,000 images including center, left, right and flipped images (20,000 original) but when I added more data, model actually performed worse. So, I came back to the optimal size of the training data and it now works like a charm. *EDIT* - Current data count is 26794; See comments on the top of the page for edits done after review.
* Few first attempts

First attempt - 

```
Train on 12109 samples, validate on 3028 samples
Epoch 1/10
12109/12109 [==============================] - 27s - loss: 2.0462 - val_loss: 2.6014
...
...
```

Third attempt - 

```
Train on 24219 samples, validate on 6055 samples
Epoch 1/10
24219/24219 [==============================] - 49s - loss: 3.3275 - val_loss: 3.2934
...
...
```

One of the late attempts (33rd) - 

```
Train on 85740 samples, validate on 21436 samples
Epoch 1/5
146758/146758 [==============================] - 341s - loss: 0.0667 - val_loss: 0.0573
...
...
```

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 50-101) consisted of a convolution neural network with the following layers and layer sizes.

```sh
model = Sequential()
model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0)))) # 70 rows pixels from the top, 25 from bottom, 15 from left & right 
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(keep_prob))
model.add(Dense(10))
model.add(Dense(num_classes))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drift back to center of the lane. These images show what a recovery looks around bridge, red colored shoulder and near water.

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points. I couldn't finish one lap driving on track two. It was very difficult to drive. So, I got only partial track's data.

To augment the data sat, I also flipped images and angles thinking that this would increase the data set and provide more generelization. 

After the collection process, I had 20203 number of data points. I then preprocessed this data by using generator.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by getting near 0.00369 training loss and 0.00356 validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
