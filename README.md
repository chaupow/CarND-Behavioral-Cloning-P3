# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I tried following the hints and pointers from the lecture, starting with a very easy model to get _any_ steering measurement, then LENET, LENET with dropout and pooling, then the NVIDIA one. While transitioning from one model to another, I added more training data.

Flipping images was an important step to get the car drive till the bridge. I was not able to handle curves well until I added the left and right camera images. In fact, the first time the car successfully drove the whole track was with LENET and a correction of `0.5` but the drive was very very swirvy from left to right.

#### 2. The final model

My model is similar to the model that NVIDIA uses which is introduced in the chapter `A more powerful network`. In the beginning, the input images are cropped to exclude the sky and the car and the values are normalized to values `[-0.5, 0.5]` aimed to have a mean of zero.

It uses four convolutional layers. The first two use a stride of `(2,2)` to reduce the output size faster. The number of filters is `24, 36, 64, 64`. All convolutional layers use ReLU activation.

Afterwards, all values are flattened and with four dense layers end up with one output value that predicts the steering.

#### 2. Attempts to reduce overfitting in the model

Before I ended up with the final model, I have tried to use the `LENET` model and added max pooling and dropout to reduce overfitting. With the final model, no special actions where required to reduce further overfitting as there was no big problem. I am assuming that a `stride > 1` also helps but running on my test set, the validation error was almost as low as the training error:

```
Epoch 1/3
27514/27514 [==================] - 59s - loss: 0.0439 - val_loss: 0.0307
Epoch 2/3
27514/27514 [==================] - 55s - loss: 0.0310 - val_loss: 0.0340
Epoch 3/3
27514/27514 [==================] - 54s - loss: 0.0293 - val_loss: 0.0318
```

Another reason why the final model needed less effort on reducing overfitting could be that by the time I had the final model, I had enough recordings and data that overfitting wasn't that much of an issues anymore. 

I ran the model for three epochs. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

At first, I had a lot of recordings trying to drive really slow and accurate (~4mph) 

- trying to stay on center line in both directions on track 1
- weaving back and forth from the center line but only recording when getting back to the center line
- additional data when on the bridge and not driving straight to steer back to the center
- additional curves especially after the bridge

Using the data, I added the images to the training set by

- using the original data + steering measurement
- using a flipped image and the negative measured steering
- using the left and right images with a correction of `0.3` (this was not done for the recordings from the weaving ride)

This got me to the bridge easily but the car often struggled with the sharp curves after the bridge.

As the car is driving ~9mph in the autonomous mode and I drove _really_ slow when recording I figured that a 9mph driving car probably needs heavier steering than a slouching 4mph car and that the steering measurement in my data is too small.

I decided to start over completely new, focusing on keeping a steady 9mph. The car managed to successfully drive track 1 only using data of two laps (one in each direction) while trying to stay on the center line.

Adding data where I tried to capture how to recover from the sidelines made the car perform worse.

Risk is that the model performs well on the track and never leaves the center line too much but if it were to leave the center line too much it would be unable to drive correctly. It is possible though that the right and left images are enough to prevent this. I would have loved to try the autonomous mode from a starting position other than the middle but this was not possible.

I didnt use data from track 2 because I was not able to drive the track successfully myself.


