import csv 
import cv2
import numpy as np


images = []
measurements = []

data_dirs = ['track1-normal', 'track1-reverse']
dirs_lr = {'track1-normal': True, 
           'track1-reverse': True, 
           'track1-weaving': False}

for d in data_dirs:
    lines = []
    with open('/home/workspace/' + d + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    for line in lines:
        if line[0] == 'center':
            continue
        correction = 0.3
        image = cv2.imread(line[0])
        image_left = cv2.imread(line[1])
        image_right = cv2.imread(line[2])
        measurement = float(line[3])

        images.append(image)
        measurements.append(measurement)
        images.append(image_left)
        measurements.append(measurement + correction)
        images.append(image_right)
        measurements.append(measurement - correction)
        if dirs_lr[d]:
            images.append(cv2.flip(image,1))
            measurements.append(measurement*-1.0)
            images.append(cv2.flip(image_left,1))
            measurements.append((measurement + correction)*-1.0)
            images.append(cv2.flip(image_right,1))
            measurements.append((measurement - correction)*-1.0)

    
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D

###### FIRST MODEL 
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) # normalize and mean center
# model.add(Flatten(input_shape=(160,320,3)))
# model.add(Dense(1))

# ###### LENET
# model = Sequential()
# model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
# model.add(Lambda(lambda x: (x / 255.0) - 0.5)) 
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dropout(0.5))
# model.add(Dense(84))
# model.add(Dropout(0.5))
# model.add(Dense(1))

###### NVIDIA Model
model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5)) 
model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=3)

model.save('model.h5')