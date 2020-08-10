#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU, Cropping2D, Dropout,Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import csv
import os
import sklearn
from math import ceil
from random import shuffle
from Img_process import augment_brightness, random_shear, random_flip, rand_shadow2 
from keras.utils.vis_utils import plot_model
from plotting_losses import TrainingPlot

#Param
n_epochs=5
# Set our batch size
batch_size=32

angle_offset=0.153

samples = []

plot_losses = TrainingPlot()

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip the header line
    header = next(reader)
    for line in reader:
        samples.append(line)

        
images=[]
measurements=[]
"""
for line in lines: 
    source_path=line[0]
    filename=source_path.split('/')[-1]
    current_path='../data/IMG/'+filename
    image=cv2.imread(current_path)
    images.append(image)
    measurement=float(line[3])
    measurements.append(measurement)
    
X_train=np.array(images)
Y_train=np.array(measurements)
"""

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image_ = cv2.imread(name)
                    if i==0:                                  
                        angle_ = float(batch_sample[3])
                    elif i==1:
                        angle_ = float(batch_sample[3])+angle_offset    
                       
                    elif i==2:
                        angle_ = float(batch_sample[3])-angle_offset  
             
                    image_=augment_brightness(image_)
                    image_=rand_shadow2(image_)
                    image_, angle_=random_shear(image_,angle_)
                    image_, angle_=random_flip(image_, angle_)
                    images.append(image_)
                    angles.append(angle_)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()

#model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: (x/127.5) - 1.))
        #input_shape=(col,row, ch),
        #output_shape=(col,row, ch)))
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x/127.5) - 0.5, input_shape=(80, 320, 3), output_shape=(80, 320, 3)))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Dropout(.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))


model.add(Flatten())
model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(1))

"""
model.compile(loss='mse',optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2,shuffle= True, nb_epoch=7)
"""
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=[ 'mse'])
plot_model(model, to_file='images/model_vis.png')

history=model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=n_epochs, verbose=1, callbacks=[plot_losses])

#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to JSON
model_json = model.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
    
model.save('Model3.h5')
#print(history.history.keys())
"""
### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc='upper right')
plt.show()
plt.savefig('output/Epoch-{}.png'.format(epoch))
"""


