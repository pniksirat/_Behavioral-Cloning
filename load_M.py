#!/usr/bin/env python

import numpy as np
import tensorflow as tf
#from tensorflow import keras
import tensorflow.contrib.keras as keras
import matplotlib.pyplot as plt
from keras.models import Model, load_model, model_from_json
from keras.utils.vis_utils import plot_model



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
r_model = model_from_json(loaded_model_json)
# load weights into new model
r_model.load_weights("Model.h5")
print("Loaded model from disk")

#reconstructed_model
#r_model = keras.models.load_model("model2.h5")
#r_model=load_model("model2.h5")

r_model.summary()
r_model.compile(loss='mse', optimizer='adam')
plot_model(r_model, to_file='images/model_vis.png')

#https://www.dataquest.io/blog/learning-curves-machine-learning/
#https://datascience.stackexchange.com/questions/45954/keras-plotting-loss-and-mse
#https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
print(history.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc='upper right')
plt.show()
