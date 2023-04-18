# -*- coding: utf-8 -*-
"""
@Time    : 2022/10/17 15:58
@Author  : zeyi li
@Site    : 
@File    : autonetworks.py
@Software: PyCharm
"""
import math

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization,Dropout
from tensorflow import keras
from keras_flops import get_flops

class autonetworks():

  def __init__(self,nclasses,nfeatures):
    self.hidden_layers = 2
    self.layer_size = 128
    self.learning_rate = 0.0002
    self.n_class = nclasses
    self.features = nfeatures
    self.metrics = [
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
    ]
  def buildmodels(self):

    model = keras.models.Sequential()
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', input_shape=[int(math.sqrt(self.features)), int(math.sqrt(self.features)), 1]))
    model.add(BatchNormalization())
    model.add(Dropout(0.05))

    for i in range(3):
        model.add(Conv2D(filters=64*(i+2), kernel_size=(2,2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.05))

    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())

    for _ in range(self.hidden_layers - 1):
        model.add(keras.layers.Dense(self.layer_size,
                                     activation = 'relu'))
        model.add(Dropout(0.05))

    model.add(keras.layers.Dense(self.layer_size, activation='tanh'))
    model.add(Dense(self.n_class, activation='softmax'))
    optimizer = keras.optimizers.RMSprop(self.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=self.metrics)
    return model


# unit test
if __name__ == '__main__':

    model = autonetworks(5, 36)
    cm = model.buildmodels()
    cm.summary()
    flops = get_flops(cm, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")