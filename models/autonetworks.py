# -*- coding: utf-8 -*-
"""
@Time    : 2022/10/17 15:58
@Author  : zeyi li
@Site    : 
@File    : autonetworks.py
@Software: PyCharm
"""
import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras_flops import get_flops

class autonetworks():

  def __init__(self,nclasses,nfeatures):
    self.hidden_layers = 3
    self.layer_size = 128
    self.learning_rate = 0.0001
    self.n_class = nclasses
    self.features = nfeatures
    self.metrics = [
      # keras.metrics.TruePositives(name='tp'),
      # keras.metrics.FalsePositives(name='fp'),
      # keras.metrics.TrueNegatives(name='tn'),
      # keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      # keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
  def buildmodels(self):

    model = keras.models.Sequential()
    # model.add(Conv2D(filters=32, kernel_size=3, padding= "same",activation='relu',input_shape=[11,7, 1]))
    # model.add(Conv2D(64, 3, padding= "same",activation='relu'))
    # model.add(MaxPool2D(2))
    # model.add(Conv2D(128, 3, padding= "same",activation='relu'))
    # model.add(Conv2D(128, 3, padding= "same",activation='relu'))
    # model.add(MaxPool2D(2))
    # model.add(Flatten())
#  输入8*8*1
    # model.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', input_shape=[8,8,1]))
    # # model.add(Conv2D(8, kernel_size=(2,2), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Conv2D(16, kernel_size=(2,2),activation='relu'))
    # # model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))
    # # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Flatten())
#  输入5*5*1
#     model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu', input_shape=[5,5,1]))
#     # model.add(Conv2D(8, kernel_size=(2,2), activation='relu'))
#     model.add(MaxPool2D(pool_size=(2,2)))
#     # model.add(Conv2D(16, kernel_size=(2,2),activation='relu'))
#     # model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))
#     # model.add(MaxPool2D(pool_size=(2,2)))
#     model.add(Flatten())

    for i in range(4):
        model.add(Conv2D(filters=16*(i+1), kernel_size=(2,2), activation='relu', input_shape=[7,7,1]))
        model.add(BatchNormalization())
        model.add(Dropout(0.05))

    # model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu', input_shape=[7,7,1]))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.05))
    # model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.05))
    # model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.05))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Conv2D(filters=128, kernel_size=(2,2),activation='relu'))
    # model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu'))
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

# RandomizedSearchCV
# 1. 转化为sklearn的model
# 2. 定义参数集合
# 3. 搜索参数
def build_model(hidden_layers = 1,
                layer_size = 30,
                learning_rate = 3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation='relu',
                                 input_shape=77))
    for _ in range(hidden_layers - 1):
        model.add(keras.layers.Dense(layer_size,
                                     activation = 'relu')
                  )
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss = 'mse', optimizer = optimizer)
    return model

# unit test
if __name__ == '__main__':

    model = autonetworks(5, 49)
    cm = model.buildmodels()
    cm.summary()
    flops = get_flops(cm, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")