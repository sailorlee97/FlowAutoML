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
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Masking
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
# from  tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import time

class autonetworks():
  def __init__(self,nclasses,nfeatures):
    self.hidden_layers = 1
    self.layer_size = 128
    self.learning_rate = 0.0001
    self.n_class = nclasses
    self.features = nfeatures
    self.metrics = [
      # keras.metrics.TruePositives(name='tp'),
      # keras.metrics.FalsePositives(name='fp'),
      # keras.metrics.TrueNegatives(name='tn'),
      # keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
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

#  输入6*6*1
    model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu', input_shape=[7,7,1]))
    model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Conv2D(64, kernel_size=(2,2),activation='relu'))
    # model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())

    for _ in range(self.hidden_layers - 1):
        model.add(keras.layers.Dense(self.layer_size,
                                     activation = 'relu'))
    model.add(keras.layers.Dense(self.layer_size, activation='relu'))
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

    from utils.csvdb import ConnectMysql

    time_i = time.time()
    connectmysql = ConnectMysql()
    X_train, X_test, y_train, y_test = connectmysql.get_data()


    #y_train = keras.utils.to_categorical(y_train, 10)

    time_o = time.time()
    end_time = time_o - time_i
    print("get data from mysql:",end_time)

    # x =  X_train.drop('appname', axis=1)
    # x = x.values
    # x_train_labeled = np.expand_dims(x, axis=1)

    X = np.expand_dims(X_train.values.astype(float), axis=2)
    inp_size = X.shape[1]
    x_test = np.expand_dims(X_test.values.astype(float), axis=2)

    cnnmodel = autonetworks(3,inp_size)
    estimator = KerasClassifier(build_fn=cnnmodel.buildmodels, epochs=40, batch_size=64, verbose=1)
    estimator.fit(X, y_train)
    y_pred = estimator.predict(x_test)
    print(classification_report(y_test,y_pred))