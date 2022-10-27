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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, concatenate
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

  def buildmodels(self):

    model = keras.models.Sequential()
    model.add(Conv1D(64, 3, input_shape=(self.features, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    for _ in range(self.hidden_layers - 1):
        model.add(keras.layers.Dense(self.layer_size,
                                     activation = 'relu'))

    # model.add(keras.layers.Dense(self.layer_size, activation='selu'))
    model.add(Dense(self.n_class, activation='softmax'))
    optimizer = keras.optimizers.RMSprop(self.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
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