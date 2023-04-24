# -*- coding: utf-8 -*-
"""
@Time    : 2023/1/31 19:59
@Author  : zeyi li
@Site    : 
@File    : embedia_test.py
@Software: PyCharm
"""
import math

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from embedia.model_generator.project_options import *
from embedia.project_generator import ProjectGenerator
from models.base import BaseClassification

import pandas as pd

class embediaModel(BaseClassification):

    def __init__(self,OUTPUT_FOLDER,PROJECT_NAME,MODEL_FILE,Test_Example,Feature_List,opt):

        super(embediaModel, self).__init__(opt)
        self.OUTPUT_FOLDER = OUTPUT_FOLDER
        self.PROJECT_NAME = PROJECT_NAME
        self.MODEL_FILE = MODEL_FILE
        self.Test_Example = Test_Example
        self.Feature_List = Feature_List

    def _delinf(self,dataframe):
        newdf = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

        return newdf

    def output_model_c(self,sorted_labels):

        df0 = pd.read_csv(self.Test_Example)
        df = df0[self.Feature_List]
        newdf = self._delinf(df)
        dataframe = newdf.copy()
        labels = dataframe.pop('appname')

        # propress labels
        newlabels, res = self._propress_label(labels, sorted_labels)
        dataArray = dataframe.values
        # mm = MinMaxScaler()
        # X = mm.fit_transform(dataArray)
        X = np.expand_dims(dataArray.astype(float), axis=2)
        lenx = len(X)
        newx = X.reshape((lenx,int(math.sqrt(self.opt.number_features)), int(math.sqrt(self.opt.number_features)), 1))
        x_train, x_test, y_train, y_test = train_test_split(newx, newlabels, test_size=0.1, random_state=0)
        model = tf.keras.models.load_model(self.MODEL_FILE)

        model._name = self.PROJECT_NAME

        example_number = 33
        sample = x_test[:example_number]
        comment= "number %d example for test" % y_test[example_number]

        options = ProjectOptions()
        options.project_type = ProjectType.C

        transformdict = {
            'FLOAT': ModelDataType.FLOAT,
            'FIXED32': ModelDataType.FIXED32,
            'FIXED16': ModelDataType.FIXED16,
            'FIXED8': ModelDataType.FIXED8,
        }
        
        options.data_type = transformdict.get(self.opt.embedd_data_type)
        options.debug_mode = DebugMode.DISCARD
        options.example_data = sample
        options.example_comment = comment
        options.example_ids = y_test[:example_number]

        options.files = ProjectFiles.ALL

        generator = ProjectGenerator(options)
        generator.create_project(self.OUTPUT_FOLDER, self.PROJECT_NAME, model, options)

        print("Project", self.PROJECT_NAME, "exported in", self.OUTPUT_FOLDER)
        print("\n" + comment)

# OUTPUT_FOLDER = 'outputs/'
# PROJECT_NAME  = 'flow10model'
# MODEL_FILE    = 'savedmodels/my_model.h5'
#
#
# df0 = pd.read_csv('.\csv_data\dataframe10.csv')


# dataframe = df.copy()
# labels = dataframe.pop('appname')
# le = LabelEncoder()
# label = le.fit_transform(labels)

# dataframe.drop(dataframe.columns[[0]], axis=1, inplace=True)
# dataArray = dataframe.values
# mm = MinMaxScaler()
# X = mm.fit_transform(dataArray)
# X = np.expand_dims(X.astype(float), axis=2)
# lenx = len(X)
# newx = X.reshape((lenx,6,6,1))


# mutual = self.select_features(x,label)
# x_train,x_test,y_train,y_test = train_test_split(newx,label,test_size=0.1,random_state=0)
# model = tf.keras.models.load_model(MODEL_FILE)
#
# model._name = 'flow10model'
#
# example_number = 33
# sample = x_test[:example_number]
# comment= "number %d example for test" % y_test[example_number]
#
# options = ProjectOptions()

# options.project_type = ProjectType.ARDUINO
# options.project_type = ProjectType.C
# options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

# options.data_type = ModelDataType.FLOAT
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8

# options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
# options.debug_mode = DebugMode.DATA

# options.example_data = sample
# options.example_comment = comment
# options.example_ids = y_test[:example_number]

# options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

# generator = ProjectGenerator(options)
# generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

# print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)
# print("\n"+comment)