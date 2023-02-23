# -*- coding: utf-8 -*-
"""
@Time    : 2023/1/31 19:59
@Author  : zeyi li
@Site    : 
@File    : embedia_test.py
@Software: PyCharm
"""

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from embedia.model_generator.project_options import *
from embedia.project_generator import ProjectGenerator

import pandas as pd

class embediaModel():

    def __init__(self,OUTPUT_FOLDER,PROJECT_NAME,MODEL_FILE,Test_Example,Feature_List):

        self.OUTPUT_FOLDER = OUTPUT_FOLDER
        self.PROJECT_NAME = PROJECT_NAME
        self.MODEL_FILE = MODEL_FILE
        self.Test_Example = Test_Example
        self.Feature_List = Feature_List

    def output_model_c(self):

        df0 = pd.read_csv(self.Test_Example)
        df = df0[self.Feature_List]
        dataframe = df.copy()
        labels = dataframe.pop('appname')
        le = LabelEncoder()
        label = le.fit_transform(labels)
        dataArray = dataframe.values
        mm = MinMaxScaler()
        X = mm.fit_transform(dataArray)
        X = np.expand_dims(X.astype(float), axis=2)
        lenx = len(X)
        newx = X.reshape((lenx, 6, 6, 1))
        x_train, x_test, y_train, y_test = train_test_split(newx, label, test_size=0.1, random_state=0)
        model = tf.keras.models.load_model(self.MODEL_FILE)

        model._name = self.PROJECT_NAME

        example_number = 33
        sample = x_test[:example_number]
        comment= "number %d example for test" % y_test[example_number]

        options = ProjectOptions()
        options.project_type = ProjectType.C
        options.data_type = ModelDataType.FIXED32
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

# 64个特性
# df=df0[[  'Bwd_Pkt_Len_Std', 'Flow_Byts/s', 'Flow_Pkts/s', 'Flow_IAT_Mean',
#        'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min', 'Fwd_IAT_Tot',
#        'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min',
#        'Bwd_IAT_Tot', 'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max',
#        'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags',
#        'Bwd_URG_Flags', 'Fwd_Header_Len', 'Bwd_Header_Len', 'Fwd_Pkts/s',
#        'Bwd_Pkts/s', 'Pkt_Len_Min', 'Pkt_Len_Max', 'Pkt_Len_Mean',
#        'Pkt_Len_Std', 'Pkt_Len_Var', 'FIN_Flag_Cnt', 'SYN_Flag_Cnt',
#        'RST_Flag_Cnt', 'PSH_Flag_Cnt', 'ACK_Flag_Cnt', 'URG_Flag_Cnt',
#        'CWE_Flag_Count', 'ECE_Flag_Cnt', 'Down/Up_Ratio', 'Pkt_Size_Avg',
#        'Fwd_Seg_Size_Avg', 'Bwd_Seg_Size_Avg', 'Fwd_Byts/b_Avg',
#        'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg',
#        'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg', 'Subflow_Fwd_Pkts',
#        'Subflow_Fwd_Byts', 'Subflow_Bwd_Pkts', 'Subflow_Bwd_Byts',
#        'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts', 'Fwd_Act_Data_Pkts',
#        'Fwd_Seg_Size_Min', 'Active_Mean', 'Active_Std', 'Active_Max',
#        'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min',
#        'appname']]

# 25个特征
# df = df0[
#        ['ACK_Flag_Cnt', 'Active_Max', 'Active_Mean', 'Active_Min', 'Active_Std', 'Bwd_Blk_Rate_Avg',
#         'Bwd_Byts/b_Avg', 'Bwd_Header_Len', 'Bwd_IAT_Max', 'Bwd_IAT_Mean', 'Bwd_IAT_Min', 'Bwd_IAT_Std',
#         'Bwd_IAT_Tot', 'Bwd_PSH_Flags', 'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Mean', 'Bwd_Pkt_Len_Min',
#         'Bwd_Pkt_Len_Std', 'Bwd_Pkts/b_Avg', 'Bwd_Pkts/s', 'Bwd_Seg_Size_Avg', 'Bwd_URG_Flags', 'CWE_Flag_Count',
#         'Down/Up_Ratio', 'ECE_Flag_Cnt', 'appname']]

# 36个特征
# df = df0[[
#     'Pkt_Len_Std', 'Pkt_Len_Var', 'FIN_Flag_Cnt', 'SYN_Flag_Cnt',
#     'RST_Flag_Cnt', 'PSH_Flag_Cnt', 'ACK_Flag_Cnt', 'URG_Flag_Cnt',
#     'CWE_Flag_Count', 'ECE_Flag_Cnt', 'Down/Up_Ratio', 'Pkt_Size_Avg',
#     'Fwd_Seg_Size_Avg', 'Bwd_Seg_Size_Avg', 'Fwd_Byts/b_Avg',
#     'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg',
#     'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg', 'Subflow_Fwd_Pkts',
#     'Subflow_Fwd_Byts', 'Subflow_Bwd_Pkts', 'Subflow_Bwd_Byts',
#     'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts', 'Fwd_Act_Data_Pkts',
#     'Fwd_Seg_Size_Min', 'Active_Mean', 'Active_Std', 'Active_Max',
#     'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min',
#     'appname']]

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