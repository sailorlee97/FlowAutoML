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
from embedia.model_generator.project_options import *
from embedia.project_generator import ProjectGenerator
from propress_data import preproessData
import pandas as pd

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME  = 'flow10model'
MODEL_FILE    = 'savedmodel/my_model.h5'

processdata = preproessData("./dataset/dataframe.csv")
df0 = pd.read_csv(processdata.path)
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
df = df0[[
    'Pkt_Len_Std', 'Pkt_Len_Var', 'FIN_Flag_Cnt', 'SYN_Flag_Cnt',
    'RST_Flag_Cnt', 'PSH_Flag_Cnt', 'ACK_Flag_Cnt', 'URG_Flag_Cnt',
    'CWE_Flag_Count', 'ECE_Flag_Cnt', 'Down/Up_Ratio', 'Pkt_Size_Avg',
    'Fwd_Seg_Size_Avg', 'Bwd_Seg_Size_Avg', 'Fwd_Byts/b_Avg',
    'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg',
    'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg', 'Subflow_Fwd_Pkts',
    'Subflow_Fwd_Byts', 'Subflow_Bwd_Pkts', 'Subflow_Bwd_Byts',
    'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts', 'Fwd_Act_Data_Pkts',
    'Fwd_Seg_Size_Min', 'Active_Mean', 'Active_Std', 'Active_Max',
    'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min',
    'appname']]

label,x = processdata.df_to_dataset(df)
# mutual = self.select_features(x,label)
x_train,x_test,y_train,y_test = train_test_split(x,label,test_size=0.1,random_state=0)
model = tf.keras.models.load_model(MODEL_FILE)

model._name = 'flow10model'

example_number = 33
sample = x_test[:example_number]
comment= "number %d example for test" % y_test[example_number]

options = ProjectOptions()

# options.project_type = ProjectType.ARDUINO
options.project_type = ProjectType.C
# options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

# options.data_type = ModelDataType.FLOAT
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
options.data_type = ModelDataType.FIXED8

options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
# options.debug_mode = DebugMode.DATA

options.example_data = sample
options.example_comment = comment
options.example_ids = y_test[:example_number]

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)
print("\n"+comment)

