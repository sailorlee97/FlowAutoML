"""
@Time    : 2022/6/23 14:42
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: autoTask.py
@Software: PyCharm
"""
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow import feature_column
from utils.evaultaion import Eva
from models.autonetworks import autonetworks
from utils.csvdb import ConnectMysql
from tensorflow import keras
import pandas as pd
import numpy as np
import time
import tensorflow as tf

class autotask():

    def __init__(self,opt):
        self.opt = opt
        self.mysqldata = ConnectMysql()

    def df_to_dataset(self, dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('appname')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

    def select_from_model(self,x_data, y_data):
        '''

        :param x_data: dataframe
        :param y_data: dataframe
        :return: list
        '''


        # 使用ExtraTrees作为特征筛选的依据
        sf_model: SelectFromModel = SelectFromModel(ExtraTreesClassifier())
        sf_model.fit(x_data, y_data)

        new_columns = x_data.columns[sf_model.get_support()]
        columns = new_columns.tolist()
        print("保留的特征: ", columns)
        # print("特征重要性：", sf_model.estimator_.feature_importances_)
        # sf_model.threshold_
        # sf_model.get_support()  # get_support函数来得到到底是那几列被选中了
        return columns

    def obtaindata(self):
        '''

        :return:
        '''

        time_i = time.time()
        # y_train = keras.utils.to_categorical(y_train, 10)
        X_train, X_test, y_train, y_test = self.mysqldata.get_data(limitnum=10000)
        time_o = time.time()
        end_time = time_o - time_i
        print("get data from mysql:", end_time)

        new_columns = self.select_from_model(X_train,y_train)

        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.fit_transform(X_test)
        y_train = keras.utils.to_categorical(y_train, self.opt.nclass)
        y_test = keras.utils.to_categorical(y_test, self.opt.nclass)
        # train_ds = self.df_to_dataset(X_train)
        # feature_columns = []
        #
        # # numeric cols
        # for header in X_train.columns:
        #     feature_columns.append(feature_column.numeric_column(header))
        # feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

        # expand dimennsons
        X = np.expand_dims(X_train.astype(float), axis=2)
        x_test = np.expand_dims(X_test.astype(float), axis=2)
        # data = data.drop(labels=['num'], axis=1)
        return X ,x_test,y_train, y_test

#判断

    def train_predit(self):
        # 重新抓取新的数据集
        X_train, X_test, y_train, y_test = self.obtaindata()
        inp_size = X_train.shape[1]

        time_s = time.time()
        cnnmodel = autonetworks(self.opt.nclass, inp_size)
        model = cnnmodel.buildmodels()
        model.fit(X_train, y_train, epochs=self.opt.epochs,batch_size=64)
        # estimator = KerasClassifier(build_fn=cnnmodel.buildmodels, epochs=self.opt.epochs, batch_size=64, verbose=1)
        # estimator.fit(X_train, y_train)
        time_e = time.time()
        train_time = time_e - time_s
        print("train time:",train_time)

        y_pred = model.predict(X_test)
        print(classification_report(y_test.argmax(-1), y_pred.argmax(-1)))

#    @run_every(30,'day')

    # 此处设计在线学习的module
    def process_run(self):
        if(self.opt.isInitialization=='yes'):

            predictions = self.train_predit()

            # todo  设计获取参数进行下发

        elif(self.opt.isInitialization=='no'):

            pred,y= self.test_model(self.opt.label)
            eva = Eva('recall')
            rec = eva.calculate_recall(y,pred)
            precision = eva.calculate_precision(y, pred)

            print(rec)
            if (rec<0.96 and precision<0.99):
                predictions = self.train_data()
            else:
                print("No need to retrain, continue to use the current model.")
        else:
            raise