"""
@Time    : 2022/6/23 14:42
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: autoTask.py
@Software: PyCharm
"""
import itertools

from keras.layers import Normalization
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from embedia_test import embediaModel
from utils.plot_cm import plot_conf
from utils.evaultaion import Eva
from models.autonetworks import autonetworks
from utils.csvdb import ConnectMysql
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from sklearn.metrics import f1_score
from utils.selectfeatures import selectfeature
import pandas as pd
import os
import numpy as np
import time
import tensorflow as tf


class autotask():

    def __init__(self, opt):
        self.opt = opt
        self.mysqldata = ConnectMysql()

    def _propress_label(self, labels):
        """Encode target labels with value between 0 and n_classes-1.

        This transformer should be used to encode target values, *i.e.* `y`, and
        not the input `X`.

        :param data:
        :return: numic
        """
        le = LabelEncoder()
        newlabels = le.fit_transform(labels)

        res = {}
        for cl in le.classes_:
            res.update({cl: le.transform([cl])[0]})
        print(res)

        return newlabels, res

    def _process_num(self, num):
        """
            This estimator scales and translates each feature individually such
        that it is in the given range on the training set, e.g. between
        zero and one.

        The transformation is given by::

            X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            X_scaled = X_std * (max - min) + min

        where min, max = feature_range.

        This transformation is often used as an alternative to zero mean,
        unit variance scaling.
        :param num: dataframe or array
        :return:  array
        """
        mm = MinMaxScaler()
        X = mm.fit_transform(num)

        return X

    def df_to_dataset(self, dataframe, shuffle=True, batch_size=32):
        """
        Since the data has been normalized in the edge router, there is no need to do normalization.

        :param dataframe:
        :param shuffle:
        :param batch_size:
        :return:
        """
        dataframe = dataframe.copy()
        labels = dataframe.pop('appname')
        # dataframe.drop(dataframe.columns[[0]], axis=1, inplace=True)
        dataArray = dataframe.values

        # propress labels
        newlabels, res = self._propress_label(labels)
        le = LabelEncoder()
        newlabels = le.fit_transform(labels)

        # process numic values
        # X = self._process_num(dataArray)
        # Since the data has been normalized in the edge router, there is no need to do normalization.

        X = np.expand_dims(dataArray.astype(float), axis=2)

        lenx = len(X)
        newx = X.reshape((lenx, 7, 7, 1))
        print(newx.shape)
        y_train = keras.utils.to_categorical(newlabels, self.opt.nclass)

        ds = tf.data.Dataset.from_tensor_slices((newx, y_train))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds, res, newlabels

    def _get_normalization_layer(self, name, dataset):
        """

        :param name:
        :param dataset:
        :return:
        """
        # Create a Normalization layer for our feature.
        normalizer = Normalization(axis=None)

        # Prepare a Dataset that only yields our feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)

        return normalizer

    def _select_features(self,path):
        '''

        :return: list
        '''

        # sf_model: SelectFromModel = SelectFromModel(ExtraTreesClassifier())
        # sf_model.fit(x_data, y_data)
        # print("保留的特征: ", x_data.columns[sf_model.get_support()])
        # print("特征重要性：", sf_model.estimator_.feature_importances_)
        # a1 = x_data.columns[sf_model.get_support()]
        # a2 = sf_model.estimator_.feature_importances_
        df = pd.read_csv(path)
        dataframe_pd = df.copy()
        dataframe0 = dataframe_pd.drop(dataframe_pd.columns[0], axis=1)
        dataframe = dataframe0.replace([np.inf, -np.inf], np.nan).dropna()
        # dataframe_del = dataframe.drop(['port1','port2','startSec','startNSec','endSec','endNSec'],axis=1)
        labels = dataframe.pop('appname')
        # propress labels
        le = LabelEncoder()
        newlabels = le.fit_transform(labels)
        # newdata = pd.get_dummies(dataframe_del)

        # dataframe.drop(dataframe.columns[[0]], axis=1, inplace=True)
        # data = dataframe.replace([np.inf, -np.inf], np.nan).dropna()
        # labels = data.pop('appname')
        # propress labels
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)
        # newlabels = ._propress_label(labels)

        sf = selectfeature()
        secorndfeatrues = sf.search_corrlate_features(dataframe, newlabels)
        newdataframe = dataframe[secorndfeatrues]
        feauturesimportance = sf.treemodel(newdataframe, newlabels)
        features = list(feauturesimportance[:49]['Features'])
        features.append('appname')

        return features

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
        return X, x_test, y_train, y_test

    def _obtain_data_train_test(self,path = './csv_data/dataframe11.csv'):
        """
        get dataset from mysql
        the get train, val, and test dateset
        use df_to_dataset to get encapsulation
        bulid model and train

        :return: predict values
        """
        # appname=["aiqiyi","bilibili","douyin","hepingjingying","QQyinyue","tengxunhuiyi","wangzherongyao","yuanshen","zhanshuangpamishi","zuoyebang"]

        if os.path.exists(path):
            ## if exist train data, read data
            dataframe_pd = pd.read_csv(path)
            dataframe0 = dataframe_pd.drop(dataframe_pd.columns[0],axis=1)
        else:
            ## read data from sql
            time_i = time.time()
            dataframe0 = self.mysqldata.total_get_data()
            time_o = time.time()
            end_time = time_o - time_i
            print("get data from mysql:", end_time)
            dataframe0.to_csv(path)
        # # print(dataframe0.columns)
        # 36个特征
        # dataframe_del = dataframe0.drop(['port1', 'port2', 'startSec', 'startNSec', 'endSec', 'endNSec'], axis=1)
        dataframe = dataframe0.replace([np.inf, -np.inf], np.nan).dropna()
        # labels = dataframe.pop('appname')
        # newlabel = labels.to_frame('appname')
        # propress labels
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)
        # newdata = pd.get_dummies(dataframe)
        featurelist = []
        if os.path.exists('./log/features'):

            for line in open("./log/features"):
                line = line.strip('\n')
                featurelist.append(line)
        else:

            featurelist = self._select_features(path)

            f = open('./log/features', 'a')
            for i in featurelist:
                f.write(i)
                f.write('\n')
            f.close()

        dataframe1 = dataframe[featurelist]
        print(dataframe1.shape)
        # newdf = dataframe.replace([np.inf, -np.inf], np.nan).dropna()
        # newdf = dataframe[np.isinf(dataframe.T).all()]

        # dataframe3 = pd.concat([dataframe1,labels],axis=1)
        train, test = train_test_split(dataframe1, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)

        train_ds, res_tr, truelabels_tr = self.df_to_dataset(train, batch_size=self.opt.batch_size)
        val_ds, res_va, truelabels_va = self.df_to_dataset(val, shuffle=False, batch_size=self.opt.batch_size)
        test_ds, res_te, truelabels_te = self.df_to_dataset(test, shuffle=False, batch_size=self.opt.batch_size)

        reskey = list(res_te.keys())
        print(reskey)

        # all_inputs = []
        # encoded_features = []
        #
        # columns = dataframe.columns
        # newcolumns = columns[0:len(columns) - 1]
        #
        # # Numeric features.
        # feature_columns = []
        #
        # # numeric cols
        # for header in newcolumns:
        #     feature_columns.append(feature_column.numeric_column(header))
        # feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        cnnmodel = autonetworks(self.opt.nclass, 49)
        model = cnnmodel.buildmodels()
        model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=self.opt.epochs)
        loss, accuracy, precision, recall, auc = model.evaluate(test_ds)

        prediction = model.predict(test_ds, verbose=1)
        predict_label = np.argmax(prediction, axis=1)
        plot_conf(predict_label, truelabels_te, reskey)

        model.save('./savedmodels/my_model.h5')
        # print(classification_report(y_test.argmax(-1), y_pred.argmax(-1)))

        return auc
        # return train_ds,val_ds,test_ds

    def get_f1(self, y_test):
        '''
        get f1
        :param y_test:
        :return:
        '''
        y_pred = self._obtain_data_train_test()
        f1 = f1_score(y_test.argmax(-1), y_pred.argmax(-1), average='weighted')

        return f1

    # 判断
    def train_predit(self):
        # 重新抓取新的数据集
        X_train, X_test, y_train, y_test = self.obtaindata()
        # train_ds, val_ds, test_ds = self.obtaindata_new()
        inp_size = X_train.shape[1]

        time_s = time.time()
        cnnmodel = autonetworks(self.opt.nclass, inp_size)
        model = cnnmodel.buildmodels()
        model.fit(X_train, y_train, epochs=self.opt.epochs, batch_size=64)
        # estimator = KerasClassifier(build_fn=cnnmodel.buildmodels, epochs=self.opt.epochs, batch_size=64, verbose=1)
        # estimator.fit(X_train, y_train)
        time_e = time.time()
        train_time = time_e - time_s
        print("train time:", train_time)
        y_pred = model.predict(y_test)
        print(classification_report(y_test.argmax(-1), y_pred.argmax(-1)))
        # f1 = f1_score(y_test.argmax(-1), y_pred.argmax(-1), average='weighted')
        # return f1

    #    @run_every(30,'day')

    # 此处设计在线学习的module
    def process_run(self):
        if (self.opt.isInitialization == 'yes'):

            if not os.path.exists('./savedmodels/my_model'):
                auc = self._obtain_data_train_test()

            # ## 设计获取参数进行下发
            # loaded_keras_model = load_model("./savedmodels/my_model")
            # keras_to_tflite_converter = tf.lite.TFLiteConverter.from_keras_model(loaded_keras_model)
            # keras_to_tflite_converter.optimizations = [
            #     tf.lite.Optimize.OPTIMIZE_FOR_SIZE
            # ]
            # keras_to_tflite_converter.target_spec.supported_ops = [
            #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            #     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
            # ]
            # keras_tflite = keras_to_tflite_converter.convert()
            #
            # if not os.path.exists('./tflite_models'):
            #     os.mkdir('./tflite_models')
            # with open('./tflite_models/keras_tflite','wb') as f:
            #     f.write(keras_tflite)

        elif (self.opt.isInitialization == 'no'):

            alist = []
            if os.path.exists('./log/features'):

                for line in open("./log/features"):
                    line = line.strip('\n')
                    alist.append(line)
            else:
                raise Exception('dont exist folder!')

            # alist.append('appname')
            try:
                embediatest = embediaModel(
                    OUTPUT_FOLDER='outputs/',
                    PROJECT_NAME='flow10model',
                    MODEL_FILE='savedmodels/my_model.h5',
                    Test_Example='./csv_data/dataframe11.csv',
                    Feature_List=alist
                )
                embediatest.output_model_c()

            except Exception as e:
                print("error:",e.__class__.__name__)
                print(e)
                print("Model embedding fails!")

        #     pred,y= self.test_model(self.opt.label)
        #     eva = Eva('recall')
        #     rec = eva.calculate_recall(y,pred)
        #     precision = eva.calculate_precision(y, pred)
        #
        #     print(rec)
        #     if (rec<0.96 and precision<0.99):
        #         # todo 增量在线学习
        #         predictions = self.train_data()
        #     else:
        #         print("No need to retrain, continue to use the current model.")
        # else:
        #     raise
