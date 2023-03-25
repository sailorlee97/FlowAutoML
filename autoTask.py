"""
@Time    : 2022/6/23 14:42
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: autoTask.py
@Software: PyCharm
"""
from keras.layers import Normalization
from keras.saving.save import load_model
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from embedia_test import embediaModel
from utils.plot_cm import plot_conf, multi_roc
from models.autonetworks import autonetworks
from utils.csvdb import ConnectMysql
from tensorflow import keras
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

    def _propress_label(self, labels,sorted_labels):
        """Encode target labels with value between 0 and n_classes-1.

        This transformer should be used to encode target values, *i.e.* `y`, and
        not the input `X`.

        :param data:
        :return: numic
        """
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)
        #
        # res = {}
        # for cl in le.classes_:
        #     res.update({cl: le.transform([cl])[0]})
        # print(res)
        reskey = {}
        for i in sorted_labels:
            reskey.update({i: sorted_labels.index(i)})
        print(reskey)
        # map映射
        labels = labels.map(reskey).values

        print(labels)
        return labels, reskey
        # return newlabels, res

    # alas
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

    def test_single(self, model_path, csv_path, name, label_number, isall=False):
        """
          测试单类预测准确率
          作者：张泽

          Args:
              model_path: 模型的存放路径
              csv_path: 数据的存放路径
              name：单类应用名称
              label_number：此单类应用的编号
              isold：csv文件中存放的是否是全部类数据

          """
        test_model = load_model(model_path)
        # 输出模型结构
        test_model.summary()
        # 输出文件大小（以MB为单位）
        file_size = os.path.getsize(model_path)
        print(f"Model size: {file_size / (1024 * 1024)} MB")
        # 获取特征
        alist = []
        if os.path.exists('./log/features'):

            for line in open("./log/features"):
                line = line.strip('\n')
                alist.append(line)
        else:
            raise Exception('dont exist folder!')
        print(alist)
        # 提取数据
        if isall:
            df_old = pd.read_csv(csv_path)
            df = df_old.loc[df_old['appname'] == name]
        else:
            df = pd.read_csv(csv_path)
        # 数据处理
        df = df[alist[:-1]]
        dataframe = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
        dataArray = dataframe.values
        X = np.expand_dims(dataArray.astype(float), axis=2)
        lenx = len(X)
        newx = X.reshape((lenx, 7, 7, 1))
        true_label = np.ones(lenx) * label_number
        # 预测
        predict = test_model.predict(newx)
        predict_label = predict.argmax(-1)
        acc_x = accuracy_score(true_label, predict_label)
        print(f'准确率：{acc_x}')

    def test_all(self, model_path, csv_path, split_size=0):
        """
          全部类预测准确率
            作者：张泽
          Args:
              model_path: 模型的存放路径
              csv_path: 数据的存放路径
             split_size：划分测试数据集大小

          """
        test_model = load_model(model_path)
        # 输出模型结构
        test_model.summary()
        # 输出文件大小（以MB为单位）
        file_size = os.path.getsize(model_path)
        print(f"Model size: {file_size / (1024 * 1024)} MB")
        # 获取特征
        alist = []
        if os.path.exists('./log/features'):

            for line in open("./log/features"):
                line = line.strip('\n')
                alist.append(line)
        else:
            raise Exception('dont exist folder!')
        print(alist)
        # 提取数据
        df = pd.read_csv(csv_path)
        df = df[alist]
        # 数据处理
        dataframe = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
        if split_size:
            train, test = train_test_split(dataframe, test_size=split_size)
            test_dataframe, res_test, truelabel_test = self.df_to_dataset(test, shuffle=False,
                                                                          batch_size=self.opt.batch_size)
        else:
            test_dataframe, res_test, truelabel_test = self.df_to_dataset(dataframe, shuffle=False,
                                                                          batch_size=self.opt.batch_size)
        res_test = list(res_test.keys())
        # 评估和预测
        loss, accuracy, precision, recall, auc = test_model.evaluate(test_dataframe)
        print(accuracy, precision, recall, auc)
        prediction = test_model.predict(test_dataframe, verbose=1)
        predict_label = np.argmax(prediction, axis=1)
        # 绘制混淆矩阵
        plot_conf(predict_label, truelabel_test, res_test)
        # 绘制ROC曲线
        y_label = keras.utils.to_categorical(truelabel_test, self.opt.nclass)
        multi_roc(y_label, prediction, res_test)
        # 分析报告
        report = classification_report(truelabel_test, predict_label, target_names=res_test)
        print("classification_report:", classification_report(truelabel_test, predict_label, target_names=res_test))

        with open("report.txt", "w") as f:
            f.write(report)
        # 手动计算准确率
        count = 0
        for i in range(0, len(truelabel_test)):
            if truelabel_test[i] == predict_label[i]:
                count += 1
        print(count)
        print(len(truelabel_test))
        print(count / len(truelabel_test))
        # 调用accuracy_score计算准确率
        acc = accuracy_score(truelabel_test, predict_label)
        print(acc)

    def df_to_dataset(self, dataframe, shuffle=True):
        """
        Since the data has been normalized in the edge router, there is no need to do normalization.

        :param dataframe: Data after normalization
        :param shuffle:
        :param batch_size:
        :return:
        """
        dataframe = dataframe.copy()
        labels = dataframe.pop('appname')
        # dataframe.drop(dataframe.columns[[0]], axis=1, inplace=True)
        dataArray = dataframe.values
        # propress labels
        newlabels, res = self._propress_label(labels, self.opt.apps)
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)

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
        ds = ds.batch(self.opt.batch_size)
        ds = ds.prefetch(self.opt.batch_size)
        return ds, res, newlabels

    # alas
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

        sf = selectfeature()
        secorndfeatrues = sf.search_corrlate_features(dataframe, newlabels)
        newdataframe = dataframe[secorndfeatrues]
        feauturesimportance = sf.treemodel(newdataframe, newlabels)
        features = list(feauturesimportance[:49]['Features'])
        features.append('appname')

        return features

    # alas
    # def obtaindata(self):
    #     '''
    #
    #     :return:
    #     '''
    #
    #     time_i = time.time()
    #     # y_train = keras.utils.to_categorical(y_train, 10)
    #     X_train, X_test, y_train, y_test = self.mysqldata.get_data(limitnum=10000)
    #     time_o = time.time()
    #     end_time = time_o - time_i
    #     print("get data from mysql:", end_time)
    #
    #     mm = MinMaxScaler()
    #     X_train = mm.fit_transform(X_train)
    #     X_test = mm.fit_transform(X_test)
    #     y_train = keras.utils.to_categorical(y_train, self.opt.nclass)
    #     y_test = keras.utils.to_categorical(y_test, self.opt.nclass)
    #
    #     # expand dimennsons
    #     X = np.expand_dims(X_train.astype(float), axis=2)
    #     x_test = np.expand_dims(X_test.astype(float), axis=2)
    #     # data = data.drop(labels=['num'], axis=1)
    #     return X, x_test, y_train, y_test

    def _obtain_data_train_test(self):
        """
        get dataset from mysql
        the get train, val, and test dateset
        use df_to_dataset to get encapsulation
        bulid model and train

        :return: predict values
        """

        if os.path.exists('./csv_data/dataframe%d.csv'%(self.opt.nclass)):
            ## if exist train data, read data
            dataframe_pd = pd.read_csv('./csv_data/dataframe%d.csv'%(self.opt.nclass))
            dataframe0 = dataframe_pd.drop(dataframe_pd.columns[0],axis=1)
        else:
            ## read data from sql
            time_i = time.time()
            dataframe0 = self.mysqldata.total_get_data(app=self.opt.appsql,
                                                       featurebase= self.opt.tableName)
            time_o = time.time()
            end_time = time_o - time_i
            print("get data from mysql:", end_time)
            dataframe0.to_csv('./csv_data/dataframe%d.csv'%(self.opt.nclass))

        dataframe = dataframe0.replace([np.inf, -np.inf], np.nan).dropna()
        # labels = dataframe.pop('appname')
        # newlabel = labels.to_frame('appname')
        # propress labels
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)
        # newdata = pd.get_dummies(dataframe)
        featurelist = []
        time_sf = 0
        if os.path.exists('./log/features'):

            for line in open("./log/features"):
                line = line.strip('\n')
                featurelist.append(line)
        else:

            time_feature_s = time.time()

            featurelist = self._select_features('./csv_data/dataframe%d.csv'%(self.opt.nclass))

            time_feature_e = time.time()
            time_sf += time_feature_e-time_feature_s
            print('time of select features: {} min'.format(time_sf/60))

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

        train_ds, res_tr, truelabels_tr = self.df_to_dataset(train)
        val_ds, res_va, truelabels_va = self.df_to_dataset(val, shuffle=False)
        test_ds, res_te, truelabels_te = self.df_to_dataset(test, shuffle=False)

        reskey = list(res_te.keys())
        # print(reskey)

        cnnmodel = autonetworks(self.opt.nclass, 49)
        model = cnnmodel.buildmodels()

        time_train_s = time.time()

        model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=self.opt.epochs)
        loss, accuracy, precision, recall, auc = model.evaluate(test_ds)

        time_train_e = time.time()
        train_time = time_train_e-time_train_s

        print('time of training: {} min'.format(train_time/60))
        time_total = train_time+time_sf
        print('total_time: {} min'.format(time_total/60))

        prediction = model.predict(test_ds, verbose=1)
        predict_label = np.argmax(prediction, axis=1)
        plot_conf(predict_label, truelabels_te, reskey)

        model.save(self.opt.model_file)
        # print(classification_report(y_test.argmax(-1), y_pred.argmax(-1)))
        return auc

    # alas
    def get_f1(self, y_test):
        '''
        get f1
        :param y_test:
        :return:
        '''
        y_pred = self._obtain_data_train_test()
        f1 = f1_score(y_test.argmax(-1), y_pred.argmax(-1), average='weighted')

        return f1

    # alas
    # def train_predit(self):
    #     # 重新抓取新的数据集
    #     X_train, X_test, y_train, y_test = self.obtaindata()
    #     # train_ds, val_ds, test_ds = self.obtaindata_new()
    #     inp_size = X_train.shape[1]
    #
    #     time_s = time.time()
    #     cnnmodel = autonetworks(self.opt.nclass, inp_size)
    #     model = cnnmodel.buildmodels()
    #     model.fit(X_train, y_train, epochs=self.opt.epochs, batch_size=64)
    #     # estimator = KerasClassifier(build_fn=cnnmodel.buildmodels, epochs=self.opt.epochs, batch_size=64, verbose=1)
    #     # estimator.fit(X_train, y_train)
    #     time_e = time.time()
    #     train_time = time_e - time_s
    #     print("train time:", train_time)
    #     y_pred = model.predict(y_test)
    #     print(classification_report(y_test.argmax(-1), y_pred.argmax(-1)))


    # 此处设计在线学习的module
    def process_run(self):
        if (self.opt.isInitialization == 'yes'):

            if not os.path.exists(self.opt.model_file):
                auc = self._obtain_data_train_test()
            else:
                self.test_all(self.opt.model_file, './csv_data/dataframe%d.csv'%(self.opt.nclass))

        elif (self.opt.isInitialization == 'no'):

            alist = []
            if os.path.exists('./log/features'):

                for line in open("./log/features"):
                    line = line.strip('\n')
                    alist.append(line)
            else:
                raise Exception('dont exist folder!')

            try:
                embediatest = embediaModel(
                    OUTPUT_FOLDER=self.opt.output_folder,
                    PROJECT_NAME=self.opt.project_name,
                    MODEL_FILE=self.opt.model_file,
                    Test_Example='./csv_data/dataframe%d.csv'%(self.opt.nclass),
                    Feature_List=alist
                )
                embediatest.output_model_c(self.opt.apps)

            except Exception as e:
                print("error:",e.__class__.__name__)
                print(e)
                print("Model embedding fails!")
