"""
@Time    : 2022/6/23 14:42
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: autoTask.py
@Software: PyCharm
"""
from array import array

from keras.saving.save import load_model
from pandas import DataFrame
from pandas.errors import SettingWithCopyWarning
from shap import Explanation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from warnings import simplefilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM
from embedia_test import embediaModel
from options import Options
from utils.plot_cm import plot_conf, multi_roc
from models.autonetworks import autonetworks
from models.base import BaseClassification
from utils.csvdb import ConnectMysql
from tensorflow import keras
from utils.selectfeatures import selectfeature
from tensorflow.python.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import pandas as pd
import os
import math
import numpy as np
import time
import tensorflow as tf
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=SettingWithCopyWarning)



class autotask(BaseClassification):

    def __init__(self, opt):

        super(autotask, self).__init__(opt)
        # self.opt = opt
        self.mysqldata = ConnectMysql()

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
        # if 'ip1' in datacol or 'ip2' in datacol or 'port1' in datacol or 'port2' in datacol:
        dataframe0 = self.mysqldata.total_get_data(app=self.opt.appsql,
                                                   featurebase='AP_flowfeature_withIP_test')
        # dataframe0 = pd.read_csv('./csv_data/dataframe4.csv')
        datacol = dataframe0.columns
        # datacol1 = datacol[1:]
        #
        df.columns = datacol[:-1]
        # df.drop(columns=['port1', 'port2'], axis=1, inplace=True)
        # #     调用函数将ip地址转为数值
        # df['ip_1'] = self.ipToValue(df.pop('ip1'))
        # df['ip_2'] = self.ipToValue(df.pop('ip2'))
        df = df[alist[:-1]]

        dataframe = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
        dataArray = dataframe.values
        X = np.expand_dims(dataArray.astype(float), axis=2)
        lenx = len(X)
        newx = X.reshape((lenx, 5, 5, 1))
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
        if os.path.exists('./log/features_%s'%(self.opt.project_name)):

            for line in open('./log/features_%s'%(self.opt.project_name)):
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
        # standard
        pd_std = pd.read_csv('./log/std.csv')
        pd_std.columns = ['key', 'value']
        dict_std = dict(zip(pd_std['key'], pd_std['value']))
        pd_mean = pd.read_csv('./log/mean.csv')
        pd_mean.columns = ['key', 'value']
        dict_mean = dict(zip(pd_mean['key'], pd_mean['value']))
        dataframe1 = self._process_stand(alist, dataframe, dict_mean, dict_std)

        if split_size:
            train, test = train_test_split(dataframe1, test_size=split_size)
            test_dataframe, res_test, truelabel_test = self.df_to_dataset(test, shuffle=False)
        else:
            test_dataframe, res_test, truelabel_test = self.df_to_dataset(dataframe1, shuffle=False)
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

    def test_new_all(self, model_path, tableName, split_size=0):
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

        # 提取数据
        # 从数据库中拿数据
        dataframe_test = self.mysqldata.total_get_data(app=self.opt.appsql,
                                                   featurebase=tableName)
        # 获取特征
        if self.opt.enable_all_features == 'no':
            alist = []
            if os.path.exists('./log/features'):

                for line in open("./log/features"):
                    line = line.strip('\n')
                    alist.append(line)
            else:
                raise Exception('dont exist folder!')
            print(alist)

            df = dataframe_test[alist]
        else:
            df = dataframe_test.drop(columns=['s_fHeaderBytes','s_bHeaderBytes','s_lenSd','s_fLenSd','s_fIATSd',
                                                 's_bLenSd','s_bIATSd','s_flowIATSd','s_idleSd','s_activeSd'])
            alist = df.columns
        # 数据处理
        dataframe = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

        # standard
        pd_std = pd.read_csv('./log/std.csv')
        pd_std.columns = ['key','value']
        dict_std = dict(zip(pd_std['key'],pd_std['value']))
        pd_mean = pd.read_csv('./log/mean.csv')
        pd_mean.columns = ['key','value']
        dict_mean = dict(zip(pd_mean['key'], pd_mean['value']))
        dataframe1 = self._process_stand(alist,dataframe,dict_mean,dict_std)

        if split_size:
            train, test = train_test_split(dataframe1, test_size=split_size)
            test_dataframe, res_test, truelabel_test = self.df_to_dataset(test, shuffle=False)
        else:
            test_dataframe, res_test, truelabel_test = self.df_to_dataset(dataframe1, shuffle=False)
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
        self.opt.apps.append('background')
        # propress labels
        newlabels, res = self._propress_label(labels, self.opt.apps)
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)

        # process numic values
        # X = self._process_standard(dataArray)
        # Since the data has been normalized in the edge router, there is no need to do normalization.

        # preprocess
        # dataArray = self._featureBernoulliRBM(dataArray)

        X1 = np.expand_dims(dataArray.astype(float), axis=2)

        lenx = len(X1)
        newx = X1.reshape((lenx, int(math.sqrt(self.opt.number_features)), int(math.sqrt(self.opt.number_features)), 1))
        print(newx.shape)
        y_train = keras.utils.to_categorical(newlabels, self.opt.nclass)

        ds = tf.data.Dataset.from_tensor_slices((newx, y_train))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(self.opt.batch_size)
        ds = ds.prefetch(self.opt.batch_size)
        return ds, res, newlabels

    def _process_stand(self,
                       featurelist,dataframe,data_mean,data_std
                       ) -> DataFrame:

        for _ in featurelist[:-1]:
            if data_std[_] == 0:
                dataframe[_] = 0
            else:
                dataframe[_] = dataframe[_].map(lambda x: (x - data_mean[_]) / data_std[_])

        return dataframe[featurelist]

    def _select_features(self,path) -> list:
        '''
        Feature selection algorithm based on tree model

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
        dataframe = dataframe.drop(['s_fMinSegSize','s_lenMin','s_fPayloadSum','s_bytesPerSecond','s_fLenMin','s_bwdSegSizeAvg','payloadSum','s_ack','s_bHeaderBytes','s_bPktsCounts','s_bLenMean',
        's_bPayloadSum','s_flowIATMean','s_fActDataPkts','s_PktsPerSecond','s_fPktsCounts','s_fPktsPerSecond','s_bLenSd','s_bPktsPerSecond','s_bIATMean','proto','s_psh','s_bIATTotal','s_fBulkDuration','s_fPSHCnt']
        ,axis=1)
        labels = dataframe.pop('appname')
        # propress labels
        le = LabelEncoder()
        newlabels = le.fit_transform(labels)

        columns = dataframe.columns
        # columns = columns[:-1]
        df_ = self._process_standard(dataframe)
        dataframe = DataFrame(data=df_, columns=columns)
        sf = selectfeature()
        secorndfeatrues = sf.search_corrlate_features(dataframe, newlabels,self.opt.number_features)
        newdataframe = dataframe[secorndfeatrues]
        feauturesimportance = sf.treemodel(newdataframe, newlabels)
        features = list(feauturesimportance[:self.opt.number_features]['Features'])
        features.append('appname')

        return features

    def _getDataFromcsv(self) -> DataFrame:

        if os.path.exists('./csv_data/dataframe%d.csv' % (self.opt.nclass)):
            ## if exist train data, read data
            dataframe_pd = pd.read_csv('./csv_data/dataframe%d.csv' % (self.opt.nclass))
            dataframe0 = dataframe_pd.drop(dataframe_pd.columns[0], axis=1)
        else:
            ## read data from sql
            time_i = time.time()
            dataframe0 = self.mysqldata.total_get_data(app=self.opt.appsql,
                                                       featurebase=self.opt.tableName)
            datacol = dataframe0.columns
            time_o = time.time()
            end_time = time_o - time_i
            print("get data from mysql:", end_time)
            if 'ip1' in datacol or 'ip2' in datacol or 'port1' in datacol or 'port2' in datacol:
                dataframe0.drop(columns=['port1', 'port2'], axis=1, inplace=True)
                #     调用函数将ip地址转为数值
                dataframe0['ip_1'] = self.ipToValue(dataframe0.pop('ip1'))
                dataframe0['ip_2'] = self.ipToValue(dataframe0.pop('ip2'))
                dataframe0.to_csv('./csv_data/dataframe%d.csv' % (self.opt.nclass))
            else:
                dataframe0.to_csv('./csv_data/dataframe%d.csv' % (self.opt.nclass))

        dataframe = dataframe0.replace([np.inf, -np.inf], np.nan).dropna()

        return dataframe

    def _select_data_stand(self,dataframe,app,featurelist):

        global strname
        dictname = {
            'moba':['王者荣耀'],
            'shoot':['香肠派对'],
            'rpg': ['原神'],
            'sport': ['狂野飙车9竞速传奇'],
            'vr': ['picoshipin'],
            'social': ['知乎'],
            'meeting': ['网易会议'],
            'education': ['猿辅导'],
            'video': ['bilibili'],
            'music': ['QQ音乐']
        }

        for i in dictname.keys():
            if app in dictname[i]:
                strname = i
                break

        dataframe1 = dataframe[featurelist]
        dataframe_app = dataframe1[dataframe1['appname'] == app]
        data_std = dataframe_app.std()
        pd_data_std = pd.DataFrame(data_std)
        pd_data_std.to_csv('./log/std_%s.csv'%(strname))
        data_mean = dataframe_app.mean()
        pd_data_mean = pd.DataFrame(data_mean)
        pd_data_mean.to_csv('./log/mean_%s.csv'%(strname))

        for _ in featurelist[:-1]:
            if data_std[_] == 0:
                dataframe_app[_] = 0
            else:
                dataframe_app[_] = dataframe_app[_].map(lambda x: (x - data_mean[_]) / data_std[_])

        return dataframe_app

    def _featureTransform(self,x,y):

        model = LinearDiscriminantAnalysis(n_components = 4)
        x_la = model.fit_transform(x,y)

        return x_la

    def _featurepca(self,x):

        single_pca = PCA(n_components=25)
        x_la = single_pca.fit_transform(x)

        return x_la

    def _featureBernoulliRBM(self,x):

        model = BernoulliRBM(n_components=25)
        x_la = model.fit_transform(x)

        return x_la

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
            dataframe0 = dataframe0.drop('pcaptype',axis=1)
            dataframe1 = self.mysqldata.get_background(featurebase= self.opt.tableName
                                                       )
            dataframe1['appname'] = 'background'
            dataframe1 = dataframe1.drop('pcaptype',axis=1)
            dataframe0 = dataframe0.append(dataframe1)
            datacol = dataframe0.columns
            time_o = time.time()
            end_time = time_o - time_i
            print("get data from mysql:", end_time)
            if 'ip1' in datacol or 'ip2' in datacol or 'port1' in datacol or 'port2' in datacol:
                dataframe0.drop(columns=['port1','port2'],axis=1,inplace=True)
            #     调用函数将ip地址转为数值
                dataframe0['ip_1'] = self.ipToValue(dataframe0.pop('ip1'))
                dataframe0['ip_2'] = self.ipToValue(dataframe0.pop('ip2'))
                dataframe0.to_csv('./csv_data/dataframe%d.csv' % (self.opt.nclass))
            else:
                dataframe0.to_csv('./csv_data/dataframe%d.csv'%(self.opt.nclass))

        dataframe = dataframe0.replace([np.inf, -np.inf], np.nan).dropna()
        # labels = dataframe.pop('appname')
        # newlabel = labels.to_frame('appname')
        # propress labels
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)
        # newdata = pd.get_dummies(dataframe)
        if self.opt.enable_all_features == 'no':
            featurelist = []
            time_sf = 0
            if os.path.exists('./log/features_%s'%(self.opt.project_name)):

                for line in open('./log/features_%s'%(self.opt.project_name)):
                    line = line.strip('\n')
                    featurelist.append(line)
            else:

                time_feature_s = time.time()

                featurelist = self._select_features('./csv_data/dataframe%d.csv'%(self.opt.nclass))

                time_feature_e = time.time()
                time_sf += time_feature_e-time_feature_s
                print('time of select features: {} min'.format(time_sf/60))

                f = open('./log/features_%s'%(self.opt.project_name), 'a')
                for i in featurelist:
                    f.write(i)
                    f.write('\n')
                f.close()

            dataframe1 = dataframe[featurelist]
        else:
            dataframe1 = dataframe.drop(columns=['s_fHeaderBytes','s_bHeaderBytes','s_lenSd','s_fLenSd','s_fIATSd',
                                                 's_bLenSd','s_bIATSd','s_flowIATSd','s_idleSd','s_activeSd'
                                                 ])
            featurelist = dataframe1.columns
            self._writefeatures('./log/features_%s'%(self.opt.project_name),featurelist)
            # f = open('./log/features_%s'%(self.opt.project_name), 'a')
            # for i in featurelist:
            #     f.write(i)
            #     f.write('\n')
            # f.close()
            time_sf = 0
        # 特征增强
        if os.path.exists('./log/std.csv') and os.path.exists('./log/mean.csv'):

            pd_std = pd.read_csv('./log/std.csv')
            pd_std.columns = ['key', 'value']
            dict_std = dict(zip(pd_std['key'], pd_std['value']))
            pd_mean = pd.read_csv('./log/mean.csv')
            pd_mean.columns = ['key', 'value']
            dict_mean = dict(zip(pd_mean['key'], pd_mean['value']))
            dataframe1 = self._process_stand(featurelist, dataframe, dict_mean, dict_std)

        else:
            data_std = dataframe1.std()
            pd_data_std = pd.DataFrame(data_std)
            pd_data_std.to_csv('./log/std.csv')
            data_mean = dataframe1.mean()
            pd_data_mean = pd.DataFrame(data_mean)
            pd_data_mean.to_csv('./log/mean.csv')

            for _ in featurelist[:-1]:
                if data_std[_] == 0:
                    dataframe1[_] = 0
                else:
                    dataframe1[_] = dataframe1[_].map(lambda x: (x - data_mean[_]) / data_std[_])
        # print(a)

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

        cnnmodel = autonetworks(self.opt.nclass, self.opt.number_features)
        model = cnnmodel.buildmodels()

        reduce_lr = ReduceLROnPlateau(monitor='loss',factor=0.2,patience=5,min_lr=0.0001)
        earlystop_callback = EarlyStopping(monitor='accuracy', min_delta=0.0001,patience=5)

        time_train_s = time.time()

        model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=self.opt.epochs,
                  callbacks=[reduce_lr])
        loss, accuracy, precision, recall, auc = model.evaluate(test_ds)

        time_train_e = time.time()
        train_time = time_train_e-time_train_s

        print('time of training: {} min'.format(train_time/60))
        time_total = train_time+time_sf
        print('total_time: {} min'.format(time_total/60))
        model.save(self.opt.model_file)
        prediction = model.predict(test_ds, verbose=1)
        predict_label = np.argmax(prediction, axis=1)
        # plot_conf(predict_label, truelabels_te, reskey)
        # print(classification_report(y_test.argmax(-1), y_pred.argmax(-1)))
        return auc

    def _explain_model(self,model_path):

        x = pd.read_csv('./csv_data/KingofGlorymisidentifiedasPeaceElite___20230616161057.csv')

        alist = []
        if os.path.exists('./log/features_%s' % (self.opt.project_name)):

            for line in open('./log/features_%s' % (self.opt.project_name)):
                line = line.strip('\n')
                alist.append(line)
        else:
            raise Exception('dont exist folder!')

        pd_std = pd.read_csv('./log/std.csv')
        pd_std.columns = ['key', 'value']
        dict_std = dict(zip(pd_std['key'], pd_std['value']))
        pd_mean = pd.read_csv('./log/mean.csv')
        pd_mean.columns = ['key', 'value']
        dict_mean = dict(zip(pd_mean['key'], pd_mean['value']))
        x.columns = alist[:-1]
        x = self._process_stand(alist[:-1], x, dict_mean, dict_std)
        mean_ = np.array(pd_mean['value'])
        dataframe = x.copy()
        # labels = dataframe.pop('appname')
        # dataframe.drop(dataframe.columns[[0]], axis=1, inplace=True)
        dataArray = dataframe.values
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)

        # process numic values
        # X = self._process_standard(dataArray)
        # Since the data has been normalized in the edge router, there is no need to do normalization.

        # preprocess
        # dataArray = self._featureBernoulliRBM(dataArray)

        dataArray = np.expand_dims(dataArray.astype(float), axis=2)
        x_train = dataArray.reshape((len(dataArray), int(math.sqrt(self.opt.number_features)), int(math.sqrt(self.opt.number_features)), 1))
        test_model = load_model(model_path)
        explainer = shap.GradientExplainer(test_model, x_train)

        shap_values = explainer.shap_values(x_train[:3],nsamples=3)
        sample1 = DataFrame()
        sample1['values'] = list(shap_values[0][0].reshape(49))
        sample1['featureName'] = alist[:-1]

        sample1['abs_scores'] = abs(sample1['values'])
        sample1 = sample1.sort_values(by=['abs_scores'], ascending=False).drop(labels=['abs_scores'], axis=1)[:9]
        sns.barplot(x = 'featureName',y = 'values',data = sample1)
        plt.show()
        # shap.force_plot(explainer.expected_value, shap_values, x.iloc[299, :])

    # 此处设计在线学习的module
    def process_run(self):
        if (self.opt.isInitialization == 'yes'):
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
                except RuntimeError as e:
                    print(e)
            if not os.path.exists(self.opt.model_file):
                auc = self._obtain_data_train_test()
            else:
                self.test_all(self.opt.model_file, './csv_data/dataframe%d.csv'%(self.opt.nclass))
                # self._explain_model(self.opt.model_file)
                # self.test_new_all('./savedmodels/model_0504_10_49.h5', '5APP_flowfeature_update_int')
                # self.test_new_all('./savedmodels/model_0504_10_49.h5','AP_flowfeature_int_withbackground')
                # self.test_all(self.opt.model_file, './csv_data/dataframe%d.csv' % (self.opt.nclass))
                # self.test_single(self.opt.model_file, './csv_data/yuanshen_v3.5.0_Android_20230329_jkw.csv', '原神', 0, isall=False)

        elif (self.opt.isInitialization == 'no'):

            if self.opt.enable_all_features == 'no':

                if os.path.exists('./log/features_%s'%(self.opt.project_name)):
                    alist = self._readfeatures('./log/features_%s'%(self.opt.project_name))
                else:
                    raise Exception('dont exist folder!')
            else:
                dataframe = pd.read_csv('./csv_data/dataframe%d.csv'%(self.opt.nclass))
                dataframe1 = dataframe.drop(
                    columns=['s_fHeaderBytes', 's_bHeaderBytes', 's_lenSd', 's_fLenSd', 's_fIATSd',
                             's_bLenSd', 's_bIATSd', 's_flowIATSd', 's_idleSd', 's_activeSd'
                             ])
                alist = dataframe1.columns[1:]

            try:
                embediatest = embediaModel(
                    OUTPUT_FOLDER=self.opt.output_folder,
                    PROJECT_NAME=self.opt.project_name,
                    MODEL_FILE=self.opt.model_file,
                    Test_Example='./csv_data/dataframe%d.csv'%(self.opt.nclass),
                    Feature_List=alist,
                    opt = self.opt
                )
                self.opt.apps.append('background')
                embediatest.output_model_c(self.opt.apps)

            except Exception as e:
                print("error:",e.__class__.__name__)
                print(e)
                print("Model embedding fails!")

if __name__ == '__main__':
    dictname = {
        'moba': ['王者荣耀'],
        'shoot': ['香肠派对'],
        'rpg': ['原神'],
        'sport': ['狂野飙车9竞速传奇'],
        'vr': ['picoshipin'],
        'social': ['知乎'],
        'meeting': ['网易会议'],
        'education': ['猿辅导'],
        'video': ['bilibili'],
        'music': ['QQ音乐']
    }
    opt = Options().parse()

    df = pd.read_csv('./csv_data/dataframe7.csv')
    dataframe_pd = df.copy()
    dataframe0 = dataframe_pd.drop(dataframe_pd.columns[0], axis=1)
    dataframe = dataframe0.replace([np.inf, -np.inf], np.nan).dropna()
    labels = dataframe.pop('appname')
    # propress labels
    le = LabelEncoder()
    newlabels = le.fit_transform(labels)

    model = autotask(opt)
    x= model._featurepca(dataframe.values)
    print(x)