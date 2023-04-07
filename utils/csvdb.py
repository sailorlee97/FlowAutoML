"""
@Time    : 2022/7/4 16:09
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: csvdb.py
@Software: PyCharm
"""
import pymysql
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ConnectMysql():

    # 金凯威
    def __init__(self):
        self.conn = pymysql.connect(
            user='runtrend',
            password='4rfv*UHB',
            host='sh-cynosdbmysql-grp-5dmxbh9a.sql.tencentcdb.com',
            database='flowfeature',
            port=26618,
            charset='utf8mb4',
        )
        self.issavetocsv = False

    def get_data(self,app=["AcFun", "aiqiyijisuban", "aobidao"], limitnum=5000, feature="*"):

        print('Start read data! Please wait a memont.')
        df = pd.DataFrame()
        for i in app:
            sql = "select {} from AP_flowfeature where appname = {}" \
                  " order by `Active Min` limit {}".format(feature, "'{}'".format(i), limitnum)
            appfile = pd.read_sql(sql, con=self.conn)
            df = df.append(appfile)
        # sql = "select * from test_flowfeature where appname in ('AcFun','Bing','duxiaoshi')"
        # df = pd.read_sql(sql, con=self.conn)
        #print(df)
        #df.to_csv('test.csv', index=False)
        print('Read from sqlserver table successfully!')

        #data = pd.read_csv("test.csv", header=0)

        process = df.drop(
            ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Label", "appversion", "appplatform", "date","index",
             "chargeperson"], axis=1, inplace=False)

        le = LabelEncoder()
        #process.iloc[0:, -1] = le.fit_transform(process.iloc[0:, -1])

        #print(process)
        if self.issavetocsv:
            process.to_csv('testing.csv')
        print('dl successfully')

        y = process['appname']
        x = process.drop('appname',axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

        y_train = le.fit_transform(y_train)
        #print(y_train)
        y_test = le.fit_transform(y_test)

        return X_train, X_test, y_train, y_test

    def total_get_data(self,app=("哈利波特魔法觉醒", "狂野飙车9竞速传奇", "欢乐麻将","狼人杀","使命召唤手游","QQ飞车",
                                 "QQ炫舞手游","优酷视频","芒果TV","央视影音","QQ音乐","虎牙直播","爱奇艺","抖音","王者荣耀","决战平安京","重返深渊","猿辅导"),
                    featurebase='AP_flowfeature'):
        '''
        select app flows and return flows

        :param app:
        :param featurebase:
        :return: dataframe
        '''
        sql = 'select * from {} WHERE appname in {}'.format(featurebase,app)
        print(sql)
        df = pd.read_sql(sql, con=self.conn)
        print('Read from sqlserver table successfully!')

        # data = pd.read_csv("test.csv", header=0)

        process = df.drop(
            ['index','appversion','appplatform','date','chargeperson',
             's_lenVar','s_fLenVar','s_bLenVar','s_flowIATVar','s_fIATVar','s_bIATVar','s_activeVar','s_idleVar'],
             axis=1,
            inplace=False)

        return process

# unit test
if __name__ == '__main__':
    getdata = ConnectMysql()
    process = getdata.total_get_data()