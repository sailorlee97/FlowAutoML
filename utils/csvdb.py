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
            host='rm-bp1mv8ua26rj84t32eo.mysql.rds.aliyuncs.com',
            database='flowfeature',
            port=3306,
            charset='utf8mb4',
        )
        self.issavetocsv = False

    def get_data(self):

        sql = "select * from test_flowfeature where appname in ('AcFun','Bing','duxiaoshi')"

        df = pd.read_sql(sql, con=self.conn)
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
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        y_train = le.fit_transform(y_train)
        #print(y_train)
        y_test = le.fit_transform(y_test)

        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    getdata = ConnectMysql()
    getdata.get_data()