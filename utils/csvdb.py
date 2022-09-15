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
from sklearn.preprocessing import LabelEncoder

class ConnectMysql:
    # 金凯威
    conn = pymysql.connect(
        user='runtrend',
        password='4rfv*UHB',
        host='rm-bp1mv8ua26rj84t32eo.mysql.rds.aliyuncs.com',
        database='flowfeature',
        port=3306,
        charset='utf8mb4',
    )
    sql = "select * from test_flowfeature where appname in ('AcFun','aiqiyijisuban','aobidao','aolaxing','Bing','douyin','duoduoshipin','duxiaoshi','elma','fengxingshipin')"
    df = pd.read_sql(sql, con=conn)
    print(df)
    df.to_csv('test.csv', index=False)
    print('Read from sqlserver table successfully!')

    data = pd.read_csv("test.csv", header=0)

    chuli = data.drop(
        ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Label", "appversion", "appplatform", "date",
         "chargeperson"], axis=1, inplace=False)

    le = LabelEncoder()
    chuli.iloc[0:, -1] = le.fit_transform(chuli.iloc[0:, -1])

    print(chuli)
    chuli.to_csv('testing.csv')
    print('dl successfully')