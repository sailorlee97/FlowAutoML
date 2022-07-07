"""
@Time    : 2022/7/4 16:09
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: csvdb.py
@Software: PyCharm
"""
from sshtunnel import SSHTunnelForwarder
import pymysql
import pandas as pd
import numpy as np


class ConnectMysql:

    def __init__(self, opt):
        self.opt = opt
        self.server = SSHTunnelForwarder(
            ssh_address_or_host=(self.opt.ip, self.opt.port),
            ssh_username=self.opt.server_username,
            ssh_password=self.opt.server_password,
            remote_bind_address=('localhost', 3306)
        )
        self.server.start()
        self.connect = pymysql.connect(
            host='127.0.0.1',
            port=self.server.local_bind_port,
            user=self.opt.database_username,
            passwd=self.opt.database_username_password,
            db=self.opt.databaseName,
            charset='utf8'
        )
        self.cursor = self.connect.cursor()

    def __del__(self):
        self.cursor.close()
        self.connect.close()
        self.server.close()
        print('close success!')

    def read_csv_columns(self):

        # todo 这里对接王梓炫组，读取数据的代码仅供算法组测试，
        #  这里面需要传递的一个dataframe格式的数据

        csv_name = 'data/CICIDS_test.csv'
        data = pd.read_csv(csv_name, encoding="utf-8")

        # 这里不变

        columns = data.columns.tolist()
        types = data.dtypes
        make_table = []
        for item in columns:
            if 'int' in str(types[item]):
                item = str(item)
                item = item.replace(" ", "_")
                item = item.replace("/", "_")
                item = item.replace(".", "_")
                char = item + ' INT'
            elif 'float' in str(types[item]):
                item = str(item)
                item = item.replace(" ", "_")
                item = item.replace("/", "_")
                item = item.replace(".", "_")
                char = item + ' FLOAT'
            elif 'object' in str(types[item]):
                item = str(item)
                item = item.replace(" ", "_")
                item = item.replace("/", "_")
                item = item.replace(".", "_")
                char = item + ' VARCHAR(255)'
            elif 'datetime' in str(types[item]):
                item = str(item)
                item = item.replace(" ", "_")
                item = item.replace("/", "_")
                item = item.replace(".", "_")
                char = item + ' DATETIME'
            make_table.append(char)
        return ','.join(make_table)

    def read_csv_values(self):

        # todo 这里对接王梓炫组，读取数据的代码仅供算法组测试，
        #  这里面需要传递的一个dataframe格式的数据

        csv_name = 'data/CICIDS_test.csv'
        data = pd.read_csv(csv_name)

        #这里不需要变化
        print('the shape of data we get.',data.shape)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        print(data.shape)
        data = pd.DataFrame(data)
        data_3 = list(data.values)

        return data_3

    def write_mysql(self):

        data = self.read_csv_values()
        for i in data:
            data_6 = tuple(i)
            sql = 'insert into {} values{}'.format(self.opt.tableName, data_6)
            #print(sql)
            self.cursor.execute(sql)
            self.commit()
        print("\nComplete write data operation!")

    def commit(self):
        # 定义一个确认事务运行
        self.connect.commit()

    def create(self):
        columns = self.read_csv_columns()
        text = 'CREATE TABLE {}({})'.format(self.opt.tableName, columns)
        #print(text)
        self.cursor.execute('CREATE TABLE {}({})'.format(self.opt.tableName, columns))
        self.commit()

    def exists(self):
        sql = "SHOW TABLES LIKE '{}' ".format(self.opt.tableName)
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        return len(result) != 0

    def start_write_csv(self):
        '''
        yang write
        :return:none
        '''

        if ~self.exists():
            self.create()
        self.write_mysql()

    def read_data_from_mysql(self):
        '''
        get data from mysql
        :return:
        '''

        sql = 'SELECT * FROM {}'.format(self.opt.tableName)
        self.cursor.execute(sql)
        #mydata = self.cursor.fetchall()  # 获取全部数据
        count = 0
        data = self.cursor.fetchall()
        df = pd.DataFrame(list(data))
        print(df)