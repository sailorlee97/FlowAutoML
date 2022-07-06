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

    def run_server_connect_mysql(self):

        # server = SSHTunnelForwarder(
        #     ssh_address_or_host=(self.opt.ip, self.opt.port),
        #     ssh_username=self.opt.server_username,
        #     ssh_password=self.opt.server_password,
        #     remote_bind_address=('localhost', 3306)
        # )
        # self.server.start()
        # db = pymysql.connect(
        #     host='127.0.0.1',
        #     port=server.local_bind_port,
        #     user=self.opt.database_username,
        #     passwd=self.opt.database_username_password,
        #     db=self.opt.databaseName,
        #     charset='utf8'
        # )
        #        cursor = self.connect.cursor()
        # 使用execute()方法执行SQL查询
        self.cursor.execute("SELECT VERSION()")

        # todo 存 （羊） 写（刘）

        data = self.cursor.fetchone()
        print("Database version : %s " % data)
        self.run()

    def read_csv_columns(self):

        # todo 这里对接王梓炫组，下面代码仅供算法组测试
        # 读取csv文件的列索引，用于建立数据表时的字段
        csv_name = 'data/CICIDS_test.csv'
        data = pd.read_csv(csv_name, encoding="utf-8")
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

        # todo 这里对接王梓炫组，下面代码仅供算法组测试
        csv_name = 'data/CICIDS_test.csv'
        data = pd.read_csv(csv_name)
        print('the shape of data we get.',data.shape)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        print(data.shape)
        data = pd.DataFrame(data)
        data_3 = list(data.values)

        return data_3

    def write_mysql(self):

        # table_name = 't_cicids'
        # columns = self.read_csv_colnmus()
        data = self.read_csv_values()
        for i in data:
            data_6 = tuple(i)
            sql = 'insert into {} values{}'.format(self.opt.tableName, data_6)
            print(sql)
            self.cursor.execute(sql)
            self.commit()
        print("\nComplete write data operation!")

    def commit(self):
        # 定义一个确认事务运行
        self.connect.commit()

    def create(self):
        # 若已有数据表weather_year_db，则删除
        # table_name = 'zx_flow'
        # self.cursor.execute('DROP TABLE IF EXISTS {}'.format(self.opt.tableName))
        # 创建数据表，用刚才提取的列索引作为字段
        columns = self.read_csv_columns()
        text = 'CREATE TABLE {}({})'.format(self.opt.tableName, columns)
        print(text)
        self.cursor.execute('CREATE TABLE {}({})'.format(self.opt.tableName, columns))
        self.commit()

    def exists(self):
        sql = "SHOW TABLES LIKE '{}' ".format(self.opt.tableName)
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        return len(result) != 0

    def run(self):
        if ~self.exists():
            self.create()
        self.write_mysql()