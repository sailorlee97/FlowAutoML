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
        # self.ip = self.opt.ip
        # self.port = opt.port
        # self.server_username = opt.server_username
        # self.server_password = opt.server_password
        # self.database_username = opt.database_username
        # self.database_username_password = opt.database_username_password
        # self.databaseName = opt.databaseName

    def run_server_connect_mysql(self):

        server = SSHTunnelForwarder(
            ssh_address_or_host=(self.opt.ip, self.opt.port),
            ssh_username=self.opt.server_username,
            ssh_password=self.opt.server_password,
            remote_bind_address=('localhost', 3306)
        )

        server.start()

        db = pymysql.connect(
            host='127.0.0.1',
            port=server.local_bind_port,
            user=self.opt.database_username,
            passwd=self.opt.database_username_password,
            db=self.opt.databaseName,
            charset='utf8'
        )

        cursor = db.cursor()

        # 使用execute()方法执行SQL查询
        cursor.execute("SELECT VERSION()")
        # 使用 fetchone() 方法获取单条数据.4sHFKgg_456]6
        # todo 存 （羊） 写（刘）

        data = cursor.fetchone()
        print("Database version : %s " % data)

        cursor.close()
        # 关闭数据库连接
        db.close()
        server.close()

    def read_csv_colnmus(self):
        # 读取csv文件的列索引，用于建立数据表时的字段
        csv_name = 'E:/pycharm/pycharmProject/autogluon-master/autogluon-master/datasets/CICIDS2017/CIC-IDS2017/CICIDS_test.csv'
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
        # 读取csv文件数据
        csv_name = 'E:/pycharm/pycharmProject/autogluon-master/autogluon-master/datasets/CICIDS2017/CIC-IDS2017/CICIDS_test.csv'
        data = pd.read_csv(csv_name)
        print(data.shape)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        print(data.shape)
        data = pd.DataFrame(data)
        data_3 = list(data.values)
        # print(data)
        return data, data_3

    def write_csv(self):
        for i in self.read_csv_values():  # 因为数据是迭代列表，所以用循环把数据提取出来
            data_6 = tuple(i)
            sql = """insert into weather_year_db values{}""".format(data_6)
            self.cursor.execute(sql)
            self.commit()
        print("\n数据植入完成")

    def write_mysql(self):
        # 在数据表中写入数据，因为数据是列表类型，把他转化为元组更符合sql语句
        table_name = 't_cicids'
        columns = self.read_csv_colnmus()
        data, data_3 = self.read_csv_values()
        # print(data_3)
        # s = ','.join(['%s' for _ in range(len(data.columns))])
        # text = 'INSERT INTO {} ({}) VALUES ({})'.format(table_name, columns, s)
        # print(text)
        # self.cursor.executemany(text, data_3)
        # self.commit()
        for i in data_3:  # 因为数据是迭代列表，所以用循环把数据提取出来
            data_6 = tuple(i)
            # print(len(data_6))
            # print(len(data.columns))
            sql = 'insert into {} values{}'.format(table_name, data_6)
            print(sql)
            self.cursor.execute(sql)
            self.commit()
        print("\n数据植入完成")

    def commit(self):
        # 定义一个确认事务运行
        self.connect.commit()

    def create(self):
        # 若已有数据表weather_year_db，则删除
        table_name = 'zx_flow'
        self.cursor.execute('DROP TABLE IF EXISTS {}'.format(table_name))
        # 创建数据表，用刚才提取的列索引作为字段
        columns = self.read_csv_colnmus()
        text = 'CREATE TABLE {}({})'.format(table_name, columns)
        print(text)
        self.cursor.execute('CREATE TABLE {}({})'.format(table_name, columns))
        self.commit()

    def run(self):
        self.create()
        self.write_mysql()
