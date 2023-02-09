"""
@Time    : 2022/6/23 14:47
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: options.py
@Software: PyCharm
"""
import argparse
import os


class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--modelpath', default='AutomlModels/', help='path to model')
        self.parser.add_argument('--isInitialization', default='yes',
                                 help='yes - model train;no - we will load model trained.')
        self.parser.add_argument('--label', default='appname', help='data class labels.')
        self.parser.add_argument('--epochs', default=40, help='number of epochs')
        self.parser.add_argument('--nclass', default=3, help='number class labels.')
        self.parser.add_argument('--host', default='sh-cynosdbmysql-grp-5dmxbh9a.sql.tencentcdb.com',
                                 help='server ip addr.')
        self.parser.add_argument('--batch_size', default=64, help='batch_size.')
        self.parser.add_argument('--port', default=26618, help='connect server port.')
        self.parser.add_argument('--username', default='runtrend', help='the name that server log in.')
        self.parser.add_argument('--password', default='1qaz@WSX', help='the password that server log in.')
        self.parser.add_argument('--databaseName', default='flowfeature', help='the name that database log in.')
        self.parser.add_argument('--tableName', default='test_flowfeature', help='the name of table.')
        self.parser.add_argument('--charset', default='utf8mb4', help='the name of word.')
        self.opt = None

    def parse(self) -> object:
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        return self.opt