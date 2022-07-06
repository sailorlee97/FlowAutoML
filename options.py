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

        self.parser.add_argument('--modelpath', default='AutogluonModels/ag-20220622_121736', help='path to model')
        self.parser.add_argument('--isInitialization',default='no',help='yes - model train;no - we will load model trained.')
        self.parser.add_argument('--label',default='label',help='data class labels.')
        self.parser.add_argument('--ip', default='122.112.155.228', help='server ip addr.')
        self.parser.add_argument('--port', default=22, help='connect server port.')
        self.parser.add_argument('--server_username', default='root', help='the name that server log in.')
        self.parser.add_argument('--server_password', default='19970407@lizeyi', help='the password that server log in.')
        self.parser.add_argument('--database_username', default='root', help='the name that database log in.')
        self.parser.add_argument('--database_username_password', default='123456', help='the password that database '
                                                                                        'log in.')
        self.parser.add_argument('--databaseName', default='zxyflow', help='the name that database log in.')
        self.parser.add_argument('--tableName', default='zx_flow', help='the name of table.')
        self.opt = None

    def parse(self) -> object:
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        return self.opt