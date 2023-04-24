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

        # training argument
        self.parser.add_argument('--isInitialization', default='no',
                                 help='yes - model train;no - we will load model trained.')
        self.parser.add_argument('--label', default='appname', help='data class labels.')
        self.parser.add_argument('--epochs', default=40, help='number of epochs')
        self.parser.add_argument('--nclass', default=11, help='number class labels.')
        self.parser.add_argument('--batch_size', default=64, help='batch_size.')
        self.parser.add_argument('--embedd_data_type', default='FLOAT', help='FLOAT, FIXED32, FIXED16, and FIXED8.')
        self.parser.add_argument('--enable_all_features', default='no',
                                 help='If all features are enabled, the feature selection method is not required.')

        # Model curing parameter
        self.parser.add_argument('--output_folder', default='outputs/', help='Path of output.')
        self.parser.add_argument('--number_features', default=49, help='number_features name.')
        self.parser.add_argument('--project_name', default='model_0422_11_49', help='project name.')
        self.parser.add_argument('--model_file', default='savedmodels/model_0422_11_49.h5', help='Path of model.')
        self.parser.add_argument('--apps',
                                 default=['王者荣耀','和平精英','战双帕弥什','原神','网易会议',
                                          '猿辅导','抖音','bilibili','爱奇艺','picoshipin', 'QQ音乐'],
                                 help='Select the application of the classification. The order apps here is the order of the labels.')

        # database
        self.parser.add_argument('--port', default=26618, help='connect server port.')
        self.parser.add_argument('--appsql',
                                 default=('王者荣耀','和平精英','战双帕弥什','原神','网易会议',
                                          '猿辅导','抖音','bilibili','爱奇艺','picoshipin', 'QQ音乐'),
                                 help='Applications selected from the database.')
        self.parser.add_argument('--host', default='sh-cynosdbmysql-grp-5dmxbh9a.sql.tencentcdb.com',
                                 help='server ip addr.')
        self.parser.add_argument('--username', default='runtrend', help='the name that server log in.')
        self.parser.add_argument('--password', default='1qaz@WSX', help='the password that server log in.')
        self.parser.add_argument('--databaseName', default='flowfeature', help='the name that database log in.')
        self.parser.add_argument('--tableName', default='AP_flowfeature_int', help='the name of table.')
        self.parser.add_argument('--charset', default='utf8mb4', help='the name of word.')
        self.opt = None

    def parse(self) -> object:
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        return self.opt