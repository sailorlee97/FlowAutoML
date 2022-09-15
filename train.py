"""
@Time    : 2022/6/24 16:09
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: train.py
@Software: PyCharm
"""
from autoTask import autotask
from options import Options
from utils.csvdb import ConnectMysql

if __name__ == '__main__':
    opt = Options().parse()
    model = autotask(opt)
    model.process_run()