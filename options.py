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
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        return self.opt