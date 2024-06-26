"""
@Time    : 2023/3/20 16:06
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: GenerateTestData.py
@Software: PyCharm
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from embedia.model_generator.project_options import *
from embedia.model_generator.generate_files import generate_examples
from embedia.utils import file_management
from options import Options


class model_c_data():
    '''

    '''

    def __init__(self,data_path,feature_list):
        self.data_path = data_path
        self.feature_list = feature_list

    def _delinf(self,dataframe):

        newdf = dataframe.replace([np.inf, -np.inf], np.nan).dropna()
        return newdf

    def _propress_label(self, labels,sorted_labels):
        """Encode target labels with value between 0 and n_classes-1.

        This transformer should be used to encode target values, *i.e.* `y`, and
        not the input `X`.

        :param data:
        :return: numic
        """
        reskey = {}
        for i in sorted_labels:
            reskey.update({i: sorted_labels.index(i)})
        print(reskey)
        # map映射
        labels = labels.map(reskey).values

        return labels, reskey

    def _obtainData(self,sorted_labels,num):
        '''

        :param sorted_labels: 需要传入的列表
        :param num: 选取你需要生成的数据的数量
        :return:生成的数据， 标签
        '''

        df0 = pd.read_csv(self.data_path)
        df = df0[self.feature_list]
        newdf = self._delinf(df)
        dataframe = newdf.copy()
        labels = dataframe.pop('appname')
        # propress labels
        newlabels, res = self._propress_label(labels, sorted_labels)
        dataArray = dataframe.values
        # mm = MinMaxScaler()
        # X = mm.fit_transform(dataArray)
        X = np.expand_dims(dataArray.astype(float), axis=2)
        lenx = len(X)
        newx = X.reshape((lenx, 7, 7, 1))
        selectx = newx[:num,:,:,:]
        selecty = newlabels[:num]

        return selectx,selecty

    def _data_to_array_str(self,data, macro_converter, clip=120):
        output = ''
        cline = '  '
        for i in data.flatten():
            cline += macro_converter(str(i)) + ', '
            if len(cline) > clip:
                output += cline + '\n'
                cline = '  '
        output += cline
        return output[:-2]

    def _generateData(self,src_folder, var_name, options):

        if options.data_type == ModelDataType.FLOAT or options.data_type == ModelDataType.BINARY:
            def conv(s):
                return s

            data_type = 'float'
        else:
            def conv(s):
                return f"FL2FX({s})"

            data_type = 'fixed'

        smp = options.example_data
        # ids = options.example_ids

        src_h = os.path.join(src_folder, 'main/example_file.h')

        if not isinstance(smp, np.ndarray):
            smp = np.array(smp)

        examples = f'''
static {data_type} {var_name}[][49]= {{
'''
        for i in range(len(smp)):
            data = smp[i].flatten()
            # print(data)
            examples += f'''   {{ {self._data_to_array_str(data, conv)}
}},
'''
        content = file_management.read_from_file(src_h).format(examples=examples)

        return content

    def _generateLabel(self, src_folder, var_name, options):

        ids = options.example_labels
        src_h = os.path.join(src_folder, 'main/example_file.h')

        def conv1(s):
            return s

        labels = f'''
        uint16_t {var_name}_id[] = {{ 
        {self._data_to_array_str(ids,conv1)}
}};
'''
        label = file_management.read_from_file(src_h).format(examples=labels)

        return label

    def generate_data_c(self,sorted_labels):
        '''
        (FLOAT, FIXED32, FIXED16, FIXED8, BINARY) = (0, 1, 2, 3, 4)

        :param sorted_labels:
        :return:
        '''

        x,y = self._obtainData(sorted_labels,50)
        options = ProjectOptions()
        options.example_data = x
        options.data_type = 0
        options.example_labels = y
        content = self._generateData('E:\work_code_program\FlowAutoML\embedia\libraries', 'sample_data',options)
        label = self._generateLabel('E:\work_code_program\FlowAutoML\embedia\libraries', 'sample_data',options)

        return  content,label


# unit test
if __name__ == '__main__':
    featurelist = []
    time_sf = 0
    opt = Options().parse()
    if os.path.exists('../log/features'):

        for line in open("../log/features"):
            line = line.strip('\n')
            featurelist.append(line)
    else:
        raise ('no folder!')

    nl= model_c_data('../csv_data/dataframe%d.csv'%(opt.nclass),featurelist)

    con,label = nl.generate_data_c(opt.apps)
    # print(con)
    file_management.save_to_file(os.path.join('../outputs/data_c', 'five_label_c' + '.h'), label)
    file_management.save_to_file(os.path.join('../outputs/data_c', 'five_class_c' + '.h'), con)