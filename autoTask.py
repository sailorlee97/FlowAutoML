"""
@Time    : 2022/6/23 14:42
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: autoTask.py
@Software: PyCharm
"""
from sklearn.preprocessing import LabelEncoder

from options import Options
from utils.evaultaion import Eva
from autogluon.tabular import TabularDataset, TabularPredictor
import schedule, functools, time

class autotask():

    def __init__(self,opt):
        self.opt = opt

    def get_traindata(self):

        # test
        df = TabularDataset('new_nslkdd.csv')
        df = df.drop(labels=['num'], axis=1)

        # todo 后期：写入从数据库获取训练数据的逻辑 （刘沣汉）

        return df

    def get_testdata(self):

        df = TabularDataset('nsk-kdd/KDDTest+.csv')
        class_le = LabelEncoder()
        y = class_le.fit_transform(df['label'].values)

        df = df.drop(labels=['label'], axis=1)
        df = df.drop(labels=['number'], axis=1)
        test_data = df.drop(labels=['num'], axis=1)
        # todo 后期：写入从数据库获取测试数据的逻辑 (金凯威)

        return test_data,y

    def run_task(self,task, freq=1, time_unit='minute'):
        '''timed task'''
        if time_unit == 'second':
            schedule.every(freq).seconds.do(task)
        elif time_unit == 'minute':
            schedule.every(freq).minutes.do(task)
        elif time_unit == 'hour':
            schedule.every(freq).hour.do(task)
        elif time_unit == 'day':
            schedule.every(freq).day.at("4:30").do(task)

        while True:
            schedule.run_pending()
            time.sleep(1)


    def run_every(self,freq=1, time_unit='minute'):

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                try:
                    self.run_task(func, freq, time_unit)
                except:
                    pass

            return wrapper

        return decorator

#判断

    def train_data(self):
        # 重新抓取新的数据集
        df = self.get_traindata()
        # 送入训练
        predictor = TabularPredictor(label=self.opt.label).fit(train_data=df)

        return predictor

    def test_model(self,label):
        if self.opt.modelpath is not None:
            predictor = TabularPredictor.load(path=self.opt.modelpath)

            # 获取测试数据
            test_data,y = self.get_testdata()
            predictions = predictor.predict(test_data)
            pred = predictions.values
            return pred,y
        else:
            raise

#    @run_every(30,'day')
    def process_run(self):
        if(self.opt.isInitialization=='yes'):

            predictions = self.train_data()

            # todo  设计获取参数进行下发

        elif(self.opt.isInitialization=='no'):

            pred,y= self.test_model(self.opt.label)
            eva = Eva('recall')
            rec = eva.calculate_recall(y,pred)
            precision = eva.calculate_precision(y, pred)

            print(rec)
            if (rec<0.96 and precision<0.99):
                predictions = self.train_data()
            else:
                print("No need to retrain, continue to use the current model.")
        else:
            raise