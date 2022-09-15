"""
@Time    : 2022/6/23 14:42
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: autoTask.py
@Software: PyCharm
"""
from options import Options
from utils.evaultaion import Eva
from autogluon.tabular import TabularDataset, TabularPredictor
import schedule, functools, time
import pandas as pd

class autotask():

    def __init__(self,opt):
        self.opt = opt

    def get_traindata(self):

        # test
        ratio = 0.8
        data = TabularDataset(r'D:\uniti3\FlowAutoML-main\utils\testing.csv')
        # data = data.drop(labels=['num'], axis=1)

        # todo 后期：写入从数据库获取训练数据的逻辑 （刘沣汉）
        t0 = data[data["appname"] == 0]
        t1 = data[data["appname"] == 1]
        t2 = data[data["appname"] == 2]
        t3 = data[data["appname"] == 3]
        t4 = data[data["appname"] == 4]
        t5 = data[data["appname"] == 5]
        t6 = data[data["appname"] == 6]
        t7 = data[data["appname"] == 7]
        t8 = data[data["appname"] == 8]
        t9 = data[data["appname"] == 9]

        t0 = t0.sample(len(t0), random_state=0)
        t1 = t1.sample(len(t1), random_state=0)
        t2 = t2.sample(len(t2), random_state=0)
        t3 = t3.sample(len(t3), random_state=0)
        t4 = t4.sample(len(t4), random_state=0)
        t5 = t5.sample(len(t5), random_state=0)
        t6 = t6.sample(len(t6), random_state=0)
        t7 = t7.sample(len(t7), random_state=0)
        t8 = t8.sample(len(t8), random_state=0)
        t9 = t9.sample(len(t9), random_state=0)

        training = pd.concat(
            [t0.iloc[:int(len(t0) * ratio), :], t1.iloc[:int(len(t1) * ratio), :], t2.iloc[:int(len(t2) * ratio), :],
             t3.iloc[:int(len(t3) * ratio), :], t4.iloc[:int(len(t4) * ratio), :],
             t5.iloc[:int(len(t5) * ratio), :], t6.iloc[:int(len(t6) * ratio), :], t7.iloc[:int(len(t7) * ratio), :],
             t8.iloc[:int(len(t8) * ratio), :], t9.iloc[:int(len(t9) * ratio), :]
             ], axis=0)

        training.to_csv('training1.csv')
        return training

    def get_testdata(self):
        ratio = 0.8
        data = TabularDataset(r'D:\uniti3\FlowAutoML-main\utils\testing.csv')
        # data = data.drop(labels=['num'], axis=1)
        # todo 后期：写入从数据库获取测试数据的逻辑 (刘沣汉)
        t0 = data[data["appname"] == 0]
        t1 = data[data["appname"] == 1]
        t2 = data[data["appname"] == 2]
        t3 = data[data["appname"] == 3]
        t4 = data[data["appname"] == 4]
        t5 = data[data["appname"] == 5]
        t6 = data[data["appname"] == 6]
        t7 = data[data["appname"] == 7]
        t8 = data[data["appname"] == 8]
        t9 = data[data["appname"] == 9]

        t0 = t0.sample(len(t0), random_state=0)
        t1 = t1.sample(len(t1), random_state=0)
        t2 = t2.sample(len(t2), random_state=0)
        t3 = t3.sample(len(t3), random_state=0)
        t4 = t4.sample(len(t4), random_state=0)
        t5 = t5.sample(len(t5), random_state=0)
        t6 = t6.sample(len(t6), random_state=0)
        t7 = t7.sample(len(t7), random_state=0)
        t8 = t8.sample(len(t8), random_state=0)
        t9 = t9.sample(len(t9), random_state=0)

        testing = pd.concat(
            [t0.iloc[:int(len(t0) * ratio), :], t1.iloc[:int(len(t1) * ratio), :], t2.iloc[:int(len(t2) * ratio), :],
             t3.iloc[:int(len(t3) * ratio), :], t4.iloc[:int(len(t4) * ratio), :],
             t5.iloc[:int(len(t5) * ratio), :], t6.iloc[:int(len(t6) * ratio), :], t7.iloc[:int(len(t7) * ratio), :],
             t8.iloc[:int(len(t8) * ratio), :], t9.iloc[:int(len(t9) * ratio), :]
             ], axis=0)

        testing.to_csv('testing1.csv')
        return testing

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