# -*- coding: utf-8 -*-


from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,Convolution1D
from keras import backend as K
from keras import optimizers
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


LABELS=['appname 0','appname 1','appname 2','appname 3','appname 4','appname 5','appname 6','appname 7','appname 8','appname 9','appname 10','appname 11','appname 12','appname 13','appname 14','appname 15','appname 16','appname 17','appname 18','appname 19','appname 20','appname 21','appname 22','appname 23','appname 24','appname 25','appname 26','appname 27','appname 28','appname 29','appname 30','appname 31','appname 32','appname 33','appname 34','appname 35','appname 36','appname 37','appname 38','appname 39','appname 40','appname 41','appname 42','appname 43','appname 44']
#step1  加载数据集
dfDS = pd.read_csv('testing.csv')
X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
Y_full = dfDS["appname"].values
X_full = preprocessing.scale(X_full)
n_classes=len(set(Y_full))

x_train, x_test, y_train, y_test= train_test_split(X_full, Y_full, test_size = 0.1)
# get the dataset
inp_size =x_train.shape[1]

def plot_confusion_matrix(cm, savename,classes, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=500)
    # plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=3, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png', dpi=500)
    plt.show()


def getlabelindex(Y_full,n_classes,labelnum):
    Y_full=pd.DataFrame(Y_full)
    Y_full.columns=['appname']
    idxs_annot=[]
    for idx in range(n_classes):
        labelindex=Y_full.loc[Y_full['appname']==idx].index
        if len(labelindex)<labelnum:
            print("该类标签不足！，当前类共有标签：",len(labelindex.values),"个，但是设置抽取",labelnum,'个！！')
            idxs = np.random.choice(labelindex.values, labelnum)
        else:
            idxs = np.random.choice(labelindex.values, labelnum,replace=False)
        for data in list(idxs):
            idxs_annot.append(data)
    return idxs_annot

idxs_annot=getlabelindex(y_train,n_classes,10000) #每个类挑选有标签的个数

y_train = keras.utils.to_categorical(y_train, n_classes)
y_test  = keras.utils.to_categorical(y_test,  n_classes)

# only select 100 training samples
# idxs_annot = range(x_train.shape[0])
# random.seed(0)
# idxs_annot = np.random.choice(x_train.shape[0], 6000)  #5000 79%
print("idxs_annot",idxs_annot)
print("idxs_annot",len(idxs_annot))
x_train_unlabeled = x_train
x_train_labeled   = x_train[idxs_annot]
y_train_labeled   = y_train[idxs_annot]
print("x_train_labeled",x_train_labeled.shape)
print("y_train_labeled",y_train_labeled.shape)

x_train_labeled = np.expand_dims(x_train_labeled, axis=2)
x_test = np.expand_dims(x_test, axis=2)


model=Sequential()
model.add(Convolution1D(64,3,padding="same",activation="relu",input_shape=(inp_size,1)))
model.add(Convolution1D(64,3,padding="same",activation="relu"))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Convolution1D(128,3,padding="same",activation="relu"))
model.add(Convolution1D(128,3,padding="same",activation="relu"))
model.add(MaxPool1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(n_classes,activation="softmax"))
model.summary()
rmsprop=optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

time_i = time.time()
history=model.fit(x_train_labeled, y_train_labeled, epochs=50,batch_size=512,validation_split= 0.1)
scores = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test, batch_size=100)

model.save('less_model.h5')

time_o = time.time()
end_time = time_o - time_i
cm = confusion_matrix(y_test.argmax(-1), y_pred.argmax(-1))
plot_confusion_matrix(cm, "confusion_matrix.png", LABELS)




print(end_time)
print(classification_report(y_test.argmax(-1), y_pred.argmax(-1), target_names = LABELS))
print("CNN Accuracy: ", scores[1])