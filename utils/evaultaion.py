"""
@Time    : 2022/6/23 17:37
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: evaultaion.py
@Software: PyCharm
"""
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, recall_score, precision_score, \
    classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

class Eva():

    def __init__(self,parm):
        self.parm = parm

    def evaluate(self,labels, scores):
        if self.parm == 'roc':
            return self.roc(labels, scores)
        elif self.parm == 'auprc':
            return self.auprc(labels, scores)
        elif self.parm == 'f1_score':
            threshold = 0.20
            scores[scores >= threshold] = 1
            scores[scores < threshold] = 0
            return f1_score(labels, scores)
        elif self.parm == 'recall':
            return self.calculate_recall(labels, scores)
        elif self.parm == 'precision':
            return self.calculate_precision(labels, scores)
        else:
            raise NotImplementedError("Check the evaluation metric.")

    def calculate_recall(self,label,pred):
        recall = recall_score(label,pred, average='weighted')
        print("recall_score", recall)
        return recall

    def calculate_precision(self,label,pred):
        precision = precision_score(label,pred, average='weighted')
        return precision

    def roc(self,labels, scores, saveto=None):
        """Compute ROC curve and ROC area for each class"""
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        labels = labels.cpu()
        scores = scores.cpu()

        # True/False Positive Rates.
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Equal Error Rate
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        if saveto:
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
            plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
            plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(saveto, "ROC.pdf"))
            plt.close()

        return roc_auc

    def auprc(self,labels, scores):
        ap = average_precision_score(labels, scores)
        return ap

    def plot_confusion_matrix(cm, savename, classes, title='Confusion Matrix'):
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

    def plot_learning_curves(self,history):

        pd.DataFrame(history.history).plot(figsize = (8,5))
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.show()

    def caluateclassification(self,y_test,y_pred):
        print(classification_report(y_test,y_pred,digits=4))