"""
@Time    : 2022/6/23 17:37
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: evaultaion.py
@Software: PyCharm
"""
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, recall_score, precision_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

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