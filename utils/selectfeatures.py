# -*- coding: utf-8 -*-
"""
@Time    : 2023/2/8 11:06
@Author  : zeyi li
@Site    : 
@File    : selectfeatures.py
@Software: PyCharm
"""
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


class selectfeature():
  def __init__(self):
    self.name = "feature"

  def treemodel(self,dataframe,target_train):

    clf = RandomForestClassifier()
    clf.fit(dataframe.values, target_train)
    features = dataframe.columns
    feature_importances = clf.feature_importances_
    features_df = pd.DataFrame({'Features': features, 'Importance': feature_importances})
    features_df.sort_values('Importance', inplace=True, ascending=False)

    return features_df

  def perlist(self,dataframe):
    """

    :param dataframe:
    :return:
    """
    lst = dataframe.columns
    newlst = lst.tolist()
    newlist = list(itertools.combinations(newlst, 2))

    return newlist

  def caluate_spear(self,featurelist, dataframe):
    df1 = dataframe[featurelist[0]].tolist()
    df2 = dataframe[featurelist[1]].tolist()

    return spearmanr(df1, df2)[0]

  def speardict(self,dataframe):

    featurelst = self.perlist(dataframe)
    ic = {}
    for i in featurelst:
      spdata = self.caluate_spear(i, dataframe)
      if not math.isnan(spdata):
        if math.fabs(spdata) > 0.7:
          ic[i] = spdata

    return ic

  def correlation_heatmap(self,train):
    correlations = train.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()

  def search_corrlate_features(self,dataframe,newlabels,num):

    all_features = []

    feautures = self.treemodel(dataframe,newlabels)
    sfdict = self.speardict(dataframe)
    b = list(sfdict.keys())
    print('corr:', b)
    a = feautures[:num]['Features']
    print('start searching features!')
    for i in a:
      for j in b:
        all_features.append(i)
        if i in j:
          all_features.append(j[0])
          all_features.append(j[1])
    l2 = list(set(all_features))

    return l2

# united test
if __name__ == '__main__':
  df = pd.read_csv('../csv_data/dataframe10self.csv')
  dataframe = df.copy()

  # dataframe.drop(dataframe.columns[[0]], axis=1, inplace=True)
  data = dataframe.replace([np.inf, -np.inf], np.nan).dropna()
  labels = data.pop('appname')
  # propress labels
  le = LabelEncoder()
  newlabels = le.fit_transform(labels)
  # newlabels = ._propress_label(labels)

  sf = selectfeature()
  secorndfeatrues = sf.search_corrlate_features(data,newlabels)
  newdataframe = data[secorndfeatrues]
  feauturesimportance = sf.treemodel(newdataframe, newlabels)
  a = list(feauturesimportance[:36]['Features'])
  f = open('../log/features', 'a')
  for i in a:
    f.write(i)
    f.write('\n')
  f.close()

  # sf.correlation_heatmap(dataframe)
  # feautures = sf.treemodel(dataframe,newlabels)
  # allfeatures = []
  # sfdict = sf.speardict(dataframe)
  # b = list(sfdict.keys())
  # print('corr:',b)
  # a = feautures[:25]['Features']
  # print('feature importance:',a)
  # for i in a :
  #   for j in b:
  #     if i in j:
  #       allfeatures.append(j[0])
  #       allfeatures.append(j[1])
  # l2 = list(set(allfeatures))
  # print(l2)