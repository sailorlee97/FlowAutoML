from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import  SelectKBest,f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
df = pd.read_csv('testing.csv')
X = df.iloc[:, 1:len(df.columns)].values
Y = df["appname"].values
etc = ExtraTreesClassifier()
etc1 = etc.fit(X,Y)

print(etc.feature_importances_)
model = SelectFromModel(etc1,prefit=True)

#X_SFM_ETC = model.transform(X)
#print(X_SFM_ETC.shape)

