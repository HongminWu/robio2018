from sklearn import preprocessing
import pandas as pd
import numpy as np
import sys
import ipdb

x = pd.read_csv('X.csv')
y = pd.read_csv('y.csv', header=None,index_col=None, squeeze=True,usecols=[1])
y = y.values
le = preprocessing.LabelEncoder()
y  = le.fit_transform(y)
y  = pd.Series(y)
x.insert(loc=0, column='label', value=y)
x = x.drop(['id'], axis=1)
x.to_csv('00Anomalies_TRAIN',columns=None, header=False, index=False)

#---restore data format from expand dataset
try:
    x_exp = pd.read_csv('00Anomalies_EXP_TRAIN', names=x.columns)
except:
    print ('sorry, can not find the file named as 00Anomalies_EXP_TRAIN')
    sys.exit()
y_exp = x_exp['label']
y_val = le.inverse_transform(y_exp.values)
y_exp = pd.Series(y_val)
x_exp = x_exp.drop(['label'], axis=1)
x_exp.to_csv('X_EXP.csv')
y_exp.to_csv('Y_EXP.csv')
