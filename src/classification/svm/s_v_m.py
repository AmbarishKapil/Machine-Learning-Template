# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:23:38 2020

@author: Ambarish Kapil
"""

def kernel_dict(k):
    from sklearn.svm import SVC
    clf = SVC(kernel = k)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
    
    kernel_accuracy[k] = ((cm[0][0] + cm[1][1])/(len(X_test))) * 100
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, 2:-1].values
Y = df.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)

kernel_accuracy = {}

kernel_dict('linear')
kernel_dict('rbf')
kernel_dict('poly')
kernel_dict('sigmoid')


