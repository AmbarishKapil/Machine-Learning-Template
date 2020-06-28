# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:10:01 2020

@author: Ambarish Kapil
"""
from math import ceil
import numpy as np
import pandas as pd

df = pd.read_csv("../../../data/Social_Network_Ads.csv")

X = df.iloc[:,[1,2,3]]
Y = df.iloc[:, -1]

# Encoding categorical values
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

# Train-Test-Split
X_train = X.sample(frac=0.8, random_state=1)
X_test = X.drop(X_train.index)
Y_test = Y.drop(X_train.index)
Y_train = Y.drop(Y_test.index)

X_train = X_train.sort_index()


# Scaling values
from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)

# Making a Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
c_m = confusion_matrix(Y_test, Y_pred)

print(c_m)
