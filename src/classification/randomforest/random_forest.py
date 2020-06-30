# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:34:32 2020

@author: Ambarish Kapil
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:24:51 2020

@author: Ambarish Kapil
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../../../data/Social_Network_Ads.csv')

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

# Wen donot scale values for Decision Trees

# Training model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, criterion="entropy")
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
c_m = confusion_matrix(Y_test, Y_pred)

print(c_m)