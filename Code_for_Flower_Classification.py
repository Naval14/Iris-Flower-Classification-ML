#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Dataset = pd.read_table('/Users/naval.jaggi/Downloads/Iris.csv' , sep=',') #change the path of Iris.csv dataset file accordingly(according to Iris.csv file location in your computer)

X = np.array(Dataset[['SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm']])

labeling = { 'Iris-setosa': 0, 'Iris-versicolor' : 1,  'Iris-virginica'  : 2}
y  = np.array(Dataset.Species.map(labeling))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

classifier_knn = KNeighborsClassifier(n_neighbors = 3)
classifier_knn.fit(X_train, y_train)

y_prediction = classifier_knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_prediction)
print("Accuracy:",accuracy)

inputs = [[1, 2, 3, 2], [2, 8, 3, 5]]  #you can give your desired input here as a 2-D array.
d = {0:'setosa' , 1:'versicolor' , 2:'virginica'}
species = [ d[p] for p in classifier_knn.predict(inputs)]
print('Iris-species type : ',  species)

