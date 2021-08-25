#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[3]:


import numpy as np


# In[4]:


Dataset = pd.read_table('/Users/naval.jaggi/Downloads/Iris.csv' , sep=',')


# In[5]:


Dataset


# In[6]:


Dataset.shape


# In[7]:


X = np.array(Dataset[['SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm']])
X


# In[8]:


labeling = { 'Iris-setosa': 0, 'Iris-versicolor' : 1,  'Iris-virginica'  : 2}
y  = np.array(Dataset.Species.map(labeling))
y


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)


# In[10]:


print(y_train.shape , y_test.shape)


# In[11]:



classifier_knn = KNeighborsClassifier(n_neighbors = 3)


# In[12]:



classifier_knn.fit(X_train, y_train)


# In[13]:



y_prediction = classifier_knn.predict(X_test)


# In[14]:



accuracy = metrics.accuracy_score(y_test, y_prediction)
print("Accuracy:",accuracy)


# In[15]:


inputs = [[1, 2, 3, 2], [2, 8, 3, 5]]
d = {0:'setosa' , 1:'versicolor' , 2:'virginica'}
species = [ d[p] for p in classifier_knn.predict(inputs)]
print('Iris-species type : ',  species)

