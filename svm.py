#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


print(os.listdir())


# In[3]:


# importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#get graphs inline

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# load up the dataFrame
dataSet = pd.read_csv('Social_Network_Ads.csv')


# In[5]:


dataSet.head()


# In[6]:


# split the dataset
X = dataSet.iloc[:,2:4].values
y = dataSet.iloc[:,4].values


# In[7]:


X 


# In[8]:


y


# In[9]:


# splitting into training and test dataSet
from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[11]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[12]:


sc = StandardScaler()


# In[13]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[14]:


# implement classifier
from sklearn.svm import SVC


# In[15]:


classifier = SVC(kernel='linear')


# In[16]:


classifier.fit(X_train, y_train.ravel())


# In[17]:


# just predict the y based upon the testing values of X
y_predictor = classifier.predict(X_test)


# In[18]:


y_predictor


# In[19]:


from sklearn.metrics import confusion_matrix


# In[20]:


errors = confusion_matrix(y_test, y_predictor)


# In[21]:


errors


# In[ ]:




