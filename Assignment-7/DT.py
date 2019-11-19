#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np


# In[2]:


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)


# In[3]:


y = pd.get_dummies(y)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[5]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[ ]:





# In[8]:


y_pred = dt.predict(X_test)


# In[9]:


species = np.array(y_test).argmax(axis=1)
predictions = np.array(y_pred).argmax(axis=1)
confusion_matrix(species, predictions)


# In[ ]:





# In[ ]:




