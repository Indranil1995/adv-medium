#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pickle


# ### Importing dataset

# In[2]:


data=pd.read_csv('advertising.csv')
data


# In[3]:


data.isnull().sum()


# In[4]:


## So In this data we don't have any missing values & also don't have any categorical values


# ### Declare dependent & independent variables

# In[5]:


X=data.iloc[:,0:3].values
y=data.iloc[:,-1].values


# In[6]:


X.shape


# In[7]:


y.shape


# ### Splinting the data

# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split( X,y, test_size=0.25,random_state=42 )


# ### Train the model

# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ### Test the model

# In[10]:


y_pred= regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[11]:


pickle.dump(regressor, open('model.pkl','wb'))


# In[12]:


model = pickle.load(open('model.pkl','rb'))


# In[13]:


print(model.predict([[200,100,100]]))


# In[ ]:




