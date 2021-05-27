#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from penguin import Processing


# In[2]:


df = Processing('Churn_rate_train.csv')


# In[3]:


X, y = df.Preprocessing(colname = ['churn','international_plan', 'voice_mail_plan'], value = 'yes', target = 'churn')


# In[ ]:





# In[ ]:




