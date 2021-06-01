#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from penguin import Processing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


# In[2]:


df = Processing('Churn_rate_train.csv')


# In[3]:


X, y = df.Preprocessing(colname = ['international_plan', 'voice_mail_plan', 'churn'], value = 'yes', target = 'churn')


# In[4]:


pipe = make_pipeline(StandardScaler(), 
                     LogisticRegression())


# In[5]:


p_grid = {'logisticregression__C': [1, 10, 100],
    'logisticregression__tol': [.01, .1]
    }

gs = GridSearchCV(pipe, param_grid = p_grid, cv = 5, scoring = "roc_auc")

gs.fit(X, y)
print(gs.best_estimator_)
print(gs.score(X, y))

