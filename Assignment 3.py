#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import numpy as np
import sklearn
import matplotlib.patches as mpatches
import seaborn as sns


# In[2]:


df = pd.read_csv('StudentsPerformance.csv', low_memory=False)


# In[3]:


df.isnull().values.any()


# In[4]:


df.head() 


# In[5]:


df.info()


# In[6]:


df['Total'] = df['math score'] + df['reading score'] + df['writing score']


# In[7]:


df.head()


# In[8]:


df['gender_dummies'] = df.gender.map({'female':0, 'male':1})


# In[9]:


df['lunch_dummies'] = df.lunch.map({'standard':0, 'free/reduced':1})


# In[10]:


df['test_prep_dummies'] = df['test preparation course'].map({'none':0, 'completed':1})


# In[11]:


df['POED'] = df['parental level of education'].map({'some college':0, "associate's degree":1,'high school':2,'some high school':3,"bachelor's degree":4,"master's degree":5})


# In[12]:


df.sample(5)


# In[124]:


#L1 Penalty
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X = df.drop(['gender','race/ethnicity','parental level of education','lunch','test preparation course', 'Total','test_prep_dummies'], axis = 1) 
y = df['test_prep_dummies']
X_train , X_test ,y_train , y_test =train_test_split(X, y, test_size=0.4, random_state=101)
log_1 = LogisticRegression(penalty='l1', solver='liblinear',dual = False, max_iter = 400000)
log_1.fit(X_train, y_train)
cross_val_score(log_1, X_train, y_train, cv = 20, scoring = 'accuracy').mean()


# In[125]:


#L2 penalty
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X = df.drop(['gender','race/ethnicity','parental level of education','lunch','test preparation course', 'Total','test_prep_dummies'], axis = 1) 
y = df['test_prep_dummies']
X_train , X_test ,y_train , y_test =train_test_split(X, y, test_size=0.4, random_state=101)
log_2 = LogisticRegression(penalty='l2', solver='saga',dual = False, max_iter = 400000)
log_2.fit(X_train, y_train)
cross_val_score(log_2, X_train, y_train, cv = 20, scoring = 'accuracy').mean()


# In[126]:


#no penalty(regularization)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X = df.drop(['gender','race/ethnicity','parental level of education','lunch','test preparation course', 'Total','test_prep_dummies'], axis = 1) 
y = df['test_prep_dummies']
X_train , X_test ,y_train , y_test =train_test_split(X, y, test_size=0.4, random_state=101)
log_3 = LogisticRegression(penalty='none',dual = False, max_iter = 400000)
log_3.fit(X_train, y_train)
cross_val_score(log_3, X_train, y_train, cv = 20, scoring = 'accuracy').mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[102]:


from sklearn.model_selection import train_test_split


# In[134]:


X = df.drop(['gender','race/ethnicity','parental level of education','lunch','test preparation course', 'Total','test_prep_dummies'], axis = 1) 
y = df['test_prep_dummies']
X_train , X_test ,y_train , y_test =train_test_split(X, y, test_size=0.4, random_state=101)


# In[135]:


log = LogisticRegression(penalty='l2' ,solver = 'saga',dual = False, max_iter = 4000)


# In[136]:


log.fit(X_train, y_train)


# In[137]:


log.score(X_test, y_test)


# In[130]:


#Classification Report
from sklearn.metrics import classification_report, confusion_matrix
predicted = log.predict(X_test)
classification_report(y_test, predicted)
confusion_matrix(y_test, predicted)


# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = log
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))


# In[23]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[24]:


predicted = model.predict_proba(X_test)
fpr, tpr, thresh = roc_curve(y_test, predicted[:,1], pos_label=1)
random_probs=[]
for i in range(len(y_test)):
    random_probs.append(i)
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[ ]:





# In[25]:


df_mini = pd.DataFrame({'x':fpr, 'y':tpr})
x=df_mini['x']
y=df_mini['y']


# In[26]:


plt.style.use('seaborn')
fig, ax = plt.subplots()
plt.plot(x, y)
plt.plot(p_fpr, p_tpr, marker = '_')

plt.title('AUC-ROC Plot')
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.gca().legend(('Logistic Regression','No Skill Line'))

plt.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[160]:





# In[183]:





# In[ ]:





# In[ ]:





# In[ ]:




