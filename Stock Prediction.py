#!/usr/bin/env python
# coding: utf-8

# In[2]:


import yfinance as yf 
import numpy as np 


# In[64]:


data = yf.download('AAPL', start = "2010-01-01")


# In[65]:


print(data.head())


# In[66]:


data['returns'] = np.log(data.Close.pct_change() + 1)


# In[67]:


print(data)


# In[68]:


#shifts data
def lagit(data, lags):
    names = []
    for i in range(1, lags+1):
        data['Lag_'+str(i)] = data['returns'].shift(i)
        names.append('Lag_'+str(i))
    return names 


# In[69]:


lagnames = lagit(data, 3)


# In[70]:


data


# In[71]:


data.dropna(inplace=True)


# In[72]:


data


# In[73]:


from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(data[lagnames], data['returns'])


# In[74]:


data['prediction'] = model.predict(data[lagnames])


# In[75]:


data


# In[76]:


#makes it easier to determine if we should buy or not 
data['direction_LR'] = [1 if i > 0 else -1 for i in data.prediction]


# In[77]:


data['strat'] = data['direction_LR'] * data['returns']


# In[78]:


data


# In[79]:


np.exp(data[['returns', 'strat']].cumsum()).plot()


# In[119]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, shuffle = False, test_size = 0.3, random_state = 0)


# In[120]:


train


# In[121]:


test


# In[122]:


model = LinearRegression()


# In[123]:


model.fit(train[lagnames], train['returns'])


# In[124]:


test['predictions'] = model.predict(test[lagnames])


# In[125]:


test


# In[126]:


test['direction_LR'] = [1 if i > 0 else -1 for i in test.predictions]


# In[127]:


test['strat'] = test['direction_LR'] * test['returns']


# In[128]:


np.exp(test[['returns', 'strat']].sum())


# In[129]:


np.exp(test[['returns', 'strat']].cumsum()).plot()


# In[130]:


test


# In[118]:


#checking how many trades it takes to get us the value we wanted
(test['direction_LR'].diff() != 0).value_counts()


# In[ ]:





# In[ ]:




