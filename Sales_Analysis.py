#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


#Importing data
df = pd.read_csv("us-retail-sales.csv")
#Displaying dataset
df.head()


# In[13]:


df.isna().sum()


# In[14]:


df["Clothing"]


# In[15]:


df = df.dropna()


# In[16]:


plt.plot(df['Month'], df['Clothing'])
plt.xlabel('Month')
plt.ylabel('Clothing')
plt.title("Scatterplot of Clothing sale")
plt.show()


# In[30]:


# split dataset for test and training
temp_2 = df[["Clothing"]]

from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

train, test = train_test_split(temp_2, test_size=0.2)


# train autoregression
model = AutoReg(train, lags=40)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)


# In[31]:


# Predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
predictions


# In[32]:


predictions.iloc[1]


# In[33]:


for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions.iloc[i], test.values[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# plot results
pyplot.plot(test.values)
pyplot.plot(predictions.values, color='red')

pyplot.legend(['Expected', 'Predicted'])
pyplot.show()


# In[35]:


#implement moving average

# MA over a period of 10 and 20 months
temp_2['MA_10'] = temp_2.Clothing.rolling(10, min_periods=1).mean()
temp_2['MA_20'] = temp_2.Clothing.rolling(20, min_periods=1).mean()


# In[36]:


temp_2


# In[39]:


# Grean = Avg Air Temp, RED = 10 yrs, ORANG colors for the line plot
colors = ['green', 'red', 'orange']
# Line plot 
temp_2.plot(y=["Clothing","MA_10","MA_20"], color=colors, linewidth=3, figsize=(12,6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(labels =['Clothing', '10-years MA', '20-years MA'], fontsize=14)

plt.title('The Clothing level prediction', fontsize=20)
plt.xlabel('Periods', fontsize=16)
plt.ylabel('Sales', fontsize=16)


# In[41]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
expected = temp_2['Clothing']
predicted = temp_2['MA_10']
print(f'Mean Absolute Error 10 = {mean_absolute_error(expected, predicted)}')
print(f'Mean Squared Error 20= {mean_squared_error(expected, predicted)}')
predicted = temp_2['MA_20']
print(f'Mean Absolute Error 10= {mean_absolute_error(expected, predicted)}')
print(f'Mean Squared Error 20= {mean_squared_error(expected, predicted)}')


# In[42]:


#weighted moving average with a 10 interval
import random
weights = []
for i in range(10):
    n = random.uniform(0,1)
    weights.append(n)

weights


# In[43]:


temp_2['WMA_10'] = temp_2.Clothing.rolling(10).apply(lambda x: np.sum(weights*x))


# In[49]:


#Exponential Smoothing Moving Average
#EMA is mainly used to identify trends and to filter out noise. 
#The weight of elements is decreased gradually over time. 
#This means It gives weight to recent data points, not historical ones.

# EMA Temperature
# Let's smoothing factor - 0.1
temp_2['EMA_0.1'] = temp_2.Clothing.ewm(alpha=0.1, adjust=False).mean()
# Let's smoothing factor  - 0.3
temp_2['EMA_0.3'] = temp_2.Clothing.ewm(alpha=0.3, adjust=False).mean()


# In[50]:


temp_2


# In[51]:


# green - Avg Air Temp, red- smoothing factor - 0.1, yellow - smoothing factor  - 0.3
colors = ['green', 'red', 'yellow', 'purple']
temp_2.plot(y=['Clothing', 'WMA_10', 'EMA_0.1', 'EMA_0.3'],color=colors, linewidth=3, figsize=(12,6), alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=['Clothing', 'WMA_10', 'EMA - alpha=0.1', 'EMA - alpha=0.3'], fontsize=14)
plt.title('The A1c level prediction', fontsize=20)
plt.xlabel('Periods', fontsize=16)
plt.ylabel('A1c_level', fontsize=16)


# In[52]:


# green - Avg Air Temp, red- smoothing factor - 0.1, yellow - smoothing factor  - 0.3
colors = ['green', 'yellow', 'purple']
temp_2.plot(y=['Clothing','EMA_0.1', 'EMA_0.3'],color=colors, linewidth=3, figsize=(12,6), alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=['Clothing level', 'EMA - alpha=0.1', 'EMA - alpha=0.3'], fontsize=14)
plt.title('The Clothing level prediction', fontsize=20)
plt.xlabel('Periods', fontsize=16)
plt.ylabel('A1c_level', fontsize=16)


# In[54]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
expected = temp_2['Clothing']
predicted = temp_2['EMA_0.1']
print(f'Mean Absolute Error 0.1 = {mean_absolute_error(expected, predicted)}')
print(f'Mean Squared Error 0.1= {mean_squared_error(expected, predicted)}')
predicted = temp_2['EMA_0.3']
print(f'Mean Absolute Error 0.3= {mean_absolute_error(expected, predicted)}')
print(f'Mean Squared Error 0.3= {mean_squared_error(expected, predicted)}')


# In[55]:


#ARIMA

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(temp_2['Clothing'], order=(0, 1, 1)) 
results_ARIMA = model.fit()


# In[56]:


results_ARIMA.summary()


# In[57]:


results = results_ARIMA.predict(start=0)
results


# In[58]:


# plot results
pyplot.plot(temp_2['Clothing'])
pyplot.plot(results, color='red')

pyplot.legend(['Expected', 'Predicted'])
pyplot.show()


# In[59]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
expected = temp_2['Clothing']
print(f'Mean Absolute Error = {mean_absolute_error(expected, results)}')
print(f'Mean Squared Error = {mean_squared_error(expected, results)}')


# In[ ]:




