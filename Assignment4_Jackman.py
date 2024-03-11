import random
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Create dataframe from csv file
df = pd.read_csv("us-retail-sales.csv")
df = df.dropna()

# Preview automobile sales by month
plt.plot(df['Month'], df['Automobiles'])
plt.xlabel('Month')
plt.ylabel('Automobiles')
plt.title("Automobiles Sales by Month")
plt.show()

# Create testing and training sets
temp_2 = df[["Automobiles"]]
train, test = train_test_split(temp_2["Automobiles"], test_size=0.2)

# Train auto-regression model
model = AutoReg(train, lags=40)
model_fit = model.fit()
print('Auto regression coefficients:\n')
print('Coefficients: %s' % model_fit.params)

# Make predictions using auto-regression model
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Print predictions next to expected values
print('\n\nAuto regression predictions vs expected:\n')
for i in range(len(predictions)):
    print('Predicted = %f, Expected = %f' % (predictions.iloc[i], test.values[i]))

# Print the root mean squared error
rmse = sqrt(mean_squared_error(test, predictions))
print('\n\nAuto regression error:\n')
print('Test RMSE: %.3f' % rmse)

# Plot predicted and expected values
plt.plot(test.values)
plt.plot(predictions.values, color='red')
plt.legend(['Expected', 'Predicted'])
plt.title('Auto Regression Model')
plt.show()

# Moving Average over a period of 10 and 20 months
temp_2['MA_10'] = temp_2.Automobiles.rolling(10, min_periods=1).mean()
temp_2['MA_20'] = temp_2.Automobiles.rolling(20, min_periods=1).mean()

# Plot the moving average for 10 and 20 years
colors = ['cyan', 'red', 'orange']
temp_2.plot(y=["Automobiles", "MA_10", "MA_20"], color=colors, linewidth=3, figsize=(12, 6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=['Automobiles', '10-years MA', '20-years MA'], fontsize=14)
plt.xlabel('Periods', fontsize=16)
plt.ylabel('Sales', fontsize=16)
plt.title('Moving Average', fontsize=20)
plt.show()

# Print mean absolute error and mean squared error for moving average periods
expected = temp_2['Automobiles']
predicted = temp_2['MA_10']
print('\n\nMoving average error:\n')
print(f'Mean absolute error 10 = {mean_absolute_error(expected, predicted)}')
print(f'Mean squared error 20= {mean_squared_error(expected, predicted)}')
predicted = temp_2['MA_20']
print(f'Mean absolute error 10= {mean_absolute_error(expected, predicted)}')
print(f'Mean squared error 20= {mean_squared_error(expected, predicted)}')

# Weighted moving average with a 10 interval
weights = []
for i in range(10):
    n = random.uniform(0, 1)
    weights.append(n)
temp_2['WMA_10'] = temp_2.Automobiles.rolling(10).apply(lambda x: np.sum(weights * x))

# Exponential smoothing moving average
temp_2['EMA_0.1'] = temp_2.Automobiles.ewm(alpha=0.1, adjust=False).mean()
temp_2['EMA_0.3'] = temp_2.Automobiles.ewm(alpha=0.3, adjust=False).mean()

# Plot exponential smoothing moving average
colors = ['cyan', 'red', 'orange']
temp_2.plot(y=['Automobiles', 'EMA_0.1', 'EMA_0.3'], color=colors, linewidth=3, figsize=(12, 6), alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=['Automobiles', 'EMA - 0.1', 'EMA - 0.3'], fontsize=14)
plt.title('Exponential Smoothing', fontsize=20)
plt.xlabel('Periods', fontsize=16)
plt.ylabel('A1c_level', fontsize=16)
plt.show()

# Plot exponential smoothing moving average with weighted moving average
colors = ['cyan', 'red', 'orange', 'blue']
temp_2.plot(y=['Automobiles', 'EMA_0.1', 'EMA_0.3', 'WMA_10'], color=colors, linewidth=3, figsize=(12, 6), alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=['Automobiles', 'EMA - 0.1', 'EMA - 0.3', 'WMA - 10'], fontsize=14)
plt.title('Exponential Smoothing with Weighted Moving Average', fontsize=20)
plt.xlabel('Periods', fontsize=16)
plt.ylabel('A1c_level', fontsize=16)
plt.show()

# Print mean absolute error and mean squared error for exponential smoothing moving averages
predicted = temp_2['EMA_0.1']
print('\n\nExponential smoothing error:\n')
print(f'Mean absolute error 0.1 = {mean_absolute_error(expected, predicted)}')
print(f'Mean squared error 0.1= {mean_squared_error(expected, predicted)}')
predicted = temp_2['EMA_0.3']
print(f'Mean absolute error 0.3= {mean_absolute_error(expected, predicted)}')
print(f'Mean squared error 0.3= {mean_squared_error(expected, predicted)}')

# Create ARIMA model
model = ARIMA(temp_2['Automobiles'], order=(0, 1, 1))
results_ARIMA = model.fit()
results_ARIMA.summary()

# Make predictions using ARIMA model
results = results_ARIMA.predict(start=0)

# Plot ARIMA model
plt.plot(temp_2['Automobiles'])
plt.plot(results, color='red')
plt.legend(['Expected', 'Predicted'])
plt.title('ARIMA Model')
plt.show()

# Print mean absolute error and mean squared error for ARIMA
print('\n\nARIMA error:\n')
print(f'Mean Absolute Error = {mean_absolute_error(expected, results)}')
print(f'Mean Squared Error = {mean_squared_error(expected, results)}')
