import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

# Import data to dataframe
df = pd.read_csv("DSC496_2022_Spring_Reddit_Fidelity.csv")

# Clean date column (YYYY-MM)
split_date = df.created_utc.str.split('-', expand=True)
split_date.columns = ["year", "month", "day"]

monthlyDF = df[['created_utc', 'score']]
monthlyDF['month'] = split_date['year'] + '-' + split_date['month']
print(monthlyDF)

#df['month'] = pd.to_datetime(df['created_utc']).dt.to_period('M')

#
# # Find total score for each month
# monthlyDF = monthlyDF.groupby(by=['month'], as_index=False).sum()
# monthlyDF.set_index('month', inplace=True)
#
# # Create testing and training sets
# train, test = train_test_split(df["score"], test_size=0.2)
#
# # Train auto-regression model
# model = AutoReg(train, lags=500)
# model_fit = model.fit()
#
# # Make predictions using auto-regression model
# predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
#
# # Calculate 6 and 12 month rolling averages
# monthlyDF['6_month_MA'] = monthlyDF.score.rolling(6, min_periods=1).mean()
# monthlyDF['12_month_MA'] = monthlyDF.score.rolling(12, min_periods=1).mean()
#
# # Exponential smoothing moving average
# monthlyDF['EMA_0.1'] = monthlyDF.score.ewm(alpha=0.1, adjust=False).mean()
# monthlyDF['EMA_0.3'] = monthlyDF.score.ewm(alpha=0.3, adjust=False).mean()
#
# # Create ARIMA model
# model = ARIMA(df['score'], order=(0, 1, 1))
# results_ARIMA = model.fit()
#
# # Make predictions using ARIMA model
# results = results_ARIMA.predict(start=0)
#
# # Plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=monthlyDF['score'],
#                              mode='lines+markers',
#                              name='Score Per Month'))
# fig.add_trace(go.Scatter(x=monthlyDF.index.to_timestamp(), y=results,
#                              mode='lines+markers',
#                              name='ARIMA'))
# fig.show()
