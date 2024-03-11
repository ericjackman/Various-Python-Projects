import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing data
df = pd.read_csv("DSC496_2022_Spring_Reddit_Fidelity.csv")

#df = df[0:2000]

split_date = df.created_utc.str.split('-', expand=True)
split_date.columns =["year","month","day"]

# Group by month and find mean score
df['month'] = split_date.year.str[2:] + '/' + split_date['month']
perMonth = df.groupby(['month']).agg({'score': ['mean']})
df['perMonth'] = perMonth
maxScore = df.groupby(['month']).agg({'score': ['max']})
print(perMonth.iloc(1))

df['MA_6'] = df.score.rolling(6, min_periods=1).mean()
df['MA_12'] = df.score.rolling(12, min_periods=1).mean()
print(df[['score', 'MA_6', 'MA_12']])

colors = ['cyan', 'red', 'orange']
# Line plot
df.plot(x='month',y=["perMonth","MA_6","MA_12"], color=colors, linewidth=3, figsize=(12,6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(labels =['Average score', '6-month MA', '12-month MA'], fontsize=14)

plt.title('The monthly average score', fontsize=20)
plt.xlabel('Month', fontsize=16)
plt.ylabel('Temperature [°C]', fontsize=16)
#plt.show()