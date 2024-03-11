# Eric Jackman - DSC411 - Assignment 6

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Import dataset
glass_data = pd.read_csv("glass.csv", names=["id", "refractive_index", "sodium", "magnesium", "aluminum", "silicon",
											 "potassium", "calcium", "barium", "iron", "class"])

# Get X values
data = glass_data[["refractive_index", "sodium", "magnesium", "aluminum", "silicon",
				   "potassium", "calcium", "barium", "iron"]]
X = data.values

# Create knn model
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X)

# Get distances
distances, indexes = knn.kneighbors(X)

# Plot distance for each instance
plt.plot(distances.mean(axis=1))
plt.title("Average Distance for Each Instance")
plt.xlabel("Index")
plt.ylabel("Average Distance")
plt.show()

# Identify anomalies by comparing distances to the threshold
outlier_index = np.where(distances.mean(axis=1) > 2.20)
outlier_values = glass_data.iloc[outlier_index]

# Output anomalies to console
print("Anomalies detected:")
print(outlier_values)

# Add anomaly column to dataframe
glass_data["anomaly"] = "false"
for i in outlier_index:
	glass_data.iloc[i, 11] = "true"

# Write to csv
glass_data.to_csv("glass_anomaly.csv", index=False)
