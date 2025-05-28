# -*- coding: utf-8 -*-
""" Customer Segmentation Using KMeans Clustering


Original file is located at
    https://colab.research.google.com/drive/1xBWaujFKPGPokKDm7b-_yr66BhN7R73O

This project focuses on segmenting mall customers using the KMeans Clustering algorithm to help businesses better understand customer behavior and develop targeted marketing strategies. By analyzing key features such as age, annual income, and spending score, the model identifies distinct customer groups, enabling data-driven decision-making.

The segmentation allows businesses to:

Target high-value customers
Personalize marketing campaigns
Improve customer retention
Identify potential growth opportunities
Through data visualization and clustering, this project provides insights into how customers differ based on their spending habits and income levels.
"""

# KMeans Clustering on Mall Customers Dataset
# ----------------------------------------------------
# This code segments mall customers based on their spending habits using KMeans Clustering.
# Steps:
# 1. Import Libraries
# 2. Load and Explore the Dataset
# 3. Data Preprocessing
# 4. Find Optimal Clusters (Elbow Method)
# 5. Apply KMeans Clustering
# 6. Visualize the Clusters

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 2. Load and Explore the Dataset

df = pd.read_csv("/content/Mall_Customers.csv")
df.head()

# Display first few rows to understand the data
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values and data types
df.info()

# 3. Data Preprocessing
# For clustering, we use 'Annual Income (k$)' and 'Spending Score (1-100)'
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Find Optimal Number of Clusters using Elbow Method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(12, 7))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

"""**From above plot 5 is optimal cluster value **"""

# 5. Apply KMeans Clustering
optimal_clusters = 5  # Based on the elbow method result
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to the original dataframe
df['Cluster'] = y_kmeans

# 6. Visualize the Clusters
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for cluster in range(optimal_clusters):
    plt.scatter(
        X[y_kmeans == cluster]["Annual Income (k$)"],
        X[y_kmeans == cluster]["Spending Score (1-100)"],
        s=100, c=colors[cluster], label=f'Cluster {cluster + 1}'
    )

# Plot the cluster centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300, c='yellow', label='Centroids', edgecolors='black'
)

plt.title('Customer Segments based on Income and Spending')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

"""This project successfully demonstrates how KMeans Clustering can be used to segment customers based on their annual income and spending score. By identifying distinct customer groups, businesses can develop targeted marketing strategies, improve customer satisfaction, and increase profitability. The visualizations provide clear insights into different spending behaviors, enabling data-driven decisions for customer engagement and resource allocation."""
