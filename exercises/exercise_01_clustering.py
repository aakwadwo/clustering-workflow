"""
Exercise 1: Clustering Challenge
=================================

Difficulty: Intermediate  
Time: 30-45 minutes

SCENARIO:
You work for an e-commerce company. Your task is to segment customers
into groups based on their behavior, without any predefined labels.
This is UNSUPERVISED learning!

OBJECTIVES:
1. Generate and explore clustering data
2. Use the elbow method to find optimal K
3. Compare K-means and DBSCAN
4. Interpret and validate clusters
5. Handle outliers appropriately

LEARNING GOALS:
- Understand unsupervised learning
- Apply elbow method correctly
- Compare clustering algorithms
- Interpret cluster characteristics
- Identify and handle outliers
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import AppConfig
from data_loaders import DatasetManager
from utils import configure_logging


def exercise_3():
    """
    TODO: Complete this function to solve the clustering challenge.
    
    Steps to complete:
    1. Generate clustering data
    2. Visualize the raw data (if 2D)
    3. Use elbow method to determine optimal K
    4. Apply K-means with optimal K
    5. Apply DBSCAN with appropriate parameters
    6. Compare results using silhouette score
    7. Analyze cluster characteristics
    
    Hints:
    - Elbow method: Plot inertia vs K
    - Look for "elbow" in the plot
    - Silhouette score: Ranges from -1 to 1 (higher is better)
    - DBSCAN params: eps and min_samples
    - Try different eps values: 0.5, 1.0, 1.5, 2.0
    """
    
    # Step 1: Setup
    configure_logging()
    dm = DatasetManager()
    
    # TODO: Load clustering data
    # X = ...
    
    print("="*70)
    print("EXERCISE 3: CLUSTERING CHALLENGE")
    print("="*70)
    print()
    
    # TODO: Explore the data
    # print(f"Samples: ...")
    # print(f"Features: ...")
    
    # Step 2: Visualize raw data (if 2D)
    # TODO: Create a scatter plot of the raw data
    # This helps you understand what you're working with
    
    # Step 3: Elbow Method
    # TODO: Calculate inertia for K=1 to K=10
    # Plot the elbow curve
    # Identify the optimal K
    
    # inertias = []
    # K_range = range(1, 11)
    # for k in K_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(X)
    #     inertias.append(kmeans.inertia_)
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(K_range, inertias, 'bo-')
    # plt.xlabel('Number of Clusters (K)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method')
    # plt.show()
    
    # Step 4: K-means clustering
    # TODO: Apply K-means with optimal K
    # TODO: Calculate silhouette score
    # TODO: Visualize clusters with different colors
    # TODO: Plot cluster centers
    
    # Step 5: DBSCAN clustering
    # TODO: Try different eps values (0.5, 1.0, 1.5)
    # TODO: For each eps, count clusters and noise points
    # TODO: Calculate silhouette score (excluding noise)
    # TODO: Visualize best DBSCAN result
    
    # Step 6: Comparison
    # TODO: Compare K-means vs DBSCAN
    # - Number of clusters found
    # - Silhouette scores
    # - Number of outliers (DBSCAN only)
    # - Visual quality of clusters
    
    # Step 7: Cluster analysis
    # TODO: For the best clustering result:
    # - Calculate cluster sizes
    # - Calculate cluster centers (means)
    # - Describe characteristics of each cluster
    
    print("\n" + "="*70)
    print("QUESTIONS TO ANSWER:")
    print("="*70)
    print("1. What is the optimal K from the elbow method?")
    print("2. Which algorithm performed better: K-means or DBSCAN?")
    print("3. How many outliers did DBSCAN identify?")
    print("4. What are the characteristics of each cluster?")
    print("5. When would you use K-means vs DBSCAN?")
    print("="*70)


# QUESTIONS (write your answers here as comments):

# Q1: Optimal K value:
# Answer:

# Q2: Best algorithm and why:
# Answer:

# Q3: Number of outliers found:
# Answer:

# Q4: Cluster characteristics:
# Answer:

# Q5: When to use K-means vs DBSCAN:
# Answer:


if __name__ == "__main__":
    exercise_3()
