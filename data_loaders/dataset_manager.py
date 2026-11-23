"""
Dataset Manager Module
======================

Purpose:
    Loads and summarizes example datasets from scikit-learn.
    Provides a consistent interface for accessing different types of data.

Key Concepts:
    - Scikit-learn datasets: Built-in datasets for learning and testing
    - NumPy arrays: Efficient numeric containers for features
    - Data validation: Always check data before returning it

Available Datasets:
    1. Blobs (Clustering): Synthetic data for learning clustering algorithms

Why Use Built-in Datasets?
    - No download needed
    - Well-documented
    - Standard benchmarks
    - Perfect for learning
    
    Later, you'll use this same pattern for your own datasets!
"""

import logging

import numpy as np
from sklearn.datasets import make_blobs

from utils.data_validator import DataValidator


class DatasetManager:
    """
    Loads and summarizes example datasets for machine learning.
    
    This class provides a unified interface to load different types of datasets.
    Each method returns validated data ready for machine learning.
    
    Methods:
        - load_clustering_data(): Synthetic data with known clusters
    """
    
    def __init__(self) -> None:
        """Initialize the dataset manager with logging."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_clustering_data(self, random_state: int = 42) -> np.ndarray:
        """
        Generate synthetic data for clustering demonstrations.
        
        This creates artificial data with known cluster structure.
        Perfect for learning and testing clustering algorithms.
        
        Data Characteristics:
            - Samples: 300 points
            - Features: 2 (for easy visualization)
            - True clusters: 3 (but algorithm doesn't know this!)
        
        Args:
            random_state: Random seed for reproducibility (default: 42)
        
        Returns:
            X (ndarray): Features (300 rows × 2 columns)
        
        Example:
            >>> dm = DatasetManager()
            >>> X = dm.load_clustering_data()
            >>> print(f"Shape: {X.shape}")
            Shape: (300, 2)
        
        Note:
            We don't return labels because clustering is unsupervised!
            The algorithm must discover the clusters on its own.
            (But make_blobs creates labels internally for our visualization)
        
        Why Synthetic Data?
            - We know the "true" clusters (for evaluation)
            - Can control difficulty (overlap, number of clusters, etc.)
            - Perfect for learning and testing
        """
        X, _ = make_blobs(
            n_samples=300,
            centers=3,          # Create 3 clusters
            random_state=random_state,
            cluster_std=1.0,    # Standard deviation (controls spread)
        )
        
        # Validate before returning (catches NaN, inf, wrong shape)
        DataValidator.validate_features(X)
        
        self.logger.info(
            "Clustering dataset generated: %s observations, %s features",
            X.shape[0],
            X.shape[1],
        )
        
        return X


# Example usage (for testing this module independently)
if __name__ == "__main__":
    from utils.logger import configure_logging
    
    # Set up logging
    configure_logging()
    
    print("=== Dataset Manager Examples ===\n")
    
    # Create a dataset manager
    dm = DatasetManager()
    
    # Load clustering data
    print("1. Loading Clustering Dataset (Synthetic):")
    X_cluster = dm.load_clustering_data()
    print(f"   Features shape: {X_cluster.shape}")
    print(f"   Feature 1 range: {X_cluster[:, 0].min():.2f} to {X_cluster[:, 0].max():.2f}")
    print(f"   Feature 2 range: {X_cluster[:, 1].min():.2f} to {X_cluster[:, 1].max():.2f}\n")
    
    print("Dataset loaded successfully!")