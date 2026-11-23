"""
Clustering Workflow Module
===========================

Purpose:
    Discover natural groupings in unlabeled data.
    Clustering is UNSUPERVISED learning - no target labels!

Key Difference from Classification:
    Classification: Given labels, predict label for new data
    Clustering: NO labels, find groups that naturally exist

Real-World Applications:
    - Customer segmentation (group similar customers)
    - Document organization (group similar articles)
    - Image compression (group similar colors)
    - Anomaly detection (find outliers)
    - Gene sequence analysis (group similar genes)

Algorithms Covered:
    1. K-Means: Partition data into K spherical clusters
    2. DBSCAN: Find clusters of arbitrary shape, detect outliers
    3. Gaussian Mixture Model (GMM): Soft, probability-based clusters

Key Concepts:
    - Elbow Method: Finding optimal number of clusters
    - Inertia: Within-cluster sum of squares (lower is better)
    - Density-based clustering: Groups based on dense regions
    - Noise points: Outliers that don't belong to any cluster
    - Soft assignment: Each point gets a probability for every cluster
    - BIC: Bayesian Information Criterion (lower is better)
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

from config.app_config import AppConfig
from utils.plot_manager import PlotManager


class ClusteringWorkflow:
    """
    Perform K-means, DBSCAN and Gaussian Mixture clustering with visualizations.

    This class implements unsupervised learning workflows:
    - Elbow method for determining optimal K
    - K-means clustering with visualization
    - DBSCAN for density-based clustering
    - Gaussian Mixture Model for probability-based clustering
    
    Attributes:
        config (AppConfig): Application configuration
        logger (Logger): For tracking progress
        plots (PlotManager): For handling visualizations
    """
    
    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the clustering workflow.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plots = PlotManager(config)
    
    def plot_elbow(
        self,
        X: np.ndarray,
        max_clusters: int = 7,
    ) -> pd.DataFrame:
        """
        Create elbow plot to help determine optimal number of clusters.
        
        The Elbow Method:
            1. Try different values of K (number of clusters)
            2. For each K, calculate inertia (within-cluster sum of squares)
            3. Plot K vs inertia
            4. Look for "elbow" - point where improvement diminishes
        
        Inertia (Within-Cluster Sum of Squares):
            - Sum of squared distances from each point to its cluster center
            - Lower = tighter, more compact clusters
            - Always decreases as K increases
            - At K=n_samples, inertia=0 (each point is its own cluster!)
        
        Why "Elbow"?
            The plot looks like an arm:
            - High K: Adding clusters doesn't help much (forearm)
            - Sweet spot: Elbow joint (optimal K)
            - Low K: Adding clusters helps a lot (upper arm)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            max_clusters: Maximum K to try (default: 7)
        
        Returns:
            DataFrame with K values and corresponding inertias
        
        Example Output:
            K=1: Inertia=5000 (all points in one cluster)
            K=2: Inertia=2500 (big improvement!)
            K=3: Inertia=1800 (good improvement)
            K=4: Inertia=1650 (small improvement) ← Elbow here!
            K=5: Inertia=1600 (tiny improvement)
        
        Interpretation:
            - Steep drop: Significant improvement
            - Gentle slope: Diminishing returns
            - Choose K at the elbow (before diminishing returns)
        
        Raises:
            ValueError: If max_clusters < 2
        """
        if max_clusters < 2:
            raise ValueError(
                f"max_clusters must be at least 2, got {max_clusters}"
            )
        
        inertias = []
        
        # Try each value of K from 1 to max_clusters
        for k in range(1, max_clusters + 1):
            # Create and fit K-means model
            model = KMeans(
                n_clusters=k,
                n_init=10,              # Run 10 times with different initializations
                random_state=self.config.random_state,
            )
            model.fit(X)
            
            # Store inertia for this K
            inertias.append(model.inertia_)
            
            self.logger.info(f"K={k}: Inertia={model.inertia_:.2f}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            "Number of Clusters": range(1, max_clusters + 1),
            "Inertia": inertias,
        })
        
        # Create elbow plot
        plt.figure(figsize=(9, 6))
        plt.plot(
            results["Number of Clusters"],
            results["Inertia"],
            'bo-',
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("Number of Clusters (K)", fontsize=12)
        plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
        plt.title("K-means Elbow Method: Finding Optimal K", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, max_clusters + 1))
        
        # Add annotation
        plt.text(
            0.95, 0.95,
            "Look for the 'elbow'\n(point of diminishing returns)",   
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        self.plots.finish("clustering_elbow.png")
        
        return results
    
    def run_kmeans(
        self,
        X: np.ndarray,
        n_clusters: int = 3,
    ) -> tuple[KMeans, np.ndarray]: 
        """
        Perform K-means clustering and visualize results.
        
        K-Means Algorithm:
            1. Randomly initialize K cluster centers
            2. Assign each point to nearest center
            3. Update centers to mean of assigned points
            4. Repeat steps 2-3 until convergence
        
        Properties:
            - Fast and simple
            - Finds spherical/circular clusters
            - Must specify K in advance
            - Sensitive to initialization (run multiple times)
            - Assumes clusters are roughly equal size
        
        When to Use K-Means:
            ✓ Clusters are roughly spherical
            ✓ Clusters are similar in size
            ✓ You know (or can estimate) K
            ✗ Clusters have complex shapes
            ✗ Outliers are present
            ✗ Clusters have very different sizes
        
        Args:
            X: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters (K)
        
        Returns:
            model: Fitted KMeans object
            labels: Cluster assignments (n_samples,)
        
        Cluster Labels:
            Array of integers (0 to K-1) indicating cluster membership
            Example: [0, 0, 1, 2, 1, 0, 2]
            - First 2 points in cluster 0
            - Third and fifth points in cluster 1
            - Fourth and seventh points in cluster 2
        
        Raises:
            ValueError: If n_clusters is invalid
        """
        if n_clusters < 1 or n_clusters > len(X):
            raise ValueError(
                f"n_clusters must be between 1 and {len(X)}, got {n_clusters}"
            )
        
        # Create and fit K-means model
        model = KMeans(
            n_clusters=n_clusters,
            n_init=10,              # Try 10 different initializations
            random_state=self.config.random_state,
        )
        
        # Fit and get cluster labels
        labels = model.fit_predict(X)
        
        # Visualize clusters (only works well for 2D data)
        if X.shape[1] == 2:
            plt.figure(figsize=(9, 7))
            
            # Plot data points colored by cluster
            scatter = plt.scatter(
                X[:, 0],
                X[:, 1],
                c=labels,
                cmap='viridis',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5,
            )
            
            # Plot cluster centers
            plt.scatter(
                model.cluster_centers_[:, 0],
                model.cluster_centers_[:, 1],
                marker='X',
                s=300,
                c='red',
                edgecolors='black',
                linewidths=2,
                label='Cluster Centers',
            )
            
            plt.xlabel("Feature 1", fontsize=12)
            plt.ylabel("Feature 2", fontsize=12)
            plt.title(f"K-means Clustering (K={n_clusters})", fontsize=14, fontweight='bold')
            plt.colorbar(scatter, label='Cluster')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            self.plots.finish("clustering_kmeans.png")
        
        self.logger.info(f"K-means completed with {n_clusters} clusters")
        self.logger.info(f"Cluster sizes: {np.bincount(labels)}")
        
        return model, labels
    
    def run_dbscan(
        self,
        X: np.ndarray,
        eps: float = 1.0,
        min_samples: int = 5,
    ) -> tuple[DBSCAN, np.ndarray]:
        """
        Perform DBSCAN (Density-Based Spatial Clustering) and visualize.
        
        DBSCAN Algorithm:
            1. For each point, find neighbors within distance eps
            2. Points with >= min_samples neighbors are "core points"
            3. Core points and their neighbors form clusters
            4. Points not in any cluster are "noise" (outliers)
        
        Key Parameters:
            eps (epsilon): Maximum distance between neighbors
            - Too small: Many small clusters and noise
            - Too large: Everything in one cluster
            - Rule of thumb: Look at distance to k-th nearest neighbor
            
            min_samples: Minimum points to form a dense region
            - Too small: Too many clusters
            - Too large: Too much noise
            - Rule of thumb: Start with 5, or 2*dimensions
        
        Advantages over K-Means:
            ✓ Finds clusters of arbitrary shape
            ✓ Automatically determines number of clusters
            ✓ Identifies outliers (noise points)
            ✓ No need to specify K
        
        Disadvantages:
            ✗ Sensitive to eps and min_samples
            ✗ Struggles with varying density clusters
            ✗ Not deterministic (order matters)
        
        When to Use DBSCAN:
            ✓ Clusters have complex shapes
            ✓ Need outlier detection
            ✓ Don't know number of clusters
            ✗ Clusters have very different densities
            ✗ High-dimensional data (distance becomes less meaningful)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            eps: Maximum distance between neighbors (default: 1.0)
            min_samples: Minimum points to form cluster (default: 5)
        
        Returns:
            model: Fitted DBSCAN object
            labels: Cluster assignments (n_samples,)
                   -1 indicates noise (outlier)
        
        Cluster Labels:
            Array of integers (starting from 0) and -1 for noise
            Example: [0, 0, -1, 1, 1, 0, -1]
            - Points 0, 1, 5 in cluster 0
            - Points 3, 4 in cluster 1
            - Points 2, 6 are noise (outliers)
        
        Raises:
            ValueError: If parameters are invalid
        """
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        
        if min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {min_samples}")
        
        # Create and fit DBSCAN model
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
        )
        
        # Fit and get cluster labels
        labels = model.fit_predict(X)
        
        # Calculate statistics
        n_clusters = len(set(labels) - {-1})  # Exclude -1 (noise)
        n_noise = int(np.sum(labels == -1))
        
        self.logger.info(f"DBSCAN found {n_clusters} clusters")
        self.logger.info(f"DBSCAN found {n_noise} noise points")
        
        if n_clusters > 0:
            # Count points in each cluster (excluding noise)
            cluster_sizes = np.bincount(labels[labels != -1])
            self.logger.info(f"Cluster sizes: {cluster_sizes}")

        # Visualize clusters (only works well for 2D data)
        if X.shape[1] == 2:
            plt.figure(figsize=(9, 7))

            is_noise = labels == -1

            # Plot clustered points colored by cluster (same style as K-Means)
            scatter = plt.scatter(
                X[~is_noise, 0],
                X[~is_noise, 1],
                c=labels[~is_noise],
                cmap='viridis',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5,
            )

            # Plot noise points (outliers) with a distinct grey 'x' marker
            plt.scatter(
                X[is_noise, 0],
                X[is_noise, 1],
                marker='x',
                s=50,
                c='grey',
                linewidths=1.5,
                label=f'Noise ({n_noise} points)',
            )

            plt.xlabel("Feature 1", fontsize=12)
            plt.ylabel("Feature 2", fontsize=12)
            plt.title(
                f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})",
                fontsize=14,
                fontweight='bold',
            )
            if n_clusters > 0:
                plt.colorbar(scatter, label='Cluster')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            self.plots.finish("clustering_dbscan.png")

        return model, labels

    def run_gmm(
        self,
        X: np.ndarray,
        n_components: int = 3,
    ) -> tuple[GaussianMixture, np.ndarray]:
        """
        Perform Gaussian Mixture Model (GMM) clustering and visualize results.

        GMM Algorithm (Expectation-Maximization):
            1. Start with K Gaussian "bell curves" (one per cluster)
            2. E-step: For each point, work out the probability it came
               from each Gaussian
            3. M-step: Update each Gaussian's centre (mean), shape
               (covariance) and size (weight) using those probabilities
            4. Repeat steps 2-3 until the model stops improving

        Key Difference from K-Means:
            K-Means: Hard assignment - each point belongs to ONE cluster
            GMM:     Soft assignment - each point gets a probability for
                     EVERY cluster (e.g. 95% cluster 0, 5% cluster 1)

        Properties:
            - Clusters can be oval (elliptical), not just circular
            - Gives a confidence for every assignment
            - Must specify K in advance (like K-Means)
            - BIC score helps choose K (lower is better)

        When to Use GMM:
            ✓ Clusters are roughly bell-shaped (Gaussian)
            ✓ Clusters may be stretched or tilted ovals
            ✓ You want to know how confident each assignment is
            ✗ Clusters have complex, non-oval shapes (use DBSCAN)
            ✗ Many outliers (GMM still forces them into a cluster)

        Args:
            X: Feature matrix (n_samples, n_features)
            n_components: Number of clusters (K), called "components" in GMM

        Returns:
            model: Fitted GaussianMixture object
            labels: Most likely cluster for each point (n_samples,)

        Raises:
            ValueError: If n_components is invalid
        """
        if n_components < 1 or n_components > len(X):
            raise ValueError(
                f"n_components must be between 1 and {len(X)}, got {n_components}"
            )

        # Create and fit the GMM model
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",  # Each cluster gets its own oval shape
            n_init=10,               # Try 10 starts, keep the best (same as K-Means)
            random_state=self.config.random_state,
        )

        # Fit and get the most likely cluster for each point
        labels = model.fit_predict(X)

        # Soft assignment: probability of each point belonging to each cluster
        # Shape: (n_samples, n_components), each row adds up to 1
        probabilities = model.predict_proba(X)

        # A point is "uncertain" if its best cluster is below 90% probability
        n_uncertain = int(np.sum(probabilities.max(axis=1) < 0.9))

        # Visualize clusters (only works well for 2D data)
        if X.shape[1] == 2:
            plt.figure(figsize=(9, 7))

            # Plot data points colored by cluster (same style as K-Means)
            scatter = plt.scatter(
                X[:, 0],
                X[:, 1],
                c=labels,
                cmap='viridis',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5,
            )

            # Plot cluster centres (the mean of each Gaussian)
            plt.scatter(
                model.means_[:, 0],
                model.means_[:, 1],
                marker='X',
                s=300,
                c='red',
                edgecolors='black',
                linewidths=2,
                label='Cluster Centers',
            )

            # Draw an oval around each cluster showing its shape.
            # The oval covers about 2 standard deviations (~95% of the cluster).
            for mean, covariance in zip(model.means_, model.covariances_):
                # Eigenvalues give the oval's size, eigenvectors its direction
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
                width, height = 2 * 2 * np.sqrt(eigenvalues[::-1])
                plt.gca().add_patch(Ellipse(
                    xy=mean,
                    width=width,
                    height=height,
                    angle=angle,
                    fill=False,
                    edgecolor='red',
                    linewidth=1.5,
                    linestyle='--',
                ))

            plt.xlabel("Feature 1", fontsize=12)
            plt.ylabel("Feature 2", fontsize=12)
            plt.title(
                f"Gaussian Mixture Model (K={n_components})",
                fontsize=14,
                fontweight='bold',
            )
            plt.colorbar(scatter, label='Cluster')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            self.plots.finish("clustering_gmm.png")

        self.logger.info(f"GMM completed with {n_components} clusters")
        self.logger.info(f"Cluster sizes: {np.bincount(labels)}")
        self.logger.info(f"BIC score: {model.bic(X):.2f}")
        self.logger.info(f"Uncertain points (<90% confidence): {n_uncertain}")

        return model, labels


# Example usage (for testing this module independently)
if __name__ == "__main__":
    from utils.logger import configure_logging
    from data_loaders.dataset_manager import DatasetManager
    
    # Set up
    configure_logging()
    config = AppConfig(save_plots=True, show_plots=False)
    
    # Load clustering data
    print("=== Clustering Workflow Demo ===\n")
    dm = DatasetManager()
    X = dm.load_clustering_data()
    
    print(f"Dataset shape: {X.shape}\n")
    
    # Create workflow
    workflow = ClusteringWorkflow(config)
    
    # 1. Elbow method
    print("1. Running Elbow Method...")
    elbow_results = workflow.plot_elbow(X, max_clusters=7)
    print("\nElbow Results:")
    print(elbow_results.to_string(index=False))
    print("\nLook at the plot to find the 'elbow' point!\n")
    
    # 2. K-means clustering
    print("2. Running K-means Clustering (K=3)...")
    kmeans_model, kmeans_labels = workflow.run_kmeans(X, n_clusters=3)
    print(f"   Cluster sizes: {np.bincount(kmeans_labels)}\n")
    
    # 3. DBSCAN clustering
    print("3. Running DBSCAN Clustering...")
    dbscan_model, dbscan_labels = workflow.run_dbscan(X, eps=1.0, min_samples=5)
    
    n_clusters = len(set(dbscan_labels) - {-1})
    n_noise = int(np.sum(dbscan_labels == -1))
    print(f"   Clusters found: {n_clusters}")
    print(f"   Noise points: {n_noise}")
    
    if n_clusters > 0:
        cluster_sizes = np.bincount(dbscan_labels[dbscan_labels != -1])
        print(f"   Cluster sizes: {cluster_sizes}")

    # 4. Gaussian Mixture clustering
    print("\n4. Running Gaussian Mixture Clustering (K=3)...")
    gmm_model, gmm_labels = workflow.run_gmm(X, n_components=3)
    print(f"   Cluster sizes: {np.bincount(gmm_labels)}")
    print(f"   BIC score: {gmm_model.bic(X):.2f}")

    print("\n✓ Clustering workflow completed successfully!")
