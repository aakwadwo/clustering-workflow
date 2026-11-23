"""
Standalone Clustering Example
==============================

This script demonstrates clustering algorithms (unsupervised learning).
Run this to see clustering in action!

Usage:
    python examples/run_clustering.py

What This Script Does:
    1. Generates synthetic clustering data
    2. Uses elbow method to find optimal K
    3. Performs K-means clustering
    4. Performs DBSCAN clustering
    5. Performs Gaussian Mixture (GMM) clustering
    6. Compares results

Learning Objectives:
    - Understand unsupervised learning
    - Use elbow method
    - Compare K-means vs DBSCAN vs GMM
    - Interpret clustering results
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from data_loaders import DatasetManager
from utils import configure_logging
from workflows import ClusteringWorkflow


def main():
    """Run complete clustering workflow."""
    
    # Configure logging
    configure_logging()
    
    # Create configuration
    config = AppConfig(
        random_state=42,
        output_dir=Path("outputs/clustering"),
        save_plots=True,
        show_plots=False,
    )
    
    print("="*70)
    print("CLUSTERING WORKFLOW DEMO")
    print("="*70)
    print()
    
    # Load data
    print("Step 1: Generating Clustering Data...")
    dm = DatasetManager()
    X = dm.load_clustering_data()
    print(f"  ✓ Generated {len(X)} samples with {X.shape[1]} features")
    print("  (Note: This is UNSUPERVISED - no labels provided!)")
    print()
    
    # Create workflow
    workflow = ClusteringWorkflow(config)
    
    # Elbow method
    print("Step 2: Elbow Method (Finding Optimal K)...")
    elbow_results = workflow.plot_elbow(X, max_clusters=7)
    print()
    print(elbow_results.to_string(index=False))
    print()
    print("  → Look for the 'elbow' in the plot!")
    print("  → That's where adding clusters stops helping much")
    print()
    
    # K-means
    print("Step 3: K-means Clustering (K=3)...")
    kmeans_model, kmeans_labels = workflow.run_kmeans(X, n_clusters=3)
    kmeans_sizes = np.bincount(kmeans_labels)
    print(f"  ✓ Found 3 clusters")
    print(f"  ✓ Cluster sizes: {kmeans_sizes}")
    print(f"  ✓ Inertia: {kmeans_model.inertia_:.2f}")
    print()
    
    # DBSCAN
    print("Step 4: DBSCAN Clustering...")
    dbscan_model, dbscan_labels = workflow.run_dbscan(X, eps=1.0, min_samples=5)
    
    n_clusters = len(set(dbscan_labels) - {-1})
    n_noise = int(np.sum(dbscan_labels == -1))
    
    print(f"  ✓ Found {n_clusters} clusters automatically")
    print(f"  ✓ Identified {n_noise} noise points (outliers)")
    
    if n_clusters > 0:
        cluster_sizes = np.bincount(dbscan_labels[dbscan_labels != -1])
        print(f"  ✓ Cluster sizes: {cluster_sizes}")
    print()

    # Gaussian Mixture
    print("Step 5: Gaussian Mixture Clustering (K=3)...")
    gmm_model, gmm_labels = workflow.run_gmm(X, n_components=3)
    gmm_sizes = np.bincount(gmm_labels)
    gmm_bic = gmm_model.bic(X)
    print(f"  ✓ Found 3 clusters")
    print(f"  ✓ Cluster sizes: {gmm_sizes}")
    print(f"  ✓ BIC score: {gmm_bic:.2f} (lower is better)")
    print()

    # Comparison
    print("="*70)
    print("K-MEANS VS DBSCAN VS GMM")
    print("="*70)
    print()
    print("K-Means:")
    print("  ✓ Requires specifying K (number of clusters)")
    print("  ✓ Finds spherical/circular clusters")
    print("  ✓ Every point assigned to a cluster")
    print(f"  → Found {len(kmeans_sizes)} clusters with sizes: {kmeans_sizes}")
    print()
    print("DBSCAN:")
    print("  ✓ Automatically determines number of clusters")
    print("  ✓ Finds arbitrarily-shaped clusters")
    print("  ✓ Can identify outliers (noise points)")
    print(f"  → Found {n_clusters} clusters and {n_noise} noise points")
    print()
    print("GMM:")
    print("  ✓ Requires specifying K (like K-means)")
    print("  ✓ Finds oval-shaped (elliptical) clusters")
    print("  ✓ Gives a probability for every cluster (soft assignment)")
    print(f"  → Found {len(gmm_sizes)} clusters with sizes: {gmm_sizes}")
    print()
    print(f"Plots saved to: {config.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
