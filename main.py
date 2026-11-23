"""
Main Orchestrator - Run All ML Workflows
=========================================

This script runs the complete clustering workflow:
1. Elbow method (choosing the number of clusters)
2. K-Means
3. DBSCAN
4. Gaussian Mixture Model (GMM)

Usage:
    python main.py

    # Save plots without displaying them:
    python main.py --save-plots --no-show

    # Custom output directory:
    python main.py --output-dir my_results --save-plots

This is useful for:
- Getting a complete overview of the clustering techniques
- Comparing different clustering approaches
- Generating a full set of results and visualizations
"""

import argparse
import logging
from pathlib import Path

from config import AppConfig
from data_loaders import DatasetManager
from utils import configure_logging
from workflows import ClusteringWorkflow


class MachineLearningApplication:
    """
    Orchestrates all machine learning workflows.
    
    This class coordinates the execution of multiple ML workflows,
    ensuring proper setup and error handling.
    """
    
    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the application.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create managers and workflows
        self.datasets = DatasetManager()
        self.clustering = ClusteringWorkflow(config)
    
    def run_all(self) -> None:
        """
        Run all ML workflows in sequence.
        
        Catches and logs errors without stopping the entire application.
        This ensures that if one workflow fails, others still run.
        """
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        workflows = [
            ("Clustering", self._run_clustering),
        ]
        
        print("="*70)
        print("CLUSTERING WORKFLOW ORCHESTRATOR")
        print("="*70)
        print()
        
        for name, workflow_func in workflows:
            try:
                print(f"\n{'='*70}")
                print(f"Running {name} Workflow...")
                print('='*70)
                workflow_func()
                print(f"\n✓ {name} workflow completed successfully!")
            except Exception as e:
                self.logger.exception(f"{name} workflow failed: {e}")
                print(f"\n✗ {name} workflow failed: {e}")
        
        print("\n" + "="*70)
        print("CLUSTERING WORKFLOW COMPLETED")
        print("="*70)
        print(f"Results saved to: {self.config.output_dir}")
    
    def _run_clustering(self) -> None:
        """Run clustering workflow."""
        X = self.datasets.load_clustering_data()
        
        elbow_results = self.clustering.plot_elbow(X)
        print("\nElbow Method Results:")
        print(elbow_results.to_string(index=False))
        
        kmeans_model, kmeans_labels = self.clustering.run_kmeans(X, n_clusters=3)
        dbscan_model, dbscan_labels = self.clustering.run_dbscan(X)
        gmm_model, gmm_labels = self.clustering.run_gmm(X, n_components=3)

        print(f"\nK-means: {len(set(kmeans_labels))} clusters")
        print(f"DBSCAN: {len(set(dbscan_labels) - {-1})} clusters, "
              f"{sum(dbscan_labels == -1)} noise points")
        print(f"GMM: {len(set(gmm_labels))} clusters, BIC={gmm_model.bic(X):.2f}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the clustering workflow demonstrations."
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated artifacts (default: outputs/)",
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save generated figures to output directory",
    )
    
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures interactively",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Configure logging
    configure_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = AppConfig(
        random_state=42,
        output_dir=args.output_dir,
        save_plots=args.save_plots,
        show_plots=not args.no_show,
    )
    
    # Run application
    app = MachineLearningApplication(config)
    app.run_all()


if __name__ == "__main__":
    main()
