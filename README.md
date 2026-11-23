# Machine Learning Student Project: Clustering

This production-oriented machine learning project has been designed for intermediate students. It focuses entirely on **clustering**, the core technique of unsupervised learning, using scikit-learn, with extensive documentation and educational comments throughout.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Guide](#module-guide)
- [Exercise](#exercise)
- [Key Concepts](#key-concepts)
- [Best Practices](#best-practices-demonstrated)
- [Troubleshooting](#troubleshooting)
- [Learning Path](#learning-path)

---

## Overview

I have organized this project as a modular, educational codebase built around a single, complete clustering workflow. Each module features:

- Comprehensive documentation with detailed comments explaining the rationale behind design decisions
- Independent executability for focused learning and testing
- Production-ready implementation following software engineering best practices
- Student-friendly structure with a clear example and a practice exercise

### Learning Objectives

Through this project, you will learn:

1. **Unsupervised Learning**: How to discover structure in data that has no labels
2. **Clustering Algorithms**: K-Means, DBSCAN, and Gaussian Mixture Models (GMM)
3. **Choosing the Number of Clusters**: The elbow method, silhouette scores, and BIC
4. **Interpreting Clusters**: Cluster sizes, noise points, and soft (probabilistic) assignments
5. **Software Engineering**: Modular code design, logging, validation, and configuration management

---

## Project Structure

```
ml-student-project/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── main.py                          # Runs the full clustering workflow
│
├── config/                          # Configuration management
│   ├── __init__.py
│   └── app_config.py                # Centralized configuration
│
├── utils/                           # Utility modules
│   ├── __init__.py
│   ├── data_validator.py            # Data validation utilities
│   ├── plot_manager.py              # Plot handling (save/display)
│   └── logger.py                    # Logging configuration
│
├── data_loaders/                    # Data loading
│   ├── __init__.py
│   └── dataset_manager.py           # Synthetic clustering dataset
│
├── workflows/                       # ML workflow
│   ├── __init__.py
│   └── clustering_workflow.py       # Elbow, K-Means, DBSCAN, GMM
│
├── examples/                        # Standalone example
│   └── run_clustering.py            # Step-by-step clustering demo
│
├── exercises/                       # Practice exercise
│   └── exercise_01_clustering.py    # Customer segmentation challenge
│
├── outputs/                         # Generated artifacts
│   └── clustering/                  # Elbow, K-Means, and GMM plots
│
└── visuals/                         # Visual documentation
    └── (optional: add your own diagrams)
```

---

## Installation

### Prerequisites

You will need the following installed on your system:

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or download this project**
   ```bash
   cd ml-student-project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # Activate on macOS/Linux:
   source venv/bin/activate

   # Activate on Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
   ```

---

## Quick Start

### Option 1: Run the Full Workflow

```bash
python main.py --save-plots --no-show
```

This command will:
- Generate the synthetic clustering dataset
- Run the elbow method, K-Means, DBSCAN, and GMM
- Save generated plots to the `outputs/` directory
- Print a summary of the results in the terminal

### Option 2: Run the Step-by-Step Example

```bash
python examples/run_clustering.py
```

### Option 3: Test Individual Modules

Each module can be executed independently for testing:

```bash
python -m config.app_config
python -m utils.data_validator
python -m data_loaders.dataset_manager
python -m workflows.clustering_workflow
```

---

## Module Guide

### 1. Configuration Module (`config/app_config.py`)

**Purpose**: Provides centralized configuration for the entire application.

**Key Concepts Demonstrated**:
- **Immutable configuration**: Uses `@dataclass(frozen=True)` to prevent accidental modifications
- **Validation**: Performs checks on configuration values during initialization
- **Type hints**: Documents expected types for each parameter

**Example**:
```python
from config import AppConfig

config = AppConfig(
    random_state=42,      # For reproducibility
    save_plots=True,      # Save visualizations
    show_plots=False,     # Do not open plot windows
)
```

---

### 2. Utilities (`utils/`)

#### Data Validator Module (`utils/data_validator.py`)

**Purpose**: Validates data before it reaches an algorithm, so errors are detected early.

**Validation Checks Performed**:
- Correct data types (DataFrame or ndarray)
- Correct dimensions (2D for features)
- Absence of missing values (NaN)
- Absence of infinite values
- Non-empty datasets

**Importance**:
Clustering algorithms are distance based. A single NaN or infinite value silently corrupts every distance calculation, so it is worth catching these problems before clustering begins.

**Example**:
```python
from utils import DataValidator

DataValidator.validate_features(X)  # Raises an error if invalid
```

#### Plot Manager Module (`utils/plot_manager.py`)

**Purpose**: Manages plot generation, saving, and display in a consistent manner.

**Example**:
```python
from utils import PlotManager

pm = PlotManager(config)

plt.scatter(X[:, 0], X[:, 1])
plt.title("My Plot")
pm.finish("my_plot.png")  # Saves and/or displays
```

#### Logger (`utils/logger.py`)

**Purpose**: Consistent logging across the application.

**Example**:
```python
from utils import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)

logger.info("Clustering started")
logger.warning("Many noise points detected")
```

---

### 3. Data Loaders (`data_loaders/dataset_manager.py`)

**Purpose**: Generate and validate the dataset used by the clustering workflow.

| Dataset | Type | Samples | Features | Hidden Structure |
|---------|------|---------|----------|------------------|
| Blobs (`make_blobs`) | Synthetic | 300 | 2 | 3 clusters |

Two features are used so every result can be plotted directly. Because the data is synthetic, we know there are truly three clusters, which allows us to check whether each algorithm discovers them.

**Example**:
```python
from data_loaders import DatasetManager

dm = DatasetManager()
X = dm.load_clustering_data()   # No labels: clustering is unsupervised
```

---

### 4. Clustering Workflow (`workflows/clustering_workflow.py`)

**Purpose**: Discover groups in unlabeled data.

**Algorithms**:
1. **K-Means**: Fast, requires K in advance, finds spherical clusters of similar size
2. **DBSCAN**: Density based, finds clusters of arbitrary shape, labels outliers as noise (`-1`)
3. **Gaussian Mixture Model (GMM)**: Probabilistic, gives each point a probability of belonging to each cluster, and supports elliptical clusters

**Key Methods**:

| Method | What It Does |
|--------|--------------|
| `plot_elbow(X)` | Fits K-Means for several K values and plots inertia |
| `run_kmeans(X, n_clusters)` | Fits K-Means and plots clusters with their centers |
| `run_dbscan(X)` | Fits DBSCAN and reports clusters and noise points |
| `run_gmm(X, n_components)` | Fits a GMM, plots its ellipses, and reports BIC and uncertain points |

**Example**:
```python
from workflows import ClusteringWorkflow

workflow = ClusteringWorkflow(config)

elbow_results = workflow.plot_elbow(X)                    # Choose K
kmeans_model, kmeans_labels = workflow.run_kmeans(X, n_clusters=3)
dbscan_model, dbscan_labels = workflow.run_dbscan(X)
gmm_model, gmm_labels = workflow.run_gmm(X, n_components=3)
```

**Generated Plots** (in `outputs/clustering/`):
- `clustering_elbow.png`: Inertia versus number of clusters
- `clustering_kmeans.png`: K-Means clusters and centers
- `clustering_gmm.png`: GMM clusters with covariance ellipses

---

## Exercise

### How to Use the Exercise

1. **Read the scenario** at the top of the exercise file
2. **Complete the TODO sections** following the hints
3. **Run your solution** and check the results
4. **Answer the questions** at the bottom of the file

### Exercise 1: Clustering Challenge

**File**: `exercises/exercise_01_clustering.py`

**Scenario**: Customer segmentation for an e-commerce company (no predefined labels)

**Learning Goals**:
- Apply the elbow method to choose K
- Compare K-Means and DBSCAN
- Use silhouette scores to judge cluster quality
- Identify and handle outliers

**Time**: 30 to 45 minutes

---

## Key Concepts

### Supervised versus Unsupervised Learning

In supervised learning, every example comes with a correct answer (a label). In clustering, there are **no labels**. The algorithm must discover groups on its own, and we must judge the result using internal measures and domain knowledge.

### The Clustering Workflow

```
1. Define the question (what would a "group" mean here?)
2. Load and validate the data
3. Choose the number of clusters (elbow method, silhouette, BIC)
4. Fit one or more clustering algorithms
5. Visualize and compare the results
6. Interpret the clusters and handle outliers
```

### The Elbow Method

Inertia is the within-cluster sum of squared distances. It always decreases as K increases, so we look for the "elbow": the point after which adding clusters gives only small improvements. On this project's data the elbow is clearly at K = 3.

### Silhouette Score

For each point, the silhouette compares how close it is to its own cluster with how close it is to the nearest other cluster. Scores range from -1 to 1; higher is better.

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

### DBSCAN Parameters

- `eps`: The neighborhood radius. Too small and most points become noise; too large and clusters merge.
- `min_samples`: The number of neighbors needed to form a dense region.

### Gaussian Mixture Models and BIC

A GMM assumes the data comes from a mixture of Gaussian distributions. Each point receives a probability for every cluster (soft assignment). The Bayesian Information Criterion (BIC) balances fit against complexity; lower is better, which makes it useful for choosing the number of components.

### Feature Scaling

Because clustering relies on distances, features with large ranges dominate those with small ranges. With real data, apply `StandardScaler` before clustering.

---

## Best Practices Demonstrated

1. **Modular Code**: Each module has a single, clear responsibility
2. **Documentation**: Docstrings explain the reasoning, not only the mechanics
3. **Validation**: Data and configuration are validated early with clear error messages
4. **Reproducibility**: `random_state` is set everywhere, so results are identical on every run
5. **Logging**: Progress and results are logged consistently
6. **Type Hints**: Expected types are documented throughout

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**: Install dependencies with `pip install -r requirements.txt`, and run commands from the project root.

### DBSCAN Labels Everything as Noise

**Solution**: Increase `eps` or decrease `min_samples`, and make sure features are scaled.

### Different Results on Each Run

**Solution**: Set `random_state` on every algorithm that uses randomness:
```python
KMeans(n_clusters=3, random_state=42)
GaussianMixture(n_components=3, random_state=42)
```

### Plots Not Displaying

**Solution**: Use `show_plots=True` in `AppConfig`, or omit `--no-show` when running `main.py`.

---

## Learning Path

### Beginner Path

1. Run `python examples/run_clustering.py` and study the output and plots
2. Read the modules in this order: `config/app_config.py`, `utils/data_validator.py`, `data_loaders/dataset_manager.py`, `workflows/clustering_workflow.py`
3. Complete Exercise 1

### Intermediate Path

1. Run `python main.py --save-plots`
2. Change the dataset in `load_clustering_data()` (more centers, larger `cluster_std`) and observe how each algorithm responds
3. Add silhouette scores to the workflow summary

### Advanced Path

1. Add a new algorithm (for example, Agglomerative Clustering or HDBSCAN)
2. Use `make_moons` to show where K-Means fails and DBSCAN succeeds
3. Apply the workflow to a real, scaled dataset (for example, customer data)
4. Add unit tests for the workflow

---

## Additional Resources

- [Scikit-learn Clustering User Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html)
- "Hands-On Machine Learning" by Aurélien Géron (Chapter on unsupervised learning)

---

## License

This project is designed for educational purposes. Feel free to use, modify, and share for learning.

## Acknowledgments

- **Scikit-learn team**: For the excellent ML library
- **Original author (Majid)**: For the initial comprehensive ML script

---

Machine learning is a continuous learning process rather than a fixed destination. I encourage you to take your time, experiment with the code, learn from errors, and build your understanding incrementally.
