# Quick Start Guide

This guide will help you get the project running in approximately 5 minutes.

## Step 1: Installation (2 minutes)

First, navigate to the project directory and install the required dependencies:

```bash
# Navigate to the project directory
cd ml-student-project

# Install all required packages
pip install -r requirements.txt
```

## Step 2: Run the Clustering Example (1 minute)

```bash
python examples/run_clustering.py
```

When this completes successfully, you should observe:
- Information about the generated dataset
- Elbow method results (inertia for each K)
- K-Means, DBSCAN, and GMM results
- Generated plots saved in the `outputs/clustering/` directory

## Step 3: Run the Full Workflow (1 minute)

```bash
python main.py --save-plots --no-show
```

## Step 4: Attempt the Practice Exercise

```bash
python exercises/exercise_01_clustering.py
```

---

## Understanding the Output

When you run the clustering example, you will see output similar to this:

```
Step 1: Generating Clustering Data...
  Generated 300 samples with 2 features
  (Note: This is UNSUPERVISED - no labels provided!)

Step 2: Elbow Method (Finding Optimal K)...
 Number of Clusters      Inertia
                  1 20402.344443
                  2  5763.464789
                  3   566.859551
                  4   496.427953

K-means: 3 clusters
DBSCAN: 3 clusters, 5 noise points
GMM: 3 clusters, BIC=2418.11
```

**Interpretation**:
- No labels are provided; the algorithms must discover the groups themselves
- Inertia drops sharply up to K = 3 and only slightly afterwards, so the elbow is at K = 3
- All three algorithms find 3 clusters; DBSCAN additionally flags 5 points as noise (outliers)
- The BIC score is useful for comparing GMMs with different numbers of components (lower is better)

---

## Recommended Next Steps

1. **Read README.md** for comprehensive documentation of all components
2. **Study the source code**, beginning with `config/app_config.py`
3. **Complete the practice exercise** to reinforce your understanding
4. **Experiment with modifications** such as changing `eps` for DBSCAN

---

## Troubleshooting Common Issues

### Import Errors

If you encounter module import errors, verify you are in the project root:

```bash
cd ml-student-project
python -c "import sys; print(sys.path)"
```

### Plots Not Displaying

To enable interactive display, modify the configuration:
```python
config = AppConfig(
    save_plots=True,
    show_plots=True,  # Change from False to True
)
```

### Insufficient Output Verbosity

```python
from utils import configure_logging
import logging

configure_logging(level=logging.DEBUG)  # Enables verbose output
```

---

## Recommended Reading Order

1. **README.md**: Complete documentation of the project
2. **config/app_config.py**: Short and straightforward configuration module
3. **examples/run_clustering.py**: Demonstrates how all components work together
4. **workflows/clustering_workflow.py**: The full clustering workflow implementation

---

You are now ready to begin exploring the project. The code contains extensive comments to support your learning process.
