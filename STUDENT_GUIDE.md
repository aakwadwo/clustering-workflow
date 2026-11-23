# Student Guide

Welcome to the Machine Learning Student Project. This guide provides a structured learning plan for the clustering codebase.

## Project Overview

This is a focused, production-style machine learning codebase that includes:

- A complete clustering workflow (elbow method, K-Means, DBSCAN, Gaussian Mixture Model)
- Supporting modules for configuration, validation, logging, plotting, and data loading
- An executable example script
- A practice exercise on customer segmentation
- Extensive inline documentation throughout all files

## Structured Learning Plan

### Week 1: Understanding the Foundation

**Days 1 through 2: Initial Familiarization**
1. Read QUICKSTART.md (approximately 5 minutes)
2. Execute the example and the full workflow:
   ```bash
   python examples/run_clustering.py
   python main.py --save-plots --no-show
   ```
3. Review the generated plots in the `outputs/clustering/` directory

**Days 3 through 4: Understanding the Structure**
1. Read the Project Structure, Module Guide, and Key Concepts sections in README.md
2. Execute individual modules to observe their standalone examples:
   ```bash
   python -m config.app_config
   python -m utils.data_validator
   python -m data_loaders.dataset_manager
   ```

**Days 5 through 7: Detailed Code Review**
Read these files in the following order, as each builds upon previous concepts:
1. `config/app_config.py`: Demonstrates configuration patterns
2. `utils/logger.py`: Shows logging implementation
3. `utils/data_validator.py`: Illustrates defensive programming
4. `utils/plot_manager.py`: Demonstrates resource management
5. `data_loaders/dataset_manager.py`: Shows data generation and validation
6. `workflows/clustering_workflow.py`: The complete clustering workflow

**Week 1 Objective**: Understand the overall structure and be able to run everything successfully.

---

### Week 2: Hands-On Practice

**Exercise 1 (Clustering)**
1. Open the file `exercises/exercise_01_clustering.py`
2. Carefully read the scenario and learning objectives
3. Complete all sections marked with TODO comments
4. Execute your solution and verify the results
5. Answer the questions at the bottom of the file
6. Compare your results with the output of `workflows/clustering_workflow.py`

**Week 2 Objective**: Be able to choose the number of clusters, compare algorithms, and explain your choices.

---

### Week 3: Experimentation and Extension

**Beginner Level Modifications:**
1. Change the number of centers and `cluster_std` in `load_clustering_data()`
2. Try different `eps` and `min_samples` values for DBSCAN
3. Customize the existing visualizations

**Intermediate Level Modifications:**
1. Add silhouette scores to the workflow summary
2. Compare GMM BIC scores for several values of `n_components`
3. Use `make_moons` data to show where K-Means fails and DBSCAN succeeds

**Advanced Level Modifications:**
1. Integrate a new algorithm (for example, Agglomerative Clustering or HDBSCAN)
2. Add automated tests for the clustering workflow
3. Create a data loader for your own (scaled) dataset

---

### Week 4 and Beyond: Real-World Applications

1. Locate an appropriate unlabeled dataset (Kaggle, UCI Machine Learning Repository, or your own research)
2. Create a custom data loader following the patterns in `data_loaders/`
3. Scale the features, then apply the clustering workflow
4. Interpret the clusters in terms of the domain
5. Document your process, findings, and conclusions

---

## Learning Objectives by Module

### Configuration Module (`config/`)
- [ ] Understand Python dataclasses and their advantages
- [ ] Explain why configuration should be immutable
- [ ] Implement validation for configuration values

### Utilities Module (`utils/`)
- [ ] Configure logging systems in Python applications
- [ ] Apply defensive programming principles
- [ ] Validate data before clustering
- [ ] Manage matplotlib plot generation and saving

### Data Loaders Module (`data_loaders/`)
- [ ] Generate synthetic data with `make_blobs`
- [ ] Explain why clustering data has no labels
- [ ] Validate generated data for quality

### Clustering Workflow (`workflows/clustering_workflow.py`)
- [ ] Understand the principles of unsupervised learning
- [ ] Apply the elbow method to determine the cluster count
- [ ] Compare and contrast K-Means, DBSCAN, and GMM
- [ ] Interpret noise points and soft assignments
- [ ] Calculate and interpret silhouette and BIC scores

---

## Strategies for Reading Code

### 1. Top-Down Approach

1. Read the module docstring
2. Read the class docstring (`ClusteringWorkflow`)
3. Read individual method docstrings
4. Finally, study the implementation details

### 2. Follow the Execution Flow

For the clustering workflow, follow this sequence:
1. `__init__`: Initial setup and configuration
2. `plot_elbow`: Choosing the number of clusters
3. `run_kmeans`: Partition-based clustering
4. `run_dbscan`: Density-based clustering and outlier detection
5. `run_gmm`: Probabilistic clustering

### 3. Execute Standalone Examples

Every module includes executable examples at the end in an `if __name__ == "__main__":` block. You can run these using `python -m module_name`.

---

## Success Criteria

**Understanding Phase (Week 1)**
- [ ] Execute the example and main workflow without errors
- [ ] Explain the purpose of each module

**Application Phase (Week 2)**
- [ ] Complete the practice exercise
- [ ] Explain how each clustering algorithm works
- [ ] Correctly interpret inertia, silhouette, and BIC

**Extension Phase (Week 3)**
- [ ] Modify existing code to add functionality
- [ ] Debug issues independently using systematic approaches

**Mastery Phase (Week 4 and beyond)**
- [ ] Apply clustering to new, unseen datasets
- [ ] Explain these concepts clearly to others

---

## Troubleshooting Guide

1. **Consult the documentation**: Module docstrings, method docstrings, inline comments, and README.md
2. **Run standalone examples**: `python -m module_name`
3. **Add diagnostic output**: `print(f"X shape: {X.shape}")`
4. **Use the Python debugger**: `import pdb; pdb.set_trace()`
5. **Analyze error messages**: They describe what went wrong and where

---

## Quick Reference

**Directory Structure:**
```
config/       (Configuration management)
utils/        (Utility functions)
data_loaders/ (Data loading module)
workflows/    (Clustering workflow)
examples/     (Executable demonstration)
exercises/    (Practice problem)
```

**Commands:**
```bash
python examples/run_clustering.py
python main.py --save-plots --no-show
python exercises/exercise_01_clustering.py
python -m workflows.clustering_workflow
```

---

I encourage you to read the code carefully, execute the examples, complete the exercise, and take time to explore the concepts thoroughly.
