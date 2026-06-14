# nptorch — Machine Learning from Scratch

NumPy implementations of machine learning algorithms, built as part of the **Machine Learning Lab** course at the **University of Passau**. Each algorithm is validated against scikit-learn and explored through Jupyter notebooks.

## Roadmap

| # | Topic | Module | Status |
|---|-------|--------|--------|
| 1 | Linear Regression | `mlab.regression` | ✅ Done |
| 2 | Logistic Regression | `mlab.regression` | ✅ Done |
| 3 | K-Means Clustering | `mlab.clustering` | ✅ Done |
| 4 | Naive Bayes | `mlab.naive_bayes` | ✅ Done |
| 5 | Decision Trees | `mlab.trees` | ✅ Done |
| 6 | Ensembles | `mlab.ensemble` | ✅ Done |
| 7 | Support Vector Machines | `mlab.svm` | 🔜 Upcoming |
| 8 | Artificial Neural Networks | `mlab.neural_network` | 🔜 Upcoming |
| 9 | Advanced Deep Learning | `mlab.neural_network` | 🔜 Upcoming |
| 10 | Convolutional Neural Networks | `mlab.neural_network` | 🔜 Upcoming |

## Implemented Algorithms

### Regression (`mlab/regression/`)
- **`LinearRegressor`** — closed-form solution via SVD (`np.linalg.lstsq`)
- **`SGDRegression`** — mini-batch stochastic gradient descent with L2 regularization and early stopping
- **`LogisticRegression`** — full-batch gradient descent with L2 regularization, configurable threshold, and loss history
- **`SGDClassifier`** — mini-batch SGD variant of logistic regression

### Clustering (`mlab/clustering/`)
- **`KMeans`** — random centroid initialization, label-stability early stopping
- **`KMeansPlusPlus`** — distance-weighted probabilistic initialization (subclass of `KMeans`)

### Naive Bayes (`mlab/naive_bayes/`)
- **`GaussianNaiveBayes`** — continuous features; variance smoothing scaled to data range
- **`MultinomialNaiveBayes`** — count features; Laplace smoothing; vectorized via matrix multiply

### Decision Trees (`mlab/trees/`)
- **`DecisionTree`** — ID3 + CART hybrid: categorical features use multi-way ID3 splits, continuous (float) features use binary threshold splits (best midpoint scan). Supports `max_depth`, `min_samples_split`, `min_impurity_decrease`, NaN handling, and `feature_importances_`. Batch predict routes sample index arrays through the tree with NumPy operations.

### Ensembles (`mlab/ensemble/`)
- **`RandomForest`** — bootstrap-sampled decision trees with per-split feature subsampling (`max_features="sqrt"` default). Majority-vote prediction; averaged `feature_importances_` across all trees.

## Setup

```bash
git clone <repo-url>
cd nptorch
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn plotly jupyterlab
```

## Usage

```python
from mlab.regression._linear import LinearRegressor, SGDRegression
from mlab.regression._logistic import LogisticRegression, SGDClassifier
from mlab.clustering._kmeans import KMeans, KMeansPlusPlus
from mlab.naive_bayes._naive_bayes import GaussianNaiveBayes, MultinomialNaiveBayes
from mlab.trees._decision_tree import DecisionTree
from mlab.ensemble._random_forest import RandomForest

# All estimators follow the scikit-learn API
model = RandomForest(n_estimators=100, max_depth=None, random_state=42).fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Notebooks

Launch JupyterLab from the repo root:

```bash
source .venv/bin/activate
jupyter lab mlab/notebooks/
```

| Notebook | Topic |
|----------|-------|
| `02-linear-regression-exercise-updated.ipynb` | Linear & polynomial regression, 3-D surface plots |
| `03-logistic-regression-exercise.ipynb` | Binary classification, decision boundaries |
