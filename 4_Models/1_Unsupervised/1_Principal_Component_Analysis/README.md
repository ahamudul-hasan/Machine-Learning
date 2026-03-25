# Principal Component Analysis (PCA): 3D to 2D Walkthrough

## Overview
This notebook demonstrates Principal Component Analysis (PCA) from first principles by:
- creating a synthetic 3-feature dataset,
- standardizing the features,
- computing the covariance matrix,
- performing eigen decomposition,
- projecting the original 3D data into a 2D space,
- visualizing the results before and after transformation.

Notebook file: `1_PCA.ipynb`

---

## Why PCA?
PCA is a linear dimensionality reduction technique used to:
- reduce the number of features while preserving as much variance as possible,
- remove redundancy from correlated variables,
- simplify visualization (for example, 3D to 2D),
- improve training speed and sometimes generalization in machine learning workflows.

In this notebook, PCA is used to transform data from 3 dimensions (`feature1`, `feature2`, `feature3`) into 2 principal components (`PC1`, `PC2`).

---

## Core Theory (Simple but Precise)

### 1) Standardization
PCA is variance-based, so features must be on a comparable scale.

For each feature:

$$
z = \frac{x - \mu}{\sigma}
$$

where:
- $x$ is the original value,
- $\mu$ is the feature mean,
- $\sigma$ is the feature standard deviation.

### 2) Covariance Matrix
The covariance matrix captures pairwise linear relationships among features.

For standardized data matrix $X$:

$$
\Sigma = \frac{1}{n-1} X^T X
$$

A $3 \times 3$ covariance matrix is created in this example.

### 3) Eigen Decomposition
PCA solves:

$$
\Sigma v = \lambda v
$$

where:
- $\lambda$ are eigenvalues (variance explained along each direction),
- $v$ are eigenvectors (principal axes/directions).

### 4) Select Top Components
Sort eigenvalues in descending order and take the corresponding eigenvectors.
In the notebook, the first two eigenvectors are used.

### 5) Project Data
If $W$ is the matrix of selected principal axes, transformed data is:

$$
X_{PCA} = X W
$$

This reduces dimensionality from 3D to 2D.

---

## Code Walkthrough by Cell

### Cell 1: Create synthetic dataset
- Generates two multivariate normal clusters in 3D.
- Assigns class labels (`target` = 1 and 0).
- Combines both groups into one DataFrame.

### Cell 2: Preview data
- Displays the top rows of the combined dataset.

### Cell 3: 3D scatter plot (before PCA)
- Uses Plotly to visualize original 3D feature space.
- Colors points by class label.

### Cell 5: Standard scaling
- Applies `StandardScaler` to `feature1`, `feature2`, `feature3`.
- Ensures each feature contributes fairly to PCA.

### Cell 6: Covariance matrix
- Computes covariance across the three standardized features.

### Cell 7: Eigen decomposition
- Uses `np.linalg.eig(covariance_matrix)` to get:
  - `eigen_values`: variance magnitude along each axis,
  - `eigen_vectors`: directions of principal axes.

### Cells 8 and 9: Inspect eigenvalues and eigenvectors
- Helps verify which directions carry most information.

### Cell 10: 3D eigenvector visualization
- Draws data cloud and principal directions in 3D using a custom `Arrow3D` class.
- Includes `do_3d_projection` for compatibility with newer Matplotlib versions.

### Cell 11: Select top 2 principal components
- Keeps first two principal directions from `eigen_vectors`.

### Cell 12: Transform to 2D
- Projects standardized 3D data onto selected principal directions.
- Creates a new DataFrame with `PC1`, `PC2`, and `target`.

### Cell 13: 2D scatter plot (after PCA)
- Visualizes transformed data in 2D.
- Shows class structure retained after dimensionality reduction.

---

## Practical Interpretation
- Larger eigenvalue => that principal direction explains more variance.
- If the first two eigenvalues dominate, 2D PCA is a good compression of 3D data.
- The 2D plot should preserve most of the separability seen in the original 3D view.

---

## Libraries Used
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `plotly`

---

## How to Run
1. Open `1_PCA.ipynb`.
2. Run cells in sequence from top to bottom.
3. Inspect:
   - original 3D class distribution,
   - covariance/eigen outputs,
   - transformed 2D projection.

---

## Notes and Good Practices
- Always standardize numeric features before PCA unless features are already on the same scale.
- In production workflows, use `sklearn.decomposition.PCA` to get:
  - `explained_variance_`,
  - `explained_variance_ratio_`,
  - stable transform pipelines.
- For educational understanding, manual covariance and eigen decomposition (as done here) is excellent.

---

## Conclusion
This notebook provides both conceptual and implementation-level understanding of PCA:
- from covariance and eigenvectors,
- to geometric interpretation,
- to practical dimensionality reduction and visualization.

It is a strong foundation for applying PCA to real-world datasets with many correlated numeric features.
