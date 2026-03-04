# Principal Component Analysis (PCA)

PCA is a linear dimensionality reduction technique that transforms correlated features into a smaller set of uncorrelated components while preserving as much variance as possible. It reduces redundancy, improves computational efficiency, and makes data easier to visualize and analyze.

## Why PCA
- Removes redundancy by summarizing correlated variables.
- Improves efficiency for training and inference.
- Enables visualization in 2D/3D for high‑dimensional data.
- Helps with noise reduction and multicollinearity handling.

## How PCA Works
- Standardize features to mean 0 and standard deviation 1 so scales are comparable.
- Compute the covariance matrix to capture feature relationships.
- Find eigenvectors (directions) and eigenvalues (importance) of the covariance matrix.
- Sort eigenvalues, select top k principal components that capture the most variance.
- Project original data onto these k components to obtain reduced‑dimension features.

## Key Concepts
- Principal Components: Orthogonal directions of maximum variance (PC1, PC2, ...).
- Eigenvectors: Directions defining principal components.
- Eigenvalues: Magnitudes indicating variance captured by each component.
- Explained Variance Ratio: Fraction of total variance captured per component; use cumulative sums to choose k (e.g., 95%).

## Python Implementation

### Imports

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

### Create Sample Dataset

```python
rng = np.random.default_rng(42)
n = 400
height = rng.normal(170, 10, n)
weight = rng.normal(70, 12, n)
age = rng.normal(35, 8, n)
gender = (0.3*height + 0.5*weight - 0.2*age + rng.normal(0, 10, n) > 0).astype(int)
df = pd.DataFrame({'Height': height, 'Weight': weight, 'Age': age, 'Gender': gender})
```

### Standardize

```python
X = df[['Height', 'Weight', 'Age']].values
y = df['Gender'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### PCA Fit and Variance

```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))
```

### Train Classifier on PCA Features

```python
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Visualize PCA Projection

```python
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1', s=30)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()
```

### Choosing k Components (e.g., 95% variance)

```python
pca_full = PCA().fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
k = np.searchsorted(cumvar, 0.95) + 1
print(k)
pca_k = PCA(n_components=k, random_state=42).fit(X_scaled)
```

### Component Loadings

```python
loadings = pd.DataFrame(pca_k.components_, columns=['Height', 'Weight', 'Age'])
print(loadings)
```

## Advantages
- Handles multicollinearity with uncorrelated components.
- Reduces noise by discarding low‑variance components.
- Compresses data for faster training and storage.
- Helps detect outliers in reduced space.

## Disadvantages
- Components are linear combinations that may be harder to interpret.
- Sensitive to scaling; improper preprocessing can mislead results.
- Some information loss when reducing dimensions.
- Assumes linear relationships; non‑linear structure may be missed.
- Can be computationally heavy on very large datasets.

## Best Practices
- Always standardize features before PCA.
- Use explained variance to pick k; avoid too few or too many components.
- Inspect component loadings to understand feature influence.
- Consider robust scaling if outliers are present.
- Use supervised dimension reduction (e.g., LDA) when labels should guide projection.
