# UMAP: Uniform Manifold Approximation and Projection

UMAP is a manifold learning and dimensionality reduction technique that preserves local and global structure while projecting high‑dimensional data to 2D or 3D. It is efficient, flexible, and widely used for visualization and general‑purpose reduction.

## Overview
- Developed by Leland McInnes, John Healy, and James Melville.
- Preserves neighborhood topology and global relationships better than many alternatives.
- Integrates with the scikit‑learn API via `umap-learn`.

## Mathematical Foundations
- Riemannian Manifold: Assumes data lie on a smooth manifold locally approximated by Euclidean space.
- Riemannian Metric: Locally constant or approximable; defines distances on the manifold.
- Topological Data Analysis: Constructs a fuzzy topological representation (simplicial set) capturing local connectivity.

## Algorithm
- Nearest Neighbors: Find k nearest neighbors for each point (often approximate for speed).
- Fuzzy Simplicial Set: Build a weighted graph with membership strengths indicating connection probabilities.
- Cross‑Entropy Optimization: Optimize a low‑dimensional embedding to match the high‑dimensional fuzzy graph.
- Stochastic Gradient Descent: Iteratively adjust point positions to minimize the objective.

## Installation

```bash
pip install umap-learn
```

## Basic Usage

```python
import numpy as np
import umap
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
X = StandardScaler().fit_transform(X)

reducer = umap.UMAP()
embedding = reducer.fit_transform(X)

plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.colorbar()
plt.tight_layout()
plt.show()
```

## Example: Random 4D to 2D

```python
import numpy as np
import umap
import matplotlib.pyplot as plt

np.random.seed(44)
data = np.random.rand(700, 4)
embedding = umap.UMAP().fit_transform(data)

colors = data[:, :3]
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10)
plt.tight_layout()
plt.show()
```

## Key Parameters
- n_neighbors: Size of local neighborhood; low values emphasize local structure, high values capture more global.
- min_dist: Minimum spacing in embedding; lower values allow tighter clusters, higher values spread points.
- n_components: Embedding dimensionality (2 or 3 typical for visualization).
- metric: Distance function (e.g., euclidean, cosine, manhattan); choose to match data characteristics.
- random_state: Controls reproducibility.

## Advanced Usage
- Supervised UMAP: Provide labels to guide embedding for clearer class separation.
- Metric Customization: Use domain‑appropriate metrics for better neighborhood fidelity.
- Initialization: Supply initial embeddings to guide optimization or stabilize runs.

## Comparison
- vs t‑SNE: UMAP is faster, scales better, and often preserves more global structure while maintaining local neighborhoods.
- vs PCA: PCA is linear and preserves variance globally; UMAP is nonlinear and preserves topology, revealing manifold structure.

## Best Practices
- Scale features (e.g., StandardScaler) before UMAP to normalize distances.
- Tune n_neighbors and min_dist for desired local/global balance and cluster compactness.
- Use appropriate metric; cosine often works well for sparse/high‑dimensional data.
- Validate embeddings with downstream tasks or neighborhood preservation checks.

## Applications
- Visualization of high‑dimensional datasets (images, text embeddings, single‑cell RNA‑seq).
- Preprocessing for clustering or classification.
- Exploration of manifold structure and prototype discovery.

## Conclusion
UMAP combines strong mathematical grounding with practical speed and flexibility. By preserving neighborhood topology and offering tunable control over local/global balance, it delivers informative embeddings for analysis and visualization at scale.
