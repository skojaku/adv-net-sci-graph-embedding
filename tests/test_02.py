# %% Import
from scipy import sparse
import pandas as pd
import igraph
import numpy as np

g = igraph.Graph.Famous("Zachary")
A = g.get_adjacency_sparse()
A = A.astype(float)  # Convert to float type

# Symmterize and binarize
A = A + A.T
A.data = A.data * 0 + 1



# %% Test -----------
D = sparse.diags(A.sum(axis=1).flatten().A1)
L = D - A
emb = LaplacianEigenMap(A, 5)

test_vals = np.sort(np.diag(emb.T @ L @ emb))
target_vals = np.array([0.13227233, 0.28704899, 0.38731323, 0.61223054, 0.64899295])
assert np.allclose(test_vals, target_vals), "The embedding vectors for the Laplacian Eigenmap are not correctly computed."
