# %% Import
from scipy import sparse
import pandas as pd
import igraph
import numpy as np

g = igraph.Graph.Famous("Zachary")
A = g.get_adjacency_sparse()
# Symmterize and binarize
A = A + A.T
A.data = A.data * 0 + 1
A = A.astype(float)  # Convert to float type

# %% Test -----------
eval, _ = sparse.linalg.eigs(A, k=1, which="LM")
eval_target = np.real(eval)[0]

ec = compute_eigencentrality(A)

eval_test = (ec.T @ A @ ec)

assert np.isclose(eval_target, eval_test)


