# %% Import
from scipy import sparse
import pandas as pd
import igraph
import numpy as np
import sys

min_accuracy = float(sys.argv[1])

node_table = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/adv-net-sci-course/main/data/airport_network_v2/node_table.csv"
)
edge_table = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/adv-net-sci-course/main/data/airport_network_v2/edge_table.csv",
    dtype={"src": np.int32, "trg": np.int32},
)
src, trg = tuple(edge_table[["src", "trg"]].values.T)

rows, cols = src, trg
nrows, ncols = node_table.shape[0], node_table.shape[0]
A = sparse.csr_matrix(
    (np.ones_like(rows), (rows, cols)),
    shape=(nrows, ncols),
).asfptype()

# Symmterize and binarize
A = A + A.T
A.data = A.data * 0 + 1


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
# benchmarking

def benchmark_node_classification(A, random_state = 42, K_fold = 5):
    """
    Benchmark the node classification task.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
      The adjacency matrix of the network.
    dim : int
      The dimension of the embedding.
    random_state : int
      The random state for the train-test split.
    K_fold : int
      The number of folds for the cross-validation.

    Returns
    -------
    accuracy : float
      The accuracy of the node classification task.
    """
    # Compute the embedding
    emb = compute_network_embedding(A)

    # Split the node indices into training and testing sets
    node_indices = np.arange(A.shape[0])
    accuracy_list = []
    for train_indices, test_indices in KFold(n_splits=K_fold, shuffle=True, random_state=random_state).split(node_indices):
        # Train
        clf = NodeClassifier()
        clf.fit(emb[train_indices], node_table.loc[train_indices, "region"].values)

        # Predict
        ypred = clf.predict(emb[test_indices])

        # Evaluation
        accuracy = accuracy_score(node_table.loc[test_indices, "region"].values, ypred)
        accuracy_list.append(accuracy)

    return np.mean(accuracy_list)


accuracy = benchmark_node_classification(A)
print(f"Accuracy: {accuracy:.4f}")

# %% Test -----------
accuracy = benchmark_node_classification(A)

assert (accuracy >= min_accuracy), f"The classification accuracy: {accuracy}."
