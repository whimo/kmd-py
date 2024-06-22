import numpy as np
import igraph
import mlpack


def KMD_MST(X, Y, M, discrete_kernel, Kernel, n_i):
    n = X.shape[0]
    if discrete_kernel:
        U_stats = np.sum(n_i * (n_i - 1)) / n / (n - 1)
        Kernel = np.eye(M)
        mean_Kii = 1
    else:
        U_stats = (n_i.T @ Kernel @ n_i - np.sum(np.diag(Kernel) * n_i)) / n / (n - 1)
        mean_Kii = np.sum(np.diag(Kernel) * n_i) / n

    if X.shape[1] == 1:
        Y = Y[np.argsort(X[:, 0])]

        def node_calculator(j):
            return Kernel[Y[j] - 1, Y[j - 1] - 1] + Kernel[Y[j] - 1, Y[j + 1] - 1]

        res = Kernel[Y[0] - 1, Y[1] - 1] + Kernel[Y[n - 1] - 1, Y[n - 2] - 1]
        return (
            (np.sum([node_calculator(j) for j in range(1, n - 1)]) / 2 + res) / n
            - U_stats
        ) / (mean_Kii - U_stats)

    if X.shape[0] == X.shape[1]:
        graph = igraph.Graph.Weighted_Adjacency(X.tolist(), mode=igraph.ADJ_UNDIRECTED)
        mst = graph.spanning_tree(weights=graph.es["weight"])
        edges = np.array(mst.get_edgelist())
    else:
        result = mlpack.emst(X)
        edges = result["output"][:, :2].astype(int)

    tmp = np.zeros((n, 2))
    for i in range(edges.shape[0]):
        tmp[edges[i, 0], 0] += 1
        tmp[edges[i, 1], 0] += 1
        tmp[edges[i, 0], 1] += Kernel[Y[edges[i, 0]] - 1, Y[edges[i, 1]] - 1]
        tmp[edges[i, 1], 1] += Kernel[Y[edges[i, 1]] - 1, Y[edges[i, 0]] - 1]

    return (np.mean(tmp[:, 1] / tmp[:, 0]) - U_stats) / (mean_Kii - U_stats)
