import numpy as np
from get_neighbors import get_neighbors
from KMD_MST import KMD_MST
import mlpack
import igraph
from scipy.stats import norm

def KMD_test(X, Y, M=None, Knn=None, Kernel="discrete", Permutation=True, B=500):
    if M is None:
        M = len(np.unique(Y))

    if not np.issubdtype(Y.dtype, np.number):
        raise ValueError("Input Y is not numeric.")

    if np.any(Y < 1) or np.any(Y > M):
        raise ValueError("Label Y should be in 1,...,M.")

    if isinstance(Kernel, str) and Kernel == "discrete":
        discrete_kernel = True
        Kernel = np.eye(M)
    elif isinstance(Kernel, np.ndarray):
        discrete_kernel = False
    else:
        raise ValueError("The Kernel argument should be a matrix.")

    if not isinstance(X, np.ndarray):
        X = np.array(X)

    n = X.shape[0]
    n_i = np.array([np.sum(Y == i) for i in range(1, M + 1)])

    if Knn is None:
        Knn = int(np.ceil(len(Y) / 10))

    if Permutation:
        if Knn != "MST":
            if not isinstance(Knn, int) or Knn <= 0:
                raise ValueError("Knn should be a positive integer or the string MST.")
            if Knn + 2 > n:
                raise ValueError("n should be greater than Knn + 1")

            if X.shape[0] == X.shape[1]:
                if np.any(np.diag(X) != 0):
                    print("Warning: The distance of some data point to itself is non-zero. Self-distances are ignored when computing the nearest neighbors.")
                def node_neighbors(i):
                    tie_dist = np.partition(X[i, :], Knn)[Knn]
                    id_small = np.where(X[i, :] < tie_dist)[0]
                    id_small = id_small + (id_small >= i)
                    id_equal = np.where(X[i, :] == tie_dist)[0]
                    if len(id_equal) == 1:
                        return np.concatenate((id_small, id_equal + (id_equal >= i)))
                    if len(id_equal) > 1:
                        return np.concatenate((id_small, np.random.choice(id_equal + (id_equal >= i), Knn - len(id_small), replace=False)))
                nn_index_X = np.array([node_neighbors(i) for i in range(n)])
            else:
                nn_index_X = get_neighbors(X, Knn)

            def Perm_stat(Y, resample_vector):
                Y = Y[resample_vector]
                def node_calculator(j):
                    return np.sum(Kernel[Y[j] - 1, Y[nn_index_X[j, :] - 1] - 1])
                return np.sum([node_calculator(j) for j in range(n)])

        else:
            if X.shape[1] == 1:
                order_of_X = np.argsort(X[:, 0])
                def Perm_stat(Y, resample_vector):
                    Y = Y[resample_vector]
                    def node_calculator(j):
                        return Kernel[Y[order_of_X[j] - 1] - 1, Y[order_of_X[j - 1] - 1] - 1] + Kernel[Y[order_of_X[j] - 1] - 1, Y[order_of_X[j + 1] - 1] - 1]
                    res = Kernel[Y[order_of_X[0] - 1] - 1, Y[order_of_X[1] - 1] - 1] + Kernel[Y[order_of_X[n - 1] - 1] - 1, Y[order_of_X[n - 2] - 1] - 1]
                    return (np.sum([node_calculator(j) for j in range(1, n - 1)]) / 2 + res) / n

            else:
                if X.shape[0] == X.shape[1]:
                    graph = igraph.Graph.Weighted_Adjacency(X.tolist(), mode=igraph.ADJ_UNDIRECTED)
                    mst = graph.spanning_tree(weights=graph.es["weight"])
                    out = np.array(mst.get_edgelist())
                else:
                    result = mlpack.emst(X)
                    out = result["output"][:, :2].astype(int)

                def Perm_stat(Y, resample_vector):
                    Y = Y[resample_vector]
                    tmp = np.zeros((n, 2))
                    for i in range(out.shape[0]):
                        tmp[out[i, 0], 0] += 1
                        tmp[out[i, 1], 0] += 1
                        tmp[out[i, 0], 1] += Kernel[Y[out[i, 0]] - 1, Y[out[i, 1]] - 1]
                        tmp[out[i, 1], 1] += Kernel[Y[out[i, 1]] - 1, Y[out[i, 0]] - 1]
                    return np.sum(tmp[:, 1] / tmp[:, 0])

        from sklearn.utils import resample
        b = [Perm_stat(Y, resample(np.arange(n))) for _ in range(B)]
        b_t0 = Perm_stat(Y, np.arange(n))
        p_value = (np.sum(np.array(b) >= b_t0) + 1) / (B + 1)
        print(f"Permutation p-value: {p_value}, b_t0: {b_t0}")
        return p_value, b_t0

    if n < 4:
        raise ValueError("At least 4 observations are needed for the asymptotic test.")

    if Knn != "MST":
        if not isinstance(Knn, int) or Knn <= 0:
            raise ValueError("Knn should be a positive integer or the string MST.")
        if Knn + 2 > n:
            raise ValueError("n should be greater than Knn + 1")

        if X.shape[0] == X.shape[1]:
            if np.any(np.diag(X) != 0):
                print("Warning: The distance of some data point to itself is non-zero. Self-distances are ignored when computing the nearest neighbors.")
            def node_neighbors(i):
                tie_dist = np.partition(X[i, :], Knn)[Knn]
                id_small = np.where(X[i, :] < tie_dist)[0]
                id_small = id_small + (id_small >= i)
                id_equal = np.where(X[i, :] == tie_dist)[0]
                if len(id_equal) == 1:
                    return np.concatenate((id_small, id_equal + (id_equal >= i)))
                if len(id_equal) > 1:
                    return np.concatenate((id_small, np.random.choice(id_equal + (id_equal >= i), Knn - len(id_small), replace=False)))
            nn_index_X = np.array([node_neighbors(i) for i in range(n)])
        else:
            nn_index_X = get_neighbors(X, Knn)

        def node_calculator(j):
            return np.sum(Kernel[Y[j] - 1, Y[nn_index_X[j, :] - 1] - 1])
        First_term_in_numerator = np.mean([node_calculator(j) for j in range(n)]) / Knn

        num_in_neighbors = np.zeros(n)
        g3 = 0
        for i in range(n):
            for j in nn_index_X[i, :]:
                num_in_neighbors[j] += 1
                if i in nn_index_X[j, :]:
                    g3 += 1
        g3 = g3 / n / Knn**2
        g2_g1 = np.mean(num_in_neighbors * (num_in_neighbors - 1)) / Knn**2
        g1 = 1 / Knn

    else:
        if X.shape[1] == 1:
            Y = Y[np.argsort(X[:, 0])]
            def node_calculator(j):
                return Kernel[Y[j] - 1, Y[j - 1] - 1] + Kernel[Y[j] - 1, Y[j + 1] - 1]
            First_term_in_numerator = (np.sum([node_calculator(j) for j in range(1, n - 1)]) / 2 + Kernel[Y[0] - 1, Y[1] - 1] + Kernel[Y[n - 1] - 1, Y[n - 2] - 1]) / n

            g1 = 0.5 + 1 / n
            g2_g1 = 0.5
            g3 = 0.5 + 0.5 / n

        else:
            if X.shape[0] == X.shape[1]:
                graph = igraph.Graph.Weighted_Adjacency(X.tolist(), mode=igraph.ADJ_UNDIRECTED)
                mst = graph.spanning_tree(weights=graph.es["weight"])
                out = np.array(mst.get_edgelist())
            else:
                result = mlpack.emst(X)
                out = result["output"][:, :2].astype(int)

            tmp = np.zeros((n, 2))
            in_neighbor_indices = [[] for _ in range(n)]
            for i in range(out.shape[0]):
                tmp[out[i, 0], 0] += 1
                tmp[out[i, 1], 0] += 1
                tmp[out[i, 0], 1] += Kernel[Y[out[i, 0]] - 1, Y[out[i, 1]] - 1]
                tmp[out[i, 1], 1] += Kernel[Y[out[i, 1]] - 1, Y[out[i, 0]] - 1]
                in_neighbor_indices[out[i, 0]].append(out[i, 1])
                in_neighbor_indices[out[i, 1]].append(out[i, 0])
            First_term_in_numerator = np.mean(tmp[:, 1] / tmp[:, 0])

            g1 = np.mean(1 / tmp[:, 0])
            def node_calculator(j):
                return np.sum(1 / tmp[in_neighbor_indices[j], 0])**2 - np.sum(1 / tmp[in_neighbor_indices[j], 0]**2)
            g2_g1 = np.mean([node_calculator(j) for j in range(n)])
            def node_calculator(j):
                return np.sum(1 / tmp[in_neighbor_indices[j], 0]) / tmp[j, 0]
            g3 = np.mean([node_calculator(j) for j in range(n)])

    if discrete_kernel:
        tilde_a = np.sum(n_i * (n_i - 1)) / n / (n - 1)
        tilde_b = np.sum(n_i * (n_i - 1) * (n_i - 2)) / n / (n - 1) / (n - 2)
        tilde_c = (np.sum(n_i * (n_i - 1))**2 - np.sum(n_i**2 * (n_i - 1)**2) + np.sum(n_i * (n_i - 1) * (n_i - 2) * (n_i - 3))) / n / (n - 1) / (n - 2) / (n - 3)
    else:
        tilde_a = np.dot(n_i.T, np.dot(Kernel**2, n_i)) - np.sum(np.diag(Kernel)**2 * n_i)
        tilde_b = np.dot(n_i.T, np.dot(Kernel, n_i * np.dot(Kernel, n_i))) - 2 * np.dot(n_i.T, np.dot(Kernel, np.diag(Kernel) * n_i)) - tilde_a + np.sum(np.diag(Kernel)**2 * n_i)
        tilde_c = (np.dot(n_i.T, np.dot(Kernel, n_i)) - np.sum(np.diag(Kernel) * n_i))**2 - 4 * tilde_b - 2 * tilde_a

        tilde_a = tilde_a / n / (n - 1)
        tilde_b = tilde_b / n / (n - 1) / (n - 2)
        tilde_c = tilde_c / n / (n - 1) / (n - 2) / (n - 3)

    Sn = tilde_a * (g1 + g3 - 2 / (n - 1)) + tilde_b * (g2_g1 - g1 - 2 * g3 - 1 + 4 / (n - 1)) + tilde_c * (g3 - g2_g1 + 1 - 2 / (n - 1))

    if discrete_kernel:
        U_stats = np.sum(n_i * (n_i - 1)) / n / (n - 1)
    else:
        U_stats = (np.dot(n_i.T, np.dot(Kernel, n_i)) - np.sum(np.diag(Kernel) * n_i)) / n / (n - 1)

    Output = (First_term_in_numerator - U_stats) * np.sqrt(n / Sn)
    print(f"First_term_in_numerator: {First_term_in_numerator}, U_stats: {U_stats}, Sn: {Sn}, Output: {Output}")
    print(f"First_term_in_numerator: {First_term_in_numerator}, U_stats: {U_stats}, Sn: {Sn}, Output: {Output}")
    p_value = 1 - norm.cdf(Output)
    return np.array([Output, p_value])
