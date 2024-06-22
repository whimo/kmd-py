import numpy as np
from get_neighbors import get_neighbors

def KMD(X, Y, M=None, Knn=1, Kernel="discrete"):
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
    else:
        raise NotImplementedError("MST case is not implemented yet.")
    
    if discrete_kernel:
        U_stats = np.sum(n_i * (n_i - 1)) / n / (n - 1)
    else:
        U_stats = (n_i.T @ Kernel @ n_i - np.sum(np.diag(Kernel) * n_i)) / n / (n - 1)
    
    if discrete_kernel:
        mean_Kii = 1
    else:
        mean_Kii = np.sum(np.diag(Kernel) * n_i) / n
    
        def node_calculator(j):
            return np.sum(Kernel[Y[j] - 1, Y[nn_index_X[j, :] - 1]])
        return np.sum(Kernel[Y[j] - 1, Y[nn_index_X[j, :]] - 1])
    
    return (np.mean([node_calculator(j) for j in range(n)]) / Knn - U_stats) / (mean_Kii - U_stats)

