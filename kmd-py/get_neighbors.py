import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

def get_neighbors(X, Knn):
    if Knn >= X.shape[0]:
        raise ValueError("Knn must be less than the number of data points.")
        
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    # Compute the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=Knn + 2, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    nn_index_X = indices[:, 1:(Knn + 1)]

    # Find all data points that are not unique
    repeat_data = np.where(distances[:, 1] == 0)[0]
    if len(repeat_data) > 0:
        for i in repeat_data:
            group_indices = np.where(distances[:, 1] == 0)[0]
            if len(group_indices) > Knn:
                if Knn == 1 and len(group_indices) == 2:
                    nn_index_X[i, :] = [x for x in group_indices if x != i]
                else:
                    nn_index_X[i, :] = random.sample([x for x in group_indices if x != i], Knn)
            else:
                if distances.shape[1] > Knn + 2 and distances[i, Knn + 1] < distances[i, Knn + 2]:
                    nn_index_X[i, :] = [x for x in indices[i, 1:(Knn + 2)] if x != i]
                else:
                    dist_matrix = np.linalg.norm(X[i, :] - X, axis=1)
                    tie_dist = np.partition(dist_matrix, Knn)[Knn]
                    id_small = np.where(dist_matrix < tie_dist)[0]
                    id_small = id_small + (id_small >= i)
                    nn_index_X[i, :len(id_small)] = id_small
                    id_equal = random.sample(list(np.where(dist_matrix == tie_dist)[0]), Knn - len(id_small))
                    id_equal = id_equal + (id_equal >= i)
                    nn_index_X[i, len(id_small):Knn] = id_equal

    if distances.shape[1] > Knn + 2:
        ties = np.where(distances[:, Knn + 1] == distances[:, Knn + 2])[0]
        ties = np.setdiff1d(ties, repeat_data)
        if len(ties) > 0:
            for i in ties:
                dist_matrix = np.linalg.norm(X[i, :] - X, axis=1)
                tie_dist = np.partition(dist_matrix, Knn)[Knn]
                id_small = np.where(dist_matrix < tie_dist)[0]
                if len(id_small) > 0:
                    id_small = id_small + (id_small >= i)
                    nn_index_X[i, :len(id_small)] = id_small
                id_equal = random.sample(list(np.where(dist_matrix == tie_dist)[0]), Knn - len(id_small))
                id_equal = id_equal + (id_equal >= i)
                nn_index_X[i, len(id_small):Knn] = id_equal

    return nn_index_X + 1
