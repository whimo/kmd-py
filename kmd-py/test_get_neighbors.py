import numpy as np
from rpy2.robjects import r, numpy2ri, packages
from get_neighbors import get_neighbors

# Activate the numpy to R conversion
numpy2ri.activate()

# Load the R script
r.source('../KMD/R/KMD.R')

# Generate some test data
np.random.seed(1)
X = np.random.rand(10, 2)
Knn = 3

# Call the R function
get_neighbors_r = r['get_neighbors']
nn_index_X_r = np.array(get_neighbors_r(X, Knn))

# Call the Python function
nn_index_X_py = get_neighbors(X, Knn)

# Compare the results
print("R output:\n", nn_index_X_r)
print("Python output:\n", nn_index_X_py)
print("Difference:\n", nn_index_X_r - nn_index_X_py)
