import numpy as np
from rpy2.robjects import r, numpy2ri, packages
from KMD import KMD

# Activate the numpy to R conversion
numpy2ri.activate()

# Load the R script
r.source("../KMD/R/KMD.R")

# Generate some test data
np.random.seed(1)
X1 = np.random.rand(30, 2)
X2 = np.random.rand(30, 2)
X2[:, 0] += 1
X = np.vstack((X1, X2))
Y = np.array([1] * 30 + [2] * 30)
Knn = 3

# Call the R function
KMD_r = r["KMD"]
KMD_r_result = KMD_r(X, Y, M=2, Knn=Knn, Kernel="discrete")

# Call the Python function
KMD_py_result = KMD(X, Y, M=2, Knn=Knn, Kernel="discrete")

# Compare the results
print("R output:\n", KMD_r_result)
print("Python output:\n", KMD_py_result)
print("Difference:\n", np.abs(KMD_r_result - KMD_py_result))
assert np.isclose(
    KMD_r_result, KMD_py_result, atol=1e-6
), "The Python output does not match the R output."
