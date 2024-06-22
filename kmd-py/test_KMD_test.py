import numpy as np
from rpy2.robjects import r, numpy2ri, packages
from KMD_test import KMD_test

# Activate the numpy to R conversion
numpy2ri.activate()

r.source("../KMD/R/KMD.R")

# Generate some test data
np.random.seed(1)
X1 = np.random.rand(100, 2)
X2 = np.random.rand(100, 2) * np.sqrt(1.5)
X3 = np.random.rand(100, 2) * np.sqrt(2)
X = np.vstack((X1, X2, X3))
Y = np.array([1] * 100 + [2] * 100 + [3] * 100)

# Call the Python function
KMD_test_py_result, b_t0 = KMD_test(X, Y, M=3, Knn=1, Kernel="discrete")

# Call the R function
KMD_test_r = r["KMD_test"]
KMD_test_r_result = KMD_test_r(X, Y, M=3, Knn=1, Kernel="discrete")

# Compare the results
print("R output:\n", KMD_test_r_result)
print("Python output:\n", KMD_test_py_result)
print("Difference:\n", np.abs(KMD_test_r_result - KMD_test_py_result))
assert np.allclose(KMD_test_r_result, KMD_test_py_result, atol=1e-6), "The Python output does not match the R output."
