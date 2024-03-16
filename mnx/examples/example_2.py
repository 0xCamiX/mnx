"""
Example 2: Linear Solver
"""

import numpy as np

from mnx.sel.Solver import LinearSolver

# A system for Cholesky method.

# A   = np.array([[2, 3], [3, 5]])
# b_1 = np.array([115, 185])
# b_2 = np.array([164, 264])
# b_3 = np.array([163, 265])
#
# mnx = LinearSolver(A, b_1, b_2)
# print(mnx)

# A system for LU method.

A = np.array([[30, 40, 20], [40, 45, 35], [35, 30, 20]])
b_1 = np.array([430, 590, 415])
b_2 = np.array([590, 820, 575])

mnx = LinearSolver(A, b_1, b_2)
print(mnx)
