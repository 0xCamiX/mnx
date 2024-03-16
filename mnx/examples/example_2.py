"""
Example 2: Linear Solver
"""

import numpy as np

from mnx.sel.Solver import LinearSolver

# A system for Cholesky method.

A   = np.array([[2, 3], [3, 5]])
b_1 = np.array([115, 185])
b_2 = np.array([164, 264])

mnx = LinearSolver(A, b_1)
print(mnx)

# A system for LU method.

A_l   = np.array([[0, 2, 0, 1], [2, 2, 3, 2], [4, -3, 0, 1], [6, 1, -6, -5]])
b_l = np.array([0, -2, -7, 6])

mnx_l = LinearSolver(A_l, b_l, True)
print(mnx_l)

