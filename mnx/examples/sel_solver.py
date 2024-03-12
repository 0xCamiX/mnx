"""
Given a system Ax = b, where A is a square matrix, and b is a column vector,
this example shows how to solve the system using the best method for the given
problem. The chosen method is by mnx package, which is a wrapper for numpy.
"""

import numpy as np
from mnx.sel.linear_solver import LinearSolver

A   = np.array([[30, 40, 20], [40, 45, 35], [35, 30, 20]])
b_1 = np.array([430, 590, 415])
b_2 = np.array([590, 820, 575])

mnx = LinearSolver(A, b_1)
print(mnx.solver())


