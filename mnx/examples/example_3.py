import numpy as np

from mnx.asel.AproxSolver import AproxLinearSolver

A = np.array([
    [3, -1/10, -1/5],
    [1/10, 7, -1/3],
    [3/10, -1/5, 10]
])

b = np.array([157/20, 193/20, 357/5])

mnx = AproxLinearSolver(A, b, np.array([1,0,1]), 1e-5, 1000)

print(mnx)