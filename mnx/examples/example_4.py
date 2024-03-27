"""
Actividad No. 8

1. Programe en Python un código que, dado un SEL 3×3, un x_0 y un δ:

a. Establezca las relaciones iterativas a realizar con cada uno de los tres métodos.
"""

import numpy as np
from mnx.les.LinearSolver import LinearSolver
from mnx.approx_linear_methods.Richardson import RichardsonMethod
from mnx.approx_linear_methods.Jacobi import JacobiMethod
from mnx.approx_linear_methods.GaussSeidel import GaussSeidelMethod

A = np.array([[8, 3, -3], [-2, -8, 5], [3, 5, 10]])
b = np.array([14, 5, -8])

x0 = np.array([0, 0, 0])
delta = 0.0001
max_iter = 1000

# Del objeto LinearSolver índicarle que instancie un modulo del solucionador por aproximación.

solver = LinearSolver.create_solver('approximation', A, b, x0, delta, max_iter)
print(solver.solve())