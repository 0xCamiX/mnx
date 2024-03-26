import numpy as np

from mnx.asel.IMethod import IMethod
from mnx.utils.norms import optimize_norm

def solve_richardson(A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
    n = len(b)
    A = np.array(A, float)
    b = np.array(b, float).reshape(n, 1)
    x0 = np.array(x0, float).reshape(n, 1)
    T = np.identity(n) - A

    x = x0

    for i in range(max_iter):
        error_relative = 0
        x = np.dot(T, x) + b
        norm, _ = optimize_norm(np.dot(A, x) - b)
        if norm < tol:
            error_relative = optimize_norm(np.dot(A, x) - b) / optimize_norm(b)*100
            return x, i, error_relative
    return None, max_iter
class RichardsonMethod(IMethod):

    def solve(self, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
        return solve_richardson(A, b, x0, tol, max_iter)