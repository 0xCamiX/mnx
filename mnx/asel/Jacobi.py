import numpy as np

from mnx.asel.IMethod import IMethod
from mnx.utils.norms import optimize_norm
def solve_jacobi(A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
    n = len(b)
    A = np.array(A, float)
    b = np.array(b, float).reshape(n, 1)
    x0 = np.array(x0, float).reshape(n, 1)
    D = np.diag(np.diag(A))
    LU = A - D
    x = x0
    for i in range(max_iter):
        x = np.dot(np.linalg.inv(D), b - np.dot(LU, x))
        norm, _ = optimize_norm(np.dot(A, x) - b)
        if norm < tol:
            return x, i
    return None, max_iter


class JacobiMethod(IMethod):

    def solve(self, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
        return solve_jacobi(A, b, x0, tol, max_iter)