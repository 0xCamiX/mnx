import numpy as np

from mnx.asel.IMethod import IMethod
from mnx.utils.norms import optimize_norm

def solve_gauss_seidel(A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
    n = len(b)
    A = np.array(A, float)
    b = np.array(b, float).reshape(n, 1)
    x0 = np.array(x0, float).reshape(n, 1)
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    x = x0
    for i in range(max_iter):
        x = np.dot(np.linalg.inv(D + L), b - np.dot(U, x))
        norm, _ = optimize_norm(np.dot(A, x) - b)
        if norm < tol:
            return x, i
    return None, max_iter
class GaussSeidelMethod(IMethod):

    def solve(self, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
        return solve_gauss_seidel(A, b, x0, tol, max_iter)