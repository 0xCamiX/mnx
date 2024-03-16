import numpy as np
from mnx.sel.IMethod import IMethod

class GaussJordanMethod(IMethod):
    def solve(self, A: np.array, b: np.array):
        """
        Gauss-Jordan method for solving linear systems.

        :param A: np.array
        :param b: np.array
        :return: np.array
        """
        A = np.array(A, float)
        b = np.array(b, float)
        n = len(b)

        for k in range(n):
            if np.fabs(A[k, k]) < 1.0e-12:
                for i in range(k+1, n):
                    if np.fabs(A[i, k]) > np.fabs(A[k, k]):
                        for j in range(k, n):
                            A[k, j], A[i, j] = A[i, j], A[k, j]
                        b[k], b[i] = b[i], b[k]
                        break
            pivot = A[k, k]
            for j in range(k, n):
                A[k, j] /= pivot
            b[k] /= pivot
            for i in range(n):
                if i == k or A[i, k] == 0: continue
                factor = A[i, k]
                for j in range(k, n):
                    A[i, j] -= factor * A[k, j]
                b[i] -= factor * b[k]

        return b