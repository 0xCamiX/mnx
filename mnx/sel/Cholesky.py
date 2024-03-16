import numpy as np
from mnx.sel.IMethod import IMethod

class CholeskyMethod(IMethod):
    def solve(self, A: np.array, b: np.array):
        """
        Cholesky method for solving linear systems.

        :param A: np.array
        :param b: np.array
        :return: np.array
        """
        try:
            L = np.linalg.cholesky(A)
            LT = L.T
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(LT, y)
            return x
        except np.linalg.LinAlgError:
            return None