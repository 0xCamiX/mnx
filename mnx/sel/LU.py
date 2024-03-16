import numpy as np
from scipy.linalg import lu_factor, lu_solve
from mnx.sel.IMethod import IMethod

class LUMethod(IMethod):
    def solve(self, A: np.array, b: np.array):
        """
        LU method for solving linear systems.

        :param A: np.array
        :param b: np.array
        :return: np.array
        """
        try:
            LU, piv = lu_factor(A)
            x = lu_solve((LU, piv), b)
            return x
        except np.linalg.LinAlgError:
            return None