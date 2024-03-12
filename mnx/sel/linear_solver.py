"""
Class for solving linear systems of equations using exactly and approximately methods.

This solver is capable of solving linear systems of equations using the best method
for the given problem.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from mnx.utils.linalg import is_definite_positive, is_non_singular, is_symmetric

class LinearError(ValueError):
    """
    Generic error for LinearSolver class.
    """

    def _raise_linear_error_singular(self):
        raise LinearError("Singular matrix")

    def _raise_linear_error_roche_frobenius(self):
        raise LinearError("The system has no solution")

    def _raise_linear_error_no_definite_positive(self):
        raise LinearError("The matrix is not definite positive")

    def _raise_linear_error_no_symmetric(self):
        raise LinearError("The matrix is not symmetric")

class LinearSolver:
    def __init__(self, A: np.array, b: np.array, ind_var: bool = False):
        self.A = A
        self.b = b
        self.ind_var = ind_var
        self.check_system()
        self.methods = self.optimizer()

    def solver(self):
        """
        Solve the system using the best method for the given problem.

        :return: np.array
        """

        if self.methods['cholesky']:
            L, LT = self.cholesky()
            if L is not None:
                y = np.linalg.solve(L, self.b)
                x = np.linalg.solve(LT, y)
                print("a. Cholesky method\n")
                print("b. Solution of the system is:")
                return x

        if self.methods['lu']:
            LU, piv = self.lu()
            if LU is not None:
                x = lu_solve((LU, piv), self.b)
                print("a. LU method")
                print("b. Solution of the system is:")
                return x

        if self.methods['gauss_jordan']:
            x = self.gauss_jordan()
            print("a. Gauss-Jordan method")
            print("b. Solution of the system is:")
            return x

    def optimizer(self):
        """
        Solve the system using the best method for the given problem.
        Will return a dictionary with the suitable methods and their suitability.

        :return: dict
        """

        methods = {
            'cholesky': True,
            'lu': True,
            'gauss_jordan': True
        }

        # Check if the matrix is not singular
        if not is_non_singular(self.A):
            methods['lu'] = False
            methods['gauss_jordan'] = False
            print("d. The matrix is singular. Can't use LU and Gauss-Jordan methods.")

        # Check if the matrix is symmetric
        if not is_symmetric(self.A):
            methods['cholesky'] = False
            print("d. The matrix is not symmetric. Can't use Cholesky method.")

        # Check if the matrix is definite positive
        if not is_definite_positive(self.A):
            methods['cholesky'] = False
            print("d. The matrix is not definite positive. Can't use Cholesky method.")

        # Check if the independent variable is multiple
        if self.ind_var:
            methods['gauss_jordan'] = False
            print("d. Considering multiple independent variables now or in the future.")
            print("d. Ignore Gauss and Gauss-Jordan methods.")

        print(f"Optimized the system, the recommend methods are: \n {methods}\n")
        return methods

    def cholesky(self) -> tuple:
        """
        Cholesky method.

        constraints:
            - definite positive
            - symmetric

        :return:
            - L: lower triangular matrix
            - L.T: transpose of L
        """
        try:
            L = np.linalg.cholesky(self.A)
            return L, L.T
        except np.linalg.LinAlgError:
            print("The matrix is not suitable for Cholesky method.")
            return None, None

    def lu(self) -> tuple:
        """
        LU method.

        constraints:
            - non-singular

        :return:
            - L: lower triangular matrix
            - U: upper triangular matrix
        """
        try:
            LU, piv = lu_factor(self.A)
            return LU, piv
        except np.linalg.LinAlgError:
            print("The matrix is not suitable for LU method.")
            return None, None

    def gauss_jordan(self) -> np.array:
        """
        Implementation of Gauss-Jordan method.

        constraints:
            - non-singular

        :return: np.array
        """

        # Amplify the A with b
        A_b = np.column_stack((self.A, self.b))
        n = A_b.shape[0]
        for i in range(n):
            # Divide the row by the diagonal element
            A_b[i] = A_b[i] / A_b[i, i]
            for j in range(n):
                if i != j:
                    A_b[j] = A_b[j] - A_b[j, i] * A_b[i]
        return A_b[:, -1]

    def check_system(self) -> bool:
        """
        Check if the system has a solution. Using Roche-Frobenius theorem.

        :return: True if the system has a solution, False otherwise
        """

        print("Checking fasting if the system has a solution.\n")

        flag1 = np.linalg.matrix_rank(self.A) == self.A.shape[0]
        # Amplify the A with b and check if the rank is the same
        A_b = np.column_stack((self.A, self.b))
        flag2 = np.linalg.matrix_rank(A_b) == self.A.shape[0]

        if flag1 and flag2:
            print("1. The system has a exact solution.\n")
            return True
        else:
            print("1. The system is incompatible indeterminate\n")
            return False