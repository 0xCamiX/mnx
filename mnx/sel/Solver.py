import numpy as np

from mnx.utils.linalg import is_non_singular, is_symmetric, is_definite_positive

from mnx.sel.LU import LUMethod
from mnx.sel.Cholesky import CholeskyMethod
from mnx.sel.Gauss_Jordan import GaussJordanMethod
from mnx.sel.Gauss import GaussMethod

class LinearSolver:
    def __init__(self, A: np.array, *b: np.array):
        self.scores = None
        self.A = np.array(A)
        self.b = list(np.array(b))
        print(self.b)
        self.multi_b = len(self.b) > 1
        self.methods = {
            'cholesky': CholeskyMethod(),
            'lu': LUMethod(),
            'gauss_jordan': GaussJordanMethod(),
            'gauss': GaussMethod()
        }
        self.best_method = self.optimize()

    def optimize(self):
        """
        Optimize the method for the given problem.

        :return: IMethod
        """
        scores = {
            'cholesky': 0,
            'lu': 0,
            'gauss_jordan': 0,
            'gauss': 0
        }

        if is_non_singular(self.A):
            scores['lu'] += 1
            scores['gauss'] += 1
            scores['gauss_jordan'] += 1

        if is_symmetric(self.A):
            scores['cholesky'] += 1
        else:
            scores['cholesky'] -= 1


        if is_definite_positive(self.A):
            scores['cholesky'] += 1
        else:
            scores['cholesky'] -= 1

        if self.multi_b:
            scores['lu'] += 1
            scores['cholesky'] += 1

        print("Optimized method: ", max(scores, key=scores.get), scores)
        self.scores = scores
        return max(scores, key=scores.get)

    def solve(self):
        method = self.methods[self.best_method]
        solution = method.solve(self.A, self.b)

        if solution is not None:
            return solution
        return None

    def solve_all(self):
        """
        Filter the methods that can solve the problem and return the solutions.
        :return: dict
        """
        solutions = {}
        possible_methods = [k for k, v in self.scores.items() if v >= 0]

        # Remove the best method from the list.
        possible_methods.remove(self.best_method)

        for method in possible_methods:
            solution = self.methods[method].solve(self.A, self.b)
            if solution is not None:
                solutions[method] = solution

        return solutions

    def __str__(self):
        return f"""
        >>> Linear Solver <<<
        
A: {self.A}
b: {self.b}

        >>> Result <<<
        
a. Best Method: {self.best_method}
b. Solution: {self.solve()}
c. Other Methods: {self.solve_all()}
d. Explained why no other method was chosen: 
        """