import numpy as np
from scipy.linalg import lu_factor, lu_solve


def lu_decomposition(A, b):
    # Compute the LU decomposition of the coefficient matrix A
    lu, piv = lu_factor(A)

    # Solve the system using LU decomposition
    x = lu_solve((lu, piv), b)

    return x


"""
A = np.array([[2, 1, -1], [1, 3, 2], [1, 0, 0]], dtype=float)
b = np.array([8, 11, 3], dtype=float)
solution = lu_decomposition(A, b)
print("Solution:", solution)
"""
