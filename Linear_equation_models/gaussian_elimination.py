import numpy as np


""" 
The Gauss Elimination method is a procedure to turn matrix into an upper triangular form to solve the system of equations.
Before we get to that, I'd like to introduce some simpler calculations in Linear algebra using python
"""


def gaussian_elimination(A, b):
    n = len(A)

    # Augmenting the coefficient matrix A with the constant terms b
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))

    for i in range(n):
        # Partial pivoting: Find the row with the maximum absolute value in the current column
        max_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))

        # Swap the current row with the row containing the maximum value in the current column
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        # Make the diagonal element of the current row equal to 1
        augmented_matrix[i] /= augmented_matrix[i, i]

        # Eliminate non-zero elements below the current row in the current column
        for j in range(i + 1, n):
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j, i]

    # Backward substitution to find the solution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.dot(
            augmented_matrix[i, i + 1 : n], x[i + 1 : n]
        )

    return x


"""
A = np.array([[2, 1, -1], [1, 3, 2], [1, 0, 0]], dtype=float)
b = np.array([8, 11, 3], dtype=float)

solution = gaussian_elimination(A, b)
print("Solution:", solution)
"""
