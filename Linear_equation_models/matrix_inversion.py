import numpy as np


def matrix_inversion(A, B):
    # Calculate the inverse of the coefficient matrix A
    A_inv = np.linalg.inv(A)

    # Compute the solution vector X using X = A_inv * B
    X = np.dot(A_inv, B)

    return X


def main():
    # Coefficient matrix A
    A = np.array([[7, 2, 1], [0, 3, -1], [-3, 4, -2]], dtype=float)

    # Constant vector B
    B = np.array([21, 5, -1], dtype=float)

    # Call the function to solve the linear system
    solution = matrix_inversion(A, B)

    # Print the solution
    print("Solution:", solution)


if __name__ == "__main__":
    main()
