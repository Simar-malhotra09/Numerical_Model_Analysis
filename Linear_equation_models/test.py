import time
import numpy as np
from gaussian_elimination import gaussian_elimination
from lu_decomposition import lu_decomposition
from matrix_inversion import matrix_inversion


def generate_random_system(size):
    # Helper function to generate random coefficient matrix and constant vector
    a = np.random.rand(size, size)
    b = np.random.rand(size)
    return a, b


def main():
    matrix_size = 100  # Change this to the desired matrix size

    # Generate a random system of linear equations
    a, b = generate_random_system(matrix_size)

    # Test Gaussian elimination and measure its execution time
    start_time = time.time()
    gaussian_solution = gaussian_elimination(a, b)
    gaussian_time = time.time() - start_time

    # Test LU decomposition and measure its execution time
    start_time = time.time()
    lu_solution = lu_decomposition(a, b)
    lu_time = time.time() - start_time

    start_time = time.time()
    matrix_inversion_solution = matrix_inversion(a, b)
    matrix_inversion_time = time.time() - start_time

    # print(f"Gaussian elimination solution: {gaussian_solution}")
    print(f"Gaussian elimination time: {gaussian_time:.6f} seconds")

    # print(f"LU decomposition solution: {lu_solution}")
    print(f"LU decomposition time: {lu_time:.6f} seconds")

    # print(f"Matrix inversion solution: {matrix_inversion_solution}")
    print(f"Matrix inversion time: {matrix_inversion_time:.6f} seconds")


if __name__ == "__main__":
    main()
