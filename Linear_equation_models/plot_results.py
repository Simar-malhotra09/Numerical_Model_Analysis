import numpy as np
import matplotlib.pyplot as plt
from gaussian_elimination import gaussian_elimination
from lu_decomposition import lu_decomposition
from matrix_inversion import matrix_inversion
from test import generate_random_system
import time


def measure_execution_time(func, A, b):
    # Measure the execution time of the given function
    start_time = time.time()
    func(A, b)
    return time.time() - start_time


def main():
    matrix_sizes = [10, 50, 100, 200]  # Test with different matrix sizes
    repetitions = 5  # Number of repetitions for each matrix size

    gaussian_times = []
    lu_times = []
    matrix_inversion_times = []

    for size in matrix_sizes:
        gaussian_avg_time = 0.0
        lu_avg_time = 0.0
        matrix_inversion_avg_time = 0.0

        for _ in range(repetitions):
            A, b = generate_random_system(size)

            # Measure the average execution times for each method
            gaussian_avg_time += measure_execution_time(
                gaussian_elimination, A.copy(), b.copy()
            )
            lu_avg_time += measure_execution_time(lu_decomposition, A.copy(), b.copy())
            matrix_inversion_avg_time += measure_execution_time(
                matrix_inversion, A.copy(), b.copy()
            )

        gaussian_avg_time /= repetitions
        lu_avg_time /= repetitions
        matrix_inversion_avg_time /= repetitions

        gaussian_times.append(gaussian_avg_time)
        lu_times.append(lu_avg_time)
        matrix_inversion_times.append(matrix_inversion_avg_time)

    # Create the line graph
    plt.plot(matrix_sizes, gaussian_times, label="Gaussian Elimination", marker="o")
    plt.plot(matrix_sizes, lu_times, label="LU Decomposition", marker="o")
    plt.plot(matrix_sizes, matrix_inversion_times, label="Matrix Inversion", marker="o")

    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title(
        "Execution Time Comparison: Gaussian Elimination vs. LU Decomposition vs. Matrix Inversion"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig("execution_time_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
