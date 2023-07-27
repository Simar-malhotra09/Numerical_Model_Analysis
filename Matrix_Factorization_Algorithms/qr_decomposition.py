import numpy as np

def qr_method(matrix, max_iterations=1000, tolerance=1e-10):
    a_k = matrix.copy()
    
    for i in range(max_iterations):
        q_k, r_k = np.linalg.qr(a_k)
        a_k = np.dot(r_k, q_k)
        
        # Check for convergence
        off_diagonal_sum = np.sum(np.abs(a_k - np.diag(np.diagonal(a_k))))
        if off_diagonal_sum < tolerance:
            break
    
    eigenvalues = np.diagonal(a_k)
    return eigenvalues

def display_eigenvalues(matrix, iterations):
    for i in range(iterations):
        eigenvalues = qr_method(matrix, max_iterations=i+1)
        if i+1 in p:
            print(f'Iteration {i+1}:')
            print(eigenvalues)
            print()

# Example usage:
a = np.array([[0, 2], 
              [2, 3]])

p = [1, 5, 10, 20]
display_eigenvalues(a, max(p))
