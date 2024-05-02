# Jacobi Iterative Method

import numpy as np

def jacobi_method(A, b, x0, tol, max_iter):
    n = len(b)
    x = np.array(x0)
    iteration = 0
    
    while iteration < max_iter:
        xo = x.copy()
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * xo[j]
            x[i] = (b[i] - sigma) / A[i][i]
        if np.linalg.norm(x - xo) < tol:
            print("Solution converged in", iteration + 1, "iterations.")
            return x
        iteration += 1
    
    print("Maximum number of iterations exceeded.")
    return x

# Example usage
n = 3
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
b = np.array([5, -7, 4])
x0 = [0, 0, 0]
tol = 1e-6
max_iter = 1000

solution = jacobi_method(A, b, x0, tol, max_iter)
print("Approximate solution:", solution)





# Gauss-Seidel Iterative Method

import numpy as np

def gauss_seidel_method(A, b, x0, tol, max_iter):
    n = len(b)
    x = np.array(x0)
    iteration = 0
    
    while iteration < max_iter:
        xo = x.copy()
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]
        if np.linalg.norm(x - xo) < tol:
            print("Solution converged in", iteration + 1, "iterations.")
            return x
        iteration += 1
    
    print("Maximum number of iterations exceeded.")
    return x

# Example usage
n = 10
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
b = np.array([5, -7, 4])
x0 = [0, 0, 0]
tol = 1e-6
max_iter = 1000

solution = gauss_seidel_method(A, b, x0, tol, max_iter)
print("Approximate solution:", solution)





# SOR Method

import numpy as np

def sor_method(A, b, x0, omega, tol, max_iter):
    n = len(b)
    x = np.array(x0)
    iteration = 0
    
    while iteration < max_iter:
        xo = x.copy()
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (1 - omega) * xo[i] + (omega / A[i][i]) * (b[i] - sigma)
        if np.linalg.norm(x - xo) < tol:
            print("Solution converged in", iteration + 1, "iterations.")
            return x
        iteration += 1
    
    print("Maximum number of iterations exceeded.")
    return x

# Example usage
n = 3
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
b = np.array([5, -7, 4])
x0 = [0, 0, 0]
omega = 1.2
tol = 1e-6
max_iter = 1000

solution = sor_method(A, b, x0, omega, tol, max_iter)
print("Approximate solution:", solution)





# Iterative Refinement Method

import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    for i in range(n):
        max_index = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[max_index][i]):
                max_index = j
        A[[i, max_index]] = A[[max_index, i]]
        b[[i, max_index]] = b[[max_index, i]]
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            b[j] -= factor * b[i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
    return A, b

def iterative_refinement_method(A, b, max_iter, tol, t):
    n = len(b)
    A_orig, b_orig = A.copy(), b.copy()
    A, b = gaussian_elimination(A, b)
    x = np.linalg.solve(A, b)
    cond = np.linalg.norm(A_orig, ord=np.inf) * np.linalg.norm(np.linalg.inv(A_orig), ord=np.inf)
    k = 1
    
    while k <= max_iter:
        r = b_orig - np.dot(A_orig, x)
        y = np.linalg.solve(A_orig, r)
        xx = x + y
        if k == 1:
            cond = np.linalg.norm(y, ord=np.inf) / np.linalg.norm(xx, ord=np.inf, axis=0)
        if np.max(np.abs(x - xx)) < tol:
            print("Solution converged in", k, "iterations.")
            return xx, cond
        k += 1
        x = xx
    
    print("Maximum number of iterations exceeded.")
    return xx, cond

# Example usage
n = 3
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
b = np.array([5, -7, 4])
max_iter = 1000
tol = 1e-6
t = 10

solution, cond = iterative_refinement_method(A, b, max_iter, tol, t)
print("Approximate solution:", solution)
print("Approximation condition:", cond)




