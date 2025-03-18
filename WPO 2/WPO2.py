'''------IMPORT MODULES----- '''
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg


'''------MAIN CODE------'''
def main():
    Q13 = 40
    Q12 = 90
    Q21 = 30
    Q23 = 60
    Q33 = 120
    F_in = 200
    C1 = 0
    C2 = 0
    C3 = 0

    x = np.array([[C1], [C2], [C3]])
    A = np.array([[Q13 + Q12, Q21, 0], [-Q12, Q21+Q23, 0], [Q13, Q23, -Q33]])
    b = np.array([[F_in], [0], [0]]) 

    A_sol1, b_sol1 = forward_elimination(A,b)
    sol_1 = back_substitution(A_sol1,b_sol1)
    sol_2 = lu_factorization(A,b)
    sol_3 = jacobi(A,b,x,0.0001)
    sol_4 = gauss_seidel(A,b,x,0.0001)
    print(sol_1, sol_2, sol_3, sol_4)


'''------FUNCTIONS-------'''
def forward_elimination(A,b):
    rows, cols = A.shape
    for i in range(1,rows):
        for j in range(i):
            factor = A[i,j]/A[j,j]
            A[i,:] = A[i,:] - factor*A[j,:]
            b[i] = b[i] - factor*b[j]
    return A,b
    
def back_substitution(A,b):
    rows, cols = A.shape
    x = np.zeros((rows,1))
    for i in range(rows-1,-1,-1):
        x[i] = b[i,0]/A[i, i]
        for j in range(i+1,cols):
            x[i] = x[i] - A[i,j]*x[j]/A[i,i]
    return x

def lu_factorization(A,b):
    P, L, U = scipy.linalg.lu(A)

    d = np.linalg.solve(L,np.matmul(P,b))
    x = np.linalg.solve(U,d)
    return x

def gauss_seidel(A, b, x, tol):
    rows, cols = A.shape
    while True:
        x_new = np.copy(x)
        for i in range(rows):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def jacobi(A, b, x, tol):
    rows, cols = A.shape
    while True:
        x_new = np.copy(x)
        for i in range(rows):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x


if __name__=='__main__':
    main()

