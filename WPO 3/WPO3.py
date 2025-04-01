'''------IMPORT MODULES----- '''
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg


'''------MAIN CODE------'''
def main():
    k = 20
    m1 = 1
    m2 = 1
    m3 = 1
    omega = 1
    # A1 = np.array([[2*k/m1 - omega**2, -k/m1, 0],[-k/m2, 2*k/m2 - omega**2, -k/m2],[0, -k/m3, 2*k/m3 - omega**2]])
    A = np.array([[2*k/m1, -k/m1, 0],[-k/m2, 2*k/m2, -k/m2],[0, -k/m3, 2*k/m3]])

    b = np.array([1,0,0])
    sol_1 = power_method(A, b, 1e-6)
    print(sol_1)
    sol_2 = inverse_power_method(A)
    print(sol_2)
    sol_3 = single_value_decomposition(A)
    print(sol_3)
    sol_4 = qr_decomposition(A)
    print(sol_4)

    
    
    
    '''------FUNCTIONS------'''

# Power Method
def power_method(A, x0, tol):
    x = x0
    p0 = np.dot(A,x0)
    n0 = np.max(abs(p0))
    x = p0/n0
    while True:
        p1 = np.dot(A,x)
        n1 = np.max(abs(p1))

        if abs((n1-n0)/n1) < tol:
            return n1, x
        
        x = p1/n1
        n0 = n1


def inverse_power_method(A):
    A_inv = np.linalg.inv(A)
    val, vec = np.linalg.eig(A_inv)
    return 1/val, vec

def single_value_decomposition(A):
    AAT = np.dot(A, A.T)
    ATA = np.dot(A.T, A)
    
    eigenvals_U, U = np.linalg.eig(AAT)
    eigenvals_V, V = np.linalg.eig(ATA)
    
    singular_values = np.sqrt(np.abs(eigenvals_U))
    
    idx = singular_values.argsort()[::-1]
    singular_values = singular_values[idx]
    U = U[:, idx]
    V = V[:, idx]
    
    Sigma = np.zeros_like(A, dtype=float)
    min_dim = min(A.shape)
    for i in range(min_dim):
        Sigma[i, i] = singular_values[i]
    
    return U, Sigma, V.T

def qr_decomposition(A):
    Q, R = np.linalg.qr(A)
    return Q, R

if __name__=='__main__':
    main()

