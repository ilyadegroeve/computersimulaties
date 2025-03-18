'''------IMPORT MODULES----- '''
import matplotlib.pyplot as plt
import numpy as np
import scipy


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

    A,b = forward_elimination(A,b)
    x = back_substitution(A,b)
    print(x)

'''------FUNCTIONS-------'''
def forward_elimination(A,b):
    rows, cols = A.shape
    for i in range(1,rows):
        for j in range(i):
            factor = A[i,j]/A[j,j]
            A[i,:] = A[i,:] - factor*A[j,:]
            b[i] = b[i] - factor*b[j]
    print(A)
    print(b)
    return A,b
    
def back_substitution(A,b):
    rows, cols = A.shape
    x = np.zeros(rows)
    for i in range(rows-1,-1,-1):
        x[i] = b[i, 0]/A[i, i]
        for j in range(i+1,rows):
            x[i] = x[i] - A[i,j]*x[j]/A[i,i]
    return x

    
if __name__=='__main__':
    main()
    