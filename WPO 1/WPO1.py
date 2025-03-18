import matplotlib.pyplot as plt
import numpy as np


def main():
    """ Ilya De Groeve, Arne De Weerdt"""
    g = 9.81
    theta0 = np.radians(30)
    u0 = 30
    y0 = 1.8

    f = lambda x: (np.tan(theta0) * x) - (g / (2 * u0**2 * np.cos(theta0)**2) * x**2) + y0 - 1
    f_deriv = lambda x: np.tan(theta0) - (g / (u0**2 * np.cos(theta0)**2) * x)

    sol_1, n1 = bisection(f, 0, 100)
    sol_2, n2 = regula_falsi(f, 0, 100)
    sol_3, n3 = newton(f, f_deriv, 100)
    sol_4, n4 = secant(f, 6, 100)

    print(f"bisection: {sol_1} in {n1} iterations")
    print(f"Regula Falsi: {sol_2} in {n2} iterations")
    print(f"Newton Ralphson: {sol_3} in {n3} iterations")
    print(f"Secant: {sol_4} in {n4} iterations")


def bisection(f, xl, xu, tol=1e-9, max_iter=10000):
    if f(xl) * f(xu) >= 0:
        print("There is no root between these points.")
    n = 0
    while n < max_iter:
        xr = (xl + xu) / 2
        if abs(xu - xl) < tol *abs(xr):
            break
        if f(xr)*f(xu) < 0:
            xl = xr
        else:
            xu = xr
        n += 1
    return xr, n

def regula_falsi(f, xl, xu, tol=1e-9, max_iter=10000):
    if f(xl) * f(xu) >= 0:
        print("There is no root between these points.")
    n = 0
    while n < max_iter:
        xr = xu - f(xu)*(xl-xu)/(f(xl)-f(xu))
        if abs(xu - xl) < tol *abs(xr):
            break
        if f(xr)*f(xu) < 0:
            xl = xr
        else:
            xu = xr
        n += 1
    return xr, n

def newton(f, df, initial_guess, tolerance=1e-9, max_iter=10000):
    n = 0
    x_current = initial_guess

    for n in range(max_iter):
        x_next = x_current - f(x_current) / df(x_current)
        if abs(f(x_next)) < tolerance:
            break
        x_current = x_next

    return x_next, n

def secant(f, x0, x1, tol=1e-9, max_iter=10000):
    x_0 = x0
    x_1 = x1
    n = 0

    if abs(f(x_0)) < tol:
        return x_0, 0
    if abs(f(x_1)) < tol:
        return x_1, 0

    for n in range(max_iter):
        if abs(f(x_1) - f(x_0)) < 1e-15:
            return x_1, n
            
        x_2 = x_1 - f(x_1) * (x_1 - x_0) / (f(x_1) - f(x_0))
        if abs(f(x_2)) < tol:
            return x_2, n+1
        
        x_0 = x_1
        x_1 = x_2

    return x_1, n+1



    
if __name__=='__main__':
    main()
    