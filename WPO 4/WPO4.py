'''------IMPORT MODULES----- '''
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d

'''------MAIN CODE------'''
def main():
    '''Ilya De Groeve, Arne De Weerdt'''
    speed = [30, 45, 60, 75, 90, 120]
    distance = [5.0, 12.3, 21.0, 32.9, 47.6, 84.7]


    polynomial_inter(speed,distance)
    spline_interpolation(speed,distance)
    least_squares(speed,distance)

'''------FUNCTIONS-------'''

def polynomial_inter(x, y):

    X = np.zeros((len(x), len(x)))
    for i in range(0,len(x)):
        for j in range(0, len(x)):
            X[i,j] = x[i]**j

    # solve xa = y for a
    a = np.linalg.solve(X,y)
    a = a[::-1]

    polyn = np.poly1d(a)

    # extra en intra polatie
    x_pol = np.append(x,[80,150])
    x_pol = np.sort(x_pol)
    y_pol = polyn(x_pol)
    

    plt.scatter(x,y)
    plt.plot(x_pol,y_pol, color='red', marker='*')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Distance (m)')
    plt.title('Exercise 1: Polynomial Interpolation')
    plt.show()



def spline_interpolation(x, y):
    # extra points to polate:
    x_pol = np.append(x, [80, 150])
    x_pol = np.sort(x_pol)
    
    nearest_interp = interp1d(x, y, kind='nearest', fill_value='extrapolate')
    linear_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")
    cubic_interp = interp1d(x, y, kind='cubic', fill_value='extrapolate')

    plt.scatter(x, y, color='black', label='Data points')
    plt.plot(x_pol, nearest_interp(x_pol), color='red', label='Nearest interpolation')
    plt.plot(x_pol, linear_interp(x_pol), color='blue', label='Linear interpolation')
    plt.plot(x_pol, cubic_interp(x_pol), color='green', label='Cubic interpolation')
    plt.legend()
    
    
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Distance (m)')
    plt.title('Exercise 2: Spline Interpolation')
    plt.show()



def least_squares(x,y):
    # use numpy.polyfit to find the coefficients of the polynomial to do a first order and a second order fit
    x_pol = np.append(x, [80, 150])
    x_pol = np.sort(x_pol)

    coeffs_1 = np.polyfit(x, y, 1)
    coeffs_2 = np.polyfit(x, y, 2)

    polyn_1 = np.poly1d(coeffs_1)
    polyn_2 = np.poly1d(coeffs_2)

    plt.scatter(x, y, color='black', label='Data points')
    plt.plot(x_pol, polyn_1(x_pol), color='red', label='First order fit')
    plt.plot(x_pol, polyn_2(x_pol), color='blue', label='Second order fit')
    plt.legend()
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Distance (m)')
    plt.title('Exercise 3: Least Squares Fit')
    plt.show()





if __name__=='__main__':
    main()

