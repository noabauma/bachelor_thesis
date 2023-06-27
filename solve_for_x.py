from sympy.solvers import solve
from sympy import *
import numpy as np

if __name__ == "__main__":
    #x = Symbol('x')
    x = 2.025

    rc = 2.5
    r0 = 1.42436
    Kb = 22.5
    Ec = -3.35
    dE_correction = -4.5 #0
    atom = -2.850061413 #0.978379 
    rhs = Kb/((Ec-dE_correction)*atom)
    
    B = r0**5/(x**4*r0 + 4.0*x**3 *(r0-rc)**2)
    
    a = (20.0*B*(x**4))/(r0**6)
    b = (8.0*B*x**5)/((r0-rc)**2 * r0**5)
    d = B*x**4/r0**4 - 1.0
    c = (x/(r0-rc)**3) * d *(2.0 + x/(r0-rc))

    
    print('this should be zero: ', (a + b + c)/d - rhs)

    eps = 1.0
    print('B  = ', B)
    A = ((Ec - dE_correction)*atom)/(eps*np.exp(x/(r0-rc))*d)
    print('A  = ', A)
    A_ = Kb/(eps*np.exp(x/(r0-rc))*(a+b*c))
    print('A_ = ', A_)
    print('Ec -dE = ', A_*(eps*np.exp(x/(r0-rc))*d))
    V2 = A*eps*np.exp(x/(r0-rc))*d
    print('V2(r0) = Ec = ', V2)
    K0 = 5.5
    gamma = 0.55
    lambda_ = (K0 *r0**2)/(2.0*eps*np.exp(2.0*gamma*x/(r0-rc))*0.75)
    print('lambda = ', lambda_)


    #solver
    x = Symbol('x')

    rc = 4.3
    r0 = 1.42436
    Kb = 22.5
    Ec = -6.6
    dE_correction = -4.5
    atom = -2.850061413
    rhs = Kb/(Ec-dE_correction*atom)
    
    B = r0**5/(x**4*r0 + 4.0*x**3 *(r0-rc)**2)
    
    a = (20.0*B*(x**4))/(r0**6)
    b = (8.0*B*x**5)/((r0-rc)**2 * r0**5)
    d = B*x**4/r0**4 - 1.0
    c = (x/(r0-rc)**3) * d *(2.0 + x/(r0-rc))
    
    print("\nsigma solutions: ", solve((a + b + c)/d  - rhs), "\n")

    x = 2.14505765521483
    eps = 1.0

    B = r0**5/(x**4*r0 + 4.0*x**3 *(r0-rc)**2)
    
    a = (20.0*B*(x**4))/(r0**6)
    b = (8.0*B*x**5)/((r0-rc)**2 * r0**5)
    d = B*x**4/r0**4 - 1.0
    c = (x/(r0-rc)**3) * d *(2.0 + x/(r0-rc))


    print('B  = ', B)
    A = ((Ec - dE_correction)*atom)/(eps*np.exp(x/(r0-rc))*d)
    print('A  = ', A)
    A_ = Kb/(eps*np.exp(x/(r0-rc))*(a+b*c))
    print('A_ = ', A_)
    print('Ec -dE = ', A_*(eps*np.exp(x/(r0-rc))*d))
    V2 = A*eps*np.exp(x/(r0-rc))*d
    print('V2(r0) = Ec = ', V2)
    K0 = 5.5
    gamma = 0.55
    lambda_ = (K0 *r0**2)/(2.0*eps*np.exp(2.0*gamma*x/(r0-rc))*0.75)
    print('lambda = ', lambda_)


