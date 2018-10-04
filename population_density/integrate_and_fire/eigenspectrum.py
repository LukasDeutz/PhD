import numpy as np
import itertools as it
from mpmath import findroot, cosh, sinh

def CEAS(xi, n, m):
    '''Calculates analytic approximation of the characteristic equation'''
        
    i = np.complex(0, 1)
    a_n = 2*np.pi*n
    
    if xi <= 0: # noise-dominated regime: Eigenvalues are real, i.e. zeta must be purely imaginary or real                
        
        zeta = i * (a_n + 1./a_n * (1. + xi - np.exp(xi) + (-1)**m * np.sqrt((1 + xi - np.exp(xi))**2 - 2.*a_n**2*(np.exp(xi) - 1))))                 
    
    else: # drift-dominated regime: Eigenvalues are complex, i.e. zeta must have a nonzero real and complex part        
        
        zeta = xi + np.log(1 + np.sqrt(1 - np.exp(-2*xi))) + (-1)**m*i*a_n
        
    return zeta
        
def CES(xi, n, m):
    '''Finds numeric solution for characteristic equation for given n and xi'''
        
    # characteristic equation
    ce = lambda zeta: zeta*cosh(zeta) + xi*sinh(zeta) - np.exp(xi)*zeta
    # find root using analytic approximation as the start point
                
    zeta = findroot(ce, CEAS(xi, n, m), maxsteps= 100) 
        
    return zeta

def EV(xi, sig2, theta, n, m):
    '''Find numeric solution for eigenvalues determined by the characteristic equation'''
                 
    zeta = CES(xi, n, m)    
    lam = sig2/(2*theta**2)*(zeta**2 - xi**2)
        
    return lam

def EV_xi_n(theta, sig2, xi_arr, n_arr):
    '''Calculate eigenvalues for all parameters combinations in xi_arr, n_arr for given theta and sigma'''
    
    m_arr = np.array([0,1])
    
    L_xi = len(xi_arr)
    L_n  = len(n_arr)
    L_m  = len(m_arr)         
             
    lam_mat= np.zeros((L_xi, L_n, L_m), dtype = np.cfloat)

    # parameter sweep                
    for idx, t in zip(it.product(np.arange(L_xi), np.arange(L_n), np.arange(L_m)), it.product(xi_arr, n_arr, m_arr)):    
                
        xi = t[0]
        n  = t[1]
        m  = t[2]

        if xi == 0:
            zeta = - 2*np.pi**2*sig2/theta**2 * n**2
        else:                    
            
#             print 'Lambda'
#             print EV(xi, sig2, theta, n, m)
            lam_mat[idx[0], idx[1], idx[2]] = EV(xi, sig2, theta, n, m)
        
    return lam_mat
