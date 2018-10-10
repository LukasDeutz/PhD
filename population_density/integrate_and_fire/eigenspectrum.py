import numpy as np
import itertools as it
from mpmath import findroot, cosh, sinh

#===============================================================================
# Eigenvalues
#===============================================================================

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
            lam_mat[idx[0], idx[1], idx[2]] = EV(xi, sig2, theta, n, m)
        
    return lam_mat

def EV_eta_sig2_n(theta, eta_arr, sig2_arr, n_arr):
    '''Calculate eigenvalues for all parameters combinations [(sig2_arr, xi_arr), n_arr] for given theta and sigma'''
        
    m_arr = np.array([0,1])
    
    L_xi = len(eta_arr)
    L_n  = len(n_arr)
    L_m  = len(m_arr)         
             
    lam_mat= np.zeros((L_xi, L_n, L_m), dtype = np.cfloat)
    
    # parameter sweep                
    for idx, t in zip(it.product(np.arange(L_xi), np.arange(L_n), np.arange(L_m)), it.product(zip(eta_arr, sig2_arr), n_arr, m_arr)):    
                
        eta  = t[0][0]
        sig2 = t[0][1]
        n    = t[1]
        m    = t[2]

        xi = eta*theta/sig2

        if xi == 0:
            zeta = - 2*np.pi**2*sig2/theta**2 * n**2
        else:                                        
            lam_mat[idx[0], idx[1], idx[2]] = EV(xi, sig2, theta, n, m)
        
    return lam_mat

#===============================================================================
# Eigenfunctions 
#===============================================================================

def phi0(v, xi, eta, sig2, theta):
    '''Stationary solution of the Fokker-Planck operator for lambda = 0'''

    x = (theta - v)/theta
    c0 = 2*eta*xi/(theta*(2.*xi - 1 + np.exp(-2.*xi)))
    
    return c0/mu*(1-e**(-2*xi*x))
    
def phi_n(v, lam_n, eta, xi, sig2, theta):
    '''Eigenfunction corresponding to lam_n'''
    
    zeta = theta/sig2*np.sqrt(eta**2 + 2*sig2*lam_n) 
    
    cn = 2*xi/(theta*(zeta*xi*np.cosh(zeta) + (zeta**2 + xi)*np.sinh(zeta))) 
                   
    return cn*np.exp(xi*v/theta)*np.sinh(zeta*(theta - v)/theta) 
               
def spsi_n(v):
               
    zeta = theta/sig2*np.sqrt(eta**2 + 2*sig2*lam_n) 
           
    return np.exp(-xi*v/theta)*(zeta*np.cosh(zeta*v/theta) + xi*np.cosh(zeta*v/theta))

               
#===============================================================================
# Emission rate
#===============================================================================

def flux_0(eta, sig2, theta):
    
    xi = eta*theta/sig2
    
    return 1./(sig2/(2*eta**2)*(2*xi + np.exp(-2*xi) -1))
    
def flux_n(lam_n, eta, xi, sig2, theta):
        
    zeta = theta/sig2*np.sqrt(eta**2 + 2*sig2*lam_n)     
    cn = 2*xi/(theta*(zeta*xi*np.cosh(zeta) + (zeta**2 + xi)*np.sinh(zeta))) 
    
    return 0.5*sig2*cn*zeta*np.exp(xi)

def emission_rate(t, n_arr, eta, sig2, theta):

    xi = eta*theta/sig2

    r = flux_0(eta, sig2, theta)
        
    for n in n_arr: 
            
        lam_n0 = EV(xi, sig2, theta, n, 0) #m=0
        lam_n1 = EV(xi, sig2, theta, n, 1) #m=1
        
        lam_n0 = np.complex(lam_n0.real, lam_n0.imag)
        lam_n1 = np.complex(lam_n1.real, lam_n1.imag)
        
        an0 = theta/sig2*np.sqrt(eta**2 + 2*sig2*lam_n0)
        an1 = theta/sig2*np.sqrt(eta**2 + 2*sig2*lam_n1)
        
        r_n0 = an0*flux_n(lam_n0, eta, xi, sig2, theta)*np.exp(lam_n0*t)
        r_n1 = an1*flux_n(lam_n1, eta, xi, sig2, theta)*np.exp(lam_n1*t)
        r_n = r_n0 + r_n1
        
        r += r_n
        
    return r   


