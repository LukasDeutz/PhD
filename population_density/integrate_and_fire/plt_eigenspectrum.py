from eigenspectrum import EV_xi_n
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

if __name__ == '__main__':

    theta = 1.
    sig2  = 1.

    xi_arr = np.linspace(-5, 5, 200)
    n_arr = np.array([1, 2, 3])
    
    lam_mat = EV_xi_n(theta, sig2, xi_arr, n_arr)

    gs = plt.GridSpec(2,1)
    ax0 = plt.subplot(gs[0]) 
    ax1 = plt.subplot(gs[1]) 

    for i, xi in enumerate(xi_arr):
        
        lam = lam_mat[i, :, :].flatten()
        
        xi_i = xi*np.ones(2*len(n_arr))
                                        
        ax0.plot(xi_i, np.real(lam), 'o', ms = 1.5, mfc = 'k', mec = 'k')
        ax1.plot(xi_i, np.imag(lam), 'o', ms = 1.5, mfc = 'k', mec = 'k')

    # Plot horizontal lines for orientation
    x = np.array([xi_arr[0], xi_arr[-1]])

    for n in n_arr:
        
        y = -2*np.pi**2*n**2*np.array([1, 1])
        ax0.plot(x, y, c = 'k', ls = '--', lw = 0.5)

    ax0.set_xlim(xi_arr[0], xi_arr[-1])    
    ax0.set_ylim([0, -250])    
    ax1.set_xlim(xi_arr[0], xi_arr[-1])
    ax1.set_ylim([-150, 150])    

    fz = 18
    
    ax0.set_ylabel(r'RE($\lambda$)', fontsize = fz)
    ax1.set_ylabel(r'IM($\lambda$)', fontsize = fz)
    ax1.set_xlabel(r'$\xi$', fontsize = fz)
    
    plt.show()
    
    
    



  
