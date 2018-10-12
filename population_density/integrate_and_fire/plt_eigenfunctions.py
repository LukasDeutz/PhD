from eigenspectrum import EV, phi_n
import numpy as np
import itertools as it
import matplotlib.pyplot as plt


if __name__ == '__main__':
               
    #===============================================================================
    # Parameter
    #===============================================================================
    
    theta = 1.
    sig2 = 1.
    
    eta_arr = np.array([-1, 1])               
    n_arr = np.array([1,2,3])
    
    #===============================================================================
    # Plotting
    #===============================================================================
    gs = plt.GridSpec(len(eta_arr), 2)
    
    fz = 18 # fontsize
    lfz = 14 # legend fontsize
    
    v_arr = np.linspace(0, 1, 1000)
    
    for i, eta in enumerate(eta_arr):
        
        ax_i0 = plt.subplot(gs[i, 0]) 
        ax_i1 = plt.subplot(gs[i, 1]) 
        
        for n in n_arr:
            
            xi = eta*theta/sig2
            
            m = 0  # it is sufficient to look at m=0 because m=1 corresponds to conjugate pair
            lam_n = EV(xi, sig2, theta, n, m)
            lam_n = np.complex(lam_n.real, lam_n.imag)
            
            phi_n_i = phi_n(v_arr, lam_n, eta, xi, theta, sig2)
            ax_i0.plot(v_arr, np.real(phi_n_i), label = '$n=%i$' % n)
            ax_i1.plot(v_arr, np.imag(phi_n_i), label = '$n=%i$' % n)
            
            ax_i0.set_title(r'$\eta = %.2f$' % eta, fontsize = fz)
        
        ax_i0.plot([0,1], [0, 0], c = 'k', ls = '--', lw = 0.5)
        ax_i1.plot([0,1], [0, 0], c = 'k', ls = '--', lw = 0.5)
        ax_i0.set_xlim(0,1)
        ax_i1.set_xlim(0,1) 
        ax_i0.set_ylabel('$Re(\phi_n)$', fontsize = fz)
        ax_i1.set_ylabel('$Im(\phi_n)$', fontsize = fz)
        ax_i0.legend(fontsize = lfz)
        ax_i1.legend(fontsize = lfz)
                
    ax_i0.set_xlabel('v', fontsize = fz)
    ax_i1.set_xlabel('v', fontsize = fz)
        
    plt.show()

