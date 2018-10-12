from eigenspectrum import flux_0, flux_n, EV, f_n_eta_sig2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    theta = 1.
    sig2 = 1.
    
    sig2_arr = np.array([0.6, 1.0, 1.4, 1.8])**2
    #sig2_arr = np.array([1.8])**2

    eta_arr_1 = np.linspace(-4., -0.1, 50)
    eta_arr_2 = np.linspace(0.1, 4., 50)
                
    f_1_mat_1 = f_n_eta_sig2(eta_arr_1, sig2_arr, theta, 1)            
    f_1_mat_2 = f_n_eta_sig2(eta_arr_2, sig2_arr, theta, 1)            
                    
    fig = plt.figure(figsize = (12, 8))
    
    gs = plt.GridSpec(2,1)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    for f_1_1 in f_1_mat_1:           
    
        ax1.plot(eta_arr_1, f_1_1.real, label = '$\sigma^2 = %.1f$' % sig2)
        ax2.plot(eta_arr_1, f_1_1.imag, label = '$\sigma^2 = %.1f$' % sig2) 

    for f_1_2 in f_1_mat_2:           
    
        ax1.plot(eta_arr_2, f_1_2.real, label = '$\sigma^2 = %.1f$' % sig2)
        ax2.plot(eta_arr_2, f_1_2.imag, label = '$\sigma^2 = %.1f$' % sig2) 

    ax1.grid(True)
    ax2.grid(True)

    plt.show()