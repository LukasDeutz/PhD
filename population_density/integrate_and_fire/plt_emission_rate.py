from eigenspectrum import emission_rate, flux_0, EV
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

if __name__ == '__main__':
       
    #===============================================================================
    # Parameter
    #===============================================================================
    
    theta = 1.
    sig2 = 1.
    r0 = 20.
    eta = fsolve(lambda eta: r0 - flux_0(eta, sig2, theta), 5)[0]
    xi = eta*theta/sig2
    
    #===============================================================================
    # Emission rate
    #===============================================================================
    
    # Check if eigenvalues support oscillations
    lam_1 = EV(xi, sig2, theta, 1, 0)
    lam_1 = np.complex(lam_1.real, lam_1.imag)
    
    tau_1 = 1./np.abs(lam_1.real) # decay time constant
    T_1 = 2*np.pi/np.abs(lam_1.imag)
    
    print tau_1/T_1 
    
    # Calculate emission rate
    n_arr = np.arange(1, 11)
    t = np.linspace(0, 5.*tau_1, 1000)
    
    r = emission_rate(t, n_arr, eta, sig2, theta)

    #===============================================================================
    # Plotting
    #===============================================================================
    fig = plt.figure()
    ax = plt.subplot(111)
    
    ax.plot(t, r.real)
    plt.show() 
    
  