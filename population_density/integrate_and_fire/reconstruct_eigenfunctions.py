import numpy as np


class egf():
    """
    Reconstruct eigenfunctions for a specific neuron model from the numerical solution of the population density  
    """        
    def __init__(self, rho, t, c0, ev, rho0):
        """ 
        Constructor 
        :param rho: density
        :param t: time
        :param c: initial value for the expansion coefficients 
        :param ev: sorted eigenvalues beginning with the smallest absolute value 
        :param rho0: stationary distribution
        """
                
        self.rho_arr = rho 
        self.t_arr   = t 
        self.c0_arr  = c0
        self.ev_arr  = ev 
        # characteristic time scales of the individual modes
        self.tau_arr = 1./np.abs(np.real(ev))         
        # subtract stationary distribution to isolate contribution from higher modes 
        self.rho_modes = rho - rho0                

        self.phi_arr = []
                
    def succesive_reconstruction(self, M):
        """ 
        Reconstructs the first M eigenfunctions successively
        :param rho: density
        :param t: time
        :param ev: eigenvalues
        """
        
        for n in xrange(M):
            
            pass
    
    def reconstruct(self, n):
        """
        Reconstructs eigenfunction n  
        :param n: mode 
        """
        # start time must for the reconstruction must choosen such that all higher have died out already
        alpha = 10. # exp(-alpha*x) must be small                                                           
        tau = self.tau[n+1] # characteristic time scale of mode n+1            
        t_start = alpha*tau # start time after which we run the reconstruction 
        
        idx_arr = t > t_start
                
        rho_arr = self.rho_modes[idx_arr]
        t_arr   = self.t_arr
                
        for rho in rho_arr:
        
            phi_n = np.average(rho_arr/(self.c0_arr[n]*np.exp(self.ev[n]*t_arr)), axis = 1)
            
            
        
        
            
        
        
        
            
        
        
        
    
    
    






