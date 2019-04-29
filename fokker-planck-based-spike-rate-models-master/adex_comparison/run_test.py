import numpy as np
from collections import OrderedDict 

from params import get_params
from multiprocessing import cpu_count


from methods_spectral import SpectralSolver, spectrum_enforce_complex_conjugation, \
                                          quantities_postprocess, inner_prod, \
                                          plot_raw_spectrum_sigma, plot_raw_spectrum_eigvals, \
                                          plot_quantities_eigvals, plot_quantities_real, \
                                          plot_quantities_complex, plot_quantities_composed

if __name__ == '__main__':

    #------------------------------------------------------------------------------ 
    # Initialize parameter 
    
    params = get_params() # default params
    N_eigvals = 10 # number of eigenmodes 
    N_mu = 461 # mu grid points -- s.t. -1.5 to 10 is spaced with distance 0.025
    N_sigma = 46 # sigma grid points  -- s.t. 0.5 to 5 is spaced with distance 0.1 
    N_procs = cpu_count() 
    
    mu_min    = -1.5 # min mu
    mu_max    = 5.0 # max mu
    sigma_min = 0.5 # min sigma
    sigma_max = 5. # max sigma
    
    mu = np.linspace(mu_min, mu_max, N_mu) # mu array
    sigma = np.linspace(sigma_min, sigma_max, N_sigma) # sigma_arr
    
    # the solver initialization uses the smallest mu value (which is assumed to be chosen 
    # so as to lead to purely real eigenvalues). there it (densely) evaluates the 
    # eigenflux at the lower bound on the following real grid 
    # attention: this grid  has to be fine enough, otherwise zeroes might be overlooked
    # while lambda_1,...,lambda_{N_eigvals} should lie within this interval here our 
    # code automatically enlarges the grid to finally get hold N_eigvals modes
    eigenval_init_real_grid = np.linspace(-5, -1e-4, 5000) 
    
    # Spectrum of FP Operator 
    # =============================================================================== 
    # calculate (or load) the spectrum, i.e., all nonstationary eigenvalues as indexed 
    # 1, 2, ..., N_eigvals where the index refers to the ordering w.r.t. 
    # increasingly negative real part at the smallest mean input mu 
    #===============================================================================
    
    specsolv = SpectralSolver(params)
    quantities_dict = OrderedDict() # for inserting quantities in order in hdf5 file 

        
    # SPECTRUM COMPUTATION

    # do the actual computation for N_eigvals eigenvalues on the mu sigma 
    # rectangle via the following method 
    # call initialized with all eigenvalues 
    # found by dense evaluation of the eigenvalue candidate array eigenval_init_real_grid
    lambda_all = specsolv.compute_eigenvalue_rect(mu, sigma, N_eigvals, 
                                                  eigenval_init_real_grid, N_procs=N_procs)
    
    quantities_dict['lambda_all'] = lambda_all
    quantities_dict['mu'] = mu
    quantities_dict['sigma'] = sigma
    
    # saving spectrum
    if save_spec:
    
        folder = os.path.dirname(os.path.realpath(__file__))    
        filename = 'spectrum_test.h5'
                
        specsolv.save_quantities(folder+'/'+filename, quantities_dict) 
    
        print('saving spectrum after computing done.')



class IF_Backward_Integrator():
    
    
    def __init__(self, param):
        '''Numerical integrate backward'''
        
        self.param = param # eigenvalue                
                        
        pass

    def get_A(self, x, model):
        '''Get matrix A
        :param model: neuron model'''
                
        if model == 'PIF':
            pass
        elif model == 'QIF':
            pass
        elif model == 'LIF':
            A = np.array([[0, self.lam], 
                          [1, x]])                        
        elif model == 'EIF':
            pass        
        
        return A
    
    def get_expAdx(A, dx):
        '''Caculate matrix exponential using '''

        t = dx

        s = 0.5*np.trace(A)
        q = cmath.sqrt(-np.linalg.det(A - s*np.identity(2)))
                        
        s = 0.5 * (X[0,0] + X[1,1])
        det_X_minus_sI = (X[0,0]-s) * (X[1,1]-s)  -  X[0,1] * X[1,0]
        q = cmath.sqrt(-det_X_minus_sI)
        # we have to distinguish the case q=0 (the other exceptional case t=0 should not occur)
        if abs(q) < 1e-15: # q is (numerically) 0
            sinh_qt_over_q = t
        else:
            sinh_qt_over_q = cmath.sinh(q*t) / q
        
        cosh_qt = cmath.cosh(q*t)
        cosh_q_t_minus_s_sinh_qt_over_qt = cosh_qt - s*sinh_qt_over_q
                
        expAdx = cmath.exp(s*t)*(cosh_q_t_minus_s_sinh_qt_over_qt*np.identity(2) + sinh_qt_over_q*A)
                                
        return expAdx
    
    def get_boundary(self, mu, sig):
        '''Get left and right boundary'''
        
        v_min = self.param['v_min']
        v_th  = self.param['v_th']
         
        if model == 'LIF':
            x_min = np.sqrt(2)*(v_min - mu)/sig
            x_th  = np.sqrt(2)*(v_th  - mu)/sig
            
        return x_min, x_th
                        
    def eigeneq_backwards_magnus_general(self, J_th, lam, mu, sig):        
        '''Integrate eigenfuncions backward from the threshold for given initial condition of the probability current
        :param J_th: probability current at threshold
        :param lam: eigenvalue
        :param mu: mean input
        :param sig: input fluctuations'''                
        # initialization as in Richardson 2007 and Ostojic 2011 
        # (complex version inspired by Srdjan Ostojic as in Schaffer 13)
        # as the eigenequation is linear, we can (together with the absorbing condition) scale
        # the flux at the spike boundary with an arbitrary complex constant
        # e.g. we can choose q(V_s)=init_q it to be (any) complex nonzero number   
                 
        N = 1e4 # number of grid points

        x_min, x_th = self.get_boundary(mu, sig)
            
        x_arr = np.linspace(0., 1.0, N)    
            
        phi = np.zeros_like(N) # eigenfunction
        J   = np.zeros_like(J) # probability current associated with eigenfunction
    
        phi[-1] = 0.0 + 0.0j   # absorbing boundary at threshold
        J[-1]   = J_th + 0.0j  # arbitrary complex scaling of the 
        
        A = self.get_A(model)
                
        exp_A_dV = np.zeros_like(A)
                
        dx = x_arr[1] - x_arr[0] 

        for k in range(Nv-2, -1, -1): # k= Nv-2 , ... , 0
        
            # grid spacing, allowing non-uniform grids
            dx   = x_arr[k+1]-x_arr[k]
            
            x = 0.5*(x_arr[k+1]+x_arr[k])
            
            A    = self.get_A(x, model)
            expA = self.get_expA(A)
            
            expAdx = self.exp_mat(A, dx)
            # (numba-njit-compatible) equivalent to exp_A_dV = scipy.linalg.expm(A*dx)
            
            # the following computes the matrix vector product exp(A*dx) * [q, phi]^T
            
            q[k]   = expAdx[0,0]*q[k+1] + expAdx[0,1]*phi[k+1]
            phi[k] = expAdx[1,0]*q[k+1] + expAdx[1,1]*phi[k+1]
            
            # reinjection condition
            if k == k_r:
                q[k] = q[k] - q[Nv-1] 

        return phi[k], q[k]
        






    
    

    