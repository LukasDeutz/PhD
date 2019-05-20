import numpy as np
import cmath
import h5py as h5
from LIF_response import LIF_Theory
import matplotlib.pyplot as plt

from timeit import timeit 

from scipy.optimize import brentq, curve_fit, root
import itertools as it 

import h5py as h5

# fortran for speed up
from backward_integration import complex_backward_integration, real_backward_integration
from calculate_J_lb import calc_j_lb_complex, calc_j_lb_real
from scipy.ndimage import gaussian_filter

class Spectral_Solver():
    '''Finds eigenfunctions and eigenvalues for the Fokker-Planck operator of a given one dimensional integrate-and-fire neuron.'''
    
    
    def __init__(self, model_param):
        
        self.model_param      = model_param
        self.solver_param     = self.default_solver_parameter()

        self.integrator = Integrator(self.model_param)

    def default_solver_parameter(self):
                
        lam_max  = -0.1 # maximal real part of eigenvalues 
        dlam     = 0.05 # stepsize to find eigenvalues on real line
        N_lam_bf = 3 #solver looks for the first 3 eigenvalues on the real line brute force        
        N_lam    = 10 # number of eigenvalues to be found
        dmu      = 5e-3 # grid spacing on mu axis
                        
        param = {}
        param['N_lam_bf'] = N_lam_bf
        param['N_lam']    = N_lam
                
        param['dlam']      = dlam
        param['lam_max']   = lam_max
        param['dmu']       = dmu
        
        return param
        
    def set_solver_parameter(self, **kwargs):
        '''Overwrites default parameters
        :param dict solver_param: dictionary with parameters'''
        
        for key, value in kwargs.items():
            
            self.solver_param[key] = value
        
        return
        
    def set_integrator_parameter(self, **kwargs):
        '''Overwrites default parameters of the integrator object
        :param dict solver_param: dictionary with parameters'''
        
        self.integrator.set_parameter(**kwargs)
        
        return
                
    def compute_real_ev(self, mu, sig):
        '''Caculate eigenvalues on the real line given sig and mu. It is assumed that mu is 
        choosen small enough to ensure that the spectrum of the FP operator is real.
        :param float mu: mu
        :param float sig: sigma'''

        #------------------------------------------------------------------------------ 
        # Find first N_lam_bf eigenvalues brute-force and then switch to extrapolation method         
                
        N_lam_bf = self.solver_param['N_lam_bf']          
        lam      = self.solver_param['lam_max'] # start value
        dlam     = self.solver_param['dlam'] # step size
        
        # initial flux at left boundary
        J_lb = self.integrator.calc_J_lb(lam, mu, sig)
        sign_J = np.sign(J_lb)

        # bracket zero crossings of J_lb as a function of lambda
        a_arr = np.zeros(N_lam_bf) # left boundary
        b_arr = np.zeros(N_lam_bf) # right boundary
        
        k = 0 # number zero crossings found so far 
                        
        while k < N_lam_bf:
            
            lam = lam - dlam
            J = self.integrator.calc_J_lb(lam, mu, sig)
            
            if np.sign(J) != sign_J:
                a_arr[k] = lam
                b_arr[k] = lam + dlam                                
                k += 1
                                
            sign_J = np.sign(J)

        #------------------------------------------------------------------------------ 
        # Determine eigenvalues by finding the roots in the brackets [a_arr, b_arr]        
        
        # Allocate output arrays            
        N_lam = self.solver_param['N_lam']
        
        full_lam_arr  = np.zeros(N_lam)
        full_J_lb_arr = np.zeros(N_lam)
                    
        for i, (a, b) in enumerate(zip(a_arr, b_arr)):
            
            # determine eigenvalu  by root finding root in (a,b)
            lam = brentq(self.integrator.calc_J_lb, a, b, args = (mu, sig), xtol = 1e-20)
            full_lam_arr[i]  = lam
            full_J_lb_arr[i] = self.integrator.calc_J_lb(lam, mu, sig)
            print(f'Found eigenvalue {i+1} by brute-force')

        #------------------------------------------------------------------------------ 
        # Guess values for remaining N_lam - k eigenvalues by linear extrapolation
                        
        # assume eigenvalues are linear function of n
        f_lam_n = lambda n, p0, p1: p0*n*+p1 
        
        # step size to bracket zero crossings is updated on the fly to increase performance
        dlam_n = dlam
        
        for i in np.arange(N_lam_bf, N_lam): #n is the mode number
                        
            # fit line through the last N_lam_bf eigenvalues  
            lam_arr = full_lam_arr[i-N_lam_bf:i]            
            n = i+1
            n_arr   = np.arange(n-N_lam_bf, n)
         
            popt, _ = curve_fit(f_lam_n, n_arr, lam_arr)
             
            p00 = popt[0]
            p10 = popt[1]
         
            # initial guess for next eigenvalue
            lam_n_guess = f_lam_n(n, p00, p10)
            # bracket zero crossing in the neighbourhood pf the initial guess 
            a,b = self.bracket_zc(lam_n_guess, dlam_n, mu, sig)
            # determine eigenvalu by finding the root in [a,b]  
            lam_n = brentq(self.integrator.calc_J_lb, a, b, args = (mu, sig), xtol = 1e-20)            
            # set step size on error between initial guess and true 
            dlam_n = np.abs(lam_n - lam_n_guess) 
            
            full_lam_arr[i]  = lam_n
            full_J_lb_arr[i] = self.integrator.calc_J_lb(lam_n, mu, sig)
            print(f'Found eigenvalue {n} by extrapolation')
                        
        return full_lam_arr, full_J_lb_arr  
                    
    def bracket_zc(self, lam0, dlam, mu, sig):
        '''Brackets zero crossing for initial guess lam0 by downhill search'''
                                
        # itentify down hill direction                         
        J_lb_0 = self.integrator.calc_J_lb(lam0, mu, sig)            
        J_lb_1 = self.integrator.calc_J_lb(lam0 + dlam, mu, sig)
        
        if np.sign(J_lb_0) != np.sign(J_lb_1): # found bracket in first step
            a = lam0
            b = lam0 + dlam
            return a,b
        elif np.abs(J_lb_0) > np.abs(J_lb_1): # downhill is right 
            dlam = dlam
        else: # downhill is left
            dlam = -dlam
           
        sign_J = np.sign(J_lb_0)                
        lam  = lam0
            
        while True:
            lam += dlam
            J = self.integrator.calc_J_lb(lam, mu, sig)
            
            if np.sign(J) != sign_J:
                if dlam < 0:
                    a = lam
                    b = lam - dlam
                else:
                    b = lam
                    a = lam - dlam 
                                    
                return a,b
        
        return 

    def compute_complex_ev_mu_sig(self, lam_arr_init, mu, sig):
        '''Compute eigenvalues for given mu and sig using lam_arr_init as an initial guess
        :param arr lam_arr_init: initial guess for eigenvalues
        :param float mu: mu 
        :param float sig: sig'''
        
        # allocate array for eigenvalues
        lam_arr = np.zeros_like(lam_arr_init, dtype = np.complex128)
        err_lam_arr = np.zeros_like(lam_arr_init, dtype = np.complex128)
                
        def f(lam, mu, sig):
        
            J_lb =  self.integrator.calc_J_lb(np.complex(lam[0], lam[1]), mu, sig)
            return np.array([J_lb.real, J_lb.imag])
                    
        for i, lam_n_init in enumerate(lam_arr_init):

            result = root(f, np.array([lam_n_init.real, lam_n_init.imag]), args = (mu, sig), method = 'hybr')
            lam = result.x 
            lam_arr[i] = np.complex(lam[0], lam[1])
            err_lam_arr[i] = lam_arr[i] - lam_n_init 
            print(f'Found eigenvalue {i+1} using 2d root finder')                 
        
        return lam_arr, err_lam_arr
                        
    def compute_complex_ev_mu_arr_sig(self, mu_min, mu_max, sig):
        '''Caculate eigenvalues for given mu in mu_arr and given sig
        :param arr mu_arr: mu array
        :param float sig: sigma'''
        
        mu_arr = np.arange(mu_min, mu_max, self.solver_param['dmu'])
        
        # allocate empty matrix to save eigenvalues
        lam_mat = np.zeros((self.solver_param['N_lam'], len(mu_arr)), dtype = np.complex128)                                
        
        # computate eigenvalues for mu_min assuming that spectrum is real
        lam_init_arr, _ = self.compute_real_ev(mu_arr[0], sig)        
        lam_mat[:, 0] = lam_init_arr
                        
        for i, mu in enumerate(mu_arr[1:]):
            
            lam_arr, err_lam_arr = self.compute_complex_ev_mu_sig(lam_init_arr, mu, sig)
            lam_init_arr = lam_arr + err_lam_arr
            lam_mat[:, i+1] = lam_arr
            print(f'Caculated eigenvalue for mu_{i+1}, {len(mu_arr)-i-2} left')
        
        # save
        filename = f'EV_mu_min_{mu_min:.2f}_mu_max_{mu_max:.2f}_sig_{sig:.2f}.h5'        
        data_path = './data/eigenvalues/' 
        
        with h5.File(data_path + filename, mode = 'a') as hf5:                   
        
            hf5['ev']     = lam_mat
            hf5['mu_arr'] = mu_arr
            hf5.attrs['sig'] = sig

        #------------------------------------------------------------------------------ 
        # if we know beforehand that the spectrum is real for mu < mu_real
        # we could speed up computation
#         mu_real = mu_arr[0]
# 
#         k = 1
#         mu = mu_arr[k]
#     
#         while mu < mu_real:
#                         
#             lam_arr = self.compute_real_ev(mu, sig)
#             lam_mat[k, :] = lam_arr 
#             
#             k += 1
#             mu = mu_arr[k] 
#         
#         for i, mu in enumerate(mu_arr[k:]):
        
    def scan_J_lb(self, mu, sig, lam_real_arr, lam_imag_arr):
        
        J_lb = np.zeros((len(lam_imag_arr), len(lam_real_arr)), dtype = np.complex128)
        
        for i, lam_imag in enumerate(lam_imag_arr):
            for j, lam_real in enumerate(lam_real_arr):
            
                lam = np.complex(lam_real, lam_imag)            
                J_lb[i,j] = self.integrator.calc_J_lb(lam, mu, sig)

        zc_real_1 = np.abs(np.diff(np.sign(J_lb.real), axis = 0))/2
        zc_real_2 = np.abs(np.diff(np.sign(J_lb.real), axis = 1))/2
        zc_real   = zc_real_1[:, :-1] + zc_real_2[:-1, :]
        
        zc_imag_1 = np.abs(np.diff(np.sign(J_lb.imag), axis = 0))/2
        zc_imag_2 = np.abs(np.diff(np.sign(J_lb.imag), axis = 1))/2
        zc_imag   = zc_imag_1[:, :-1] + zc_imag_2[:-1, :]

        blur_zc_real = gaussian_filter(zc_real, sigma = 1.0, mode = 'mirror')
        blur_zc_imag = gaussian_filter(zc_imag, sigma = 1.0, mode = 'mirror')
        
        roots = blur_zc_real*blur_zc_imag        
        roots[roots < 0.1*np.max(roots)] = 0

        root_idx_arr = self.calc_cluster_center(roots)
        
        lam_arr_init = np.zeros(len(root_idx_arr), dtype = np.complex)
        
        # convert indexes to points in complex plane
        for k, (i,j) in enumerate(root_idx_arr ):
        
            lam_arr_init[k] = np.complex(lam_real_arr[j], lam_imag_arr[i])
                            
        lam_arr = np.zeros_like(lam_arr_init)
        idx = np.argsort(np.abs(lam.real))
        lam_arr = lam_arr[idx]
                        
        # refine roots using root finder
        def f(lam, mu, sig):
        
            J_lb =  self.integrator.calc_J_lb(np.complex(lam[0], lam[1]), mu, sig)
            return np.array([J_lb.real, J_lb.imag])
                    
        for i, lam_n_init in enumerate(lam_arr_init):

            result = root(f, np.array([lam_n_init.real, lam_n_init.imag]), args = (mu, sig), method = 'hybr')
            lam = result.x             
            lam_arr[i] = np.complex(lam[0], lam[1]) 


        # save
        filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_numeric.h5'        
        data_path = './data/eigenvalues/' 
        
        with h5.File(data_path + filename, mode = 'a') as hf5:                   
        
            hf5['complex'] = lam_arr
            hf5.attrs['mu']  = mu
            hf5.attrs['sig'] = sig
                
        return

        
    def calc_cluster_center(self, roots):
        '''Caculates centers clusters in blured images'''

        idx_max = np.argmax(roots)
        i, j = np.unravel_index(idx_max, roots.shape)
            
        Nx = np.size(roots, axis = 1)
        Ny = np.size(roots, axis = 0)
                    
        cluster_max = roots[i, j]
        
        root_idx_arr = []
        
        w_x = 10
        w_y = 10

        while cluster_max > 0:
            
            i_start = i - w_y
            i_end   = i + w_y

            j_start = j - w_x
            j_end   = j + w_x

            if i_start < 0:
                i_start = 0
            if i_end > Ny - 1:
                i_end = Ny - 1
                
            if j_start < 0:
                j_start = 0
            if j_end > Nx - 1:
                j = Nx - 1
                
            cluster = roots[i_start:i_end, j_start:j_end]                
            cluster_size = np.sum(cluster > 0)

            roots[i_start:i_end, j_start:j_end] = 0 # delete cluster                    
                                                                        
            # check if cluster is real
            if cluster_size > 5: 
                root_idx_arr.append((i, j))
            
            idx_max = np.argmax(roots) # new cluster
            i, j = np.unravel_index(idx_max, roots.shape) # new index
            cluster_max = roots[i, j]
            
        return root_idx_arr

class Integrator():
        
    def __init__(self, model_param):
        '''Solves the eigenfunction equation of the Fokker-Planck operator and its adjoint operator for different one dimensional
        integrate-and-fire neuron models by numerical step-wise integration
        
        :param dict model_parameter: neuron model parameter 
        :param dict : model parameter: integration parameter'''
                
        self.model_param  = model_param
        self.solver_param = self.default_solver_parameter()
        self.model = model_param['model']
        
    def default_solver_parameter(self):
        '''Set default solver parameter'''
        
        N = int(1e4) # number of grid points on x-axis
        integrate_in = 'fortran' # either fortran or python 
        J_th = 1.0 # flux at threshold
        
        param = {}
        param['N'] = N         
        param['integrate_in'] = integrate_in
        param['J_th'] = J_th
        
        return param
    
    def set_parameter(self, **kwargs):
        '''Overwrites default solver parameter
        :param dict solver_param: dictionary with parameters'''
        
        for key, value in kwargs.items():            
            self.solver_param[key] = value
                                          
        return                              
    
    def get_A(self, lam, x):
        '''Get matrix A(x) for given model, x can be the membrane potential v or some variabel of v'''
                
        if self.model == 'PIF':
            pass
        elif self.model == 'QIF':
            pass
        elif self.model == 'LIF':
            A = np.array([[0, lam], [1, x]], dtype = np.complex128)                        
        elif self.model == 'EIF':
            Delta = 0.0
            x_th  = 0.0
            A = np.array([[0, lam], [1, x - Delta*np.exp((x-x_th)/Delta)]])               
        return A
    
    def get_expXdx_real(self, X, dx):

        s = 0.5 * (X[0,0] + X[1,1])
        det_X_minus_sI = (X[0,0]-s) * (X[1,1]-s)  -  X[0,1] * X[1,0]
        
        if det_X_minus_sI < 0:
            q = np.sqrt(-det_X_minus_sI)
        else:
            q = np.sqrt(det_X_minus_sI)
        # we have to distinguish the case q=0 (the other exceptional case t=0 should not occur)
        if abs(q) < 1e-15: # q is (numerically) 0
            sinh_qt_over_q = dx
        else:
            sinh_qt_over_q = np.sinh(q*dx) / q
        cosh_qt = np.cosh(q*dx)
        cosh_q_t_minus_s_sinh_qt_over_qt = cosh_qt - s*sinh_qt_over_q
        exp_st = np.exp(s*dx)
        
        expXdx = np.zeros((2,2), dtype = np.float)
        
        # abbreviations for the case of exp_Xt referencing the same array as X
        E_00 = exp_st * (cosh_q_t_minus_s_sinh_qt_over_qt   +   sinh_qt_over_q * X[0,0])
        E_01 = exp_st * (sinh_qt_over_q * X[0,1])
        E_10 = exp_st * (sinh_qt_over_q * X[1,0])
        E_11 = exp_st * (cosh_q_t_minus_s_sinh_qt_over_qt   +   sinh_qt_over_q * X[1,1])
        
        expXdx[0,0] = E_00
        expXdx[0,1] = E_01
        expXdx[1,0] = E_10
        expXdx[1,1] = E_11
        
        return expXdx
        
    def get_expXdx_complex(self, X, dx):
        '''Caculate matrix exponential exp(Xdx) using []
        :param arr X: matrix
        :param float dx: binwidth '''
        
        s = 0.5*(X[0,0] + X[1,1])
        q = np.sqrt(-(X[0,0]-s)*(X[1,1]-s) - X[0,1]*X[1,0])
        
        eps = 1.e-15
        
        if np.abs(q) < eps:
            sinh_qdx_over_q = dx
        else:
            sinh_qdx_over_q = np.sinh(q*dx)/q

        cosh_qdx = np.cosh(q*dx)
        cosh_qdx_minus_s_sinh_qdx_over_q = cosh_qdx - s*sinh_qdx_over_q
        exp_sdx = np.exp(s*dx)

        expXdx = np.zeros((2,2), dtype = np.complex)

        expXdx[0,0] = exp_sdx * (cosh_qdx_minus_s_sinh_qdx_over_q   +   sinh_qdx_over_q * X[0,0])
        expXdx[0,1] = exp_sdx * (sinh_qdx_over_q * X[0,1])
        expXdx[1,0] = exp_sdx * (sinh_qdx_over_q * X[1,0])
        expXdx[1,1] = exp_sdx * (cosh_qdx_minus_s_sinh_qdx_over_q   +   sinh_qdx_over_q * X[1,1])

        return expXdx
    
    def get_x_arr(self, v_arr, mu, sig):
        '''Get left and right boundary'''
                 
        if self.model in ['LIF', 'EIF']:
            x_arr = np.sqrt(2)*(v_arr - mu)/sig
                                    
        return x_arr
    
    def get_reset_idx(self, x_arr, mu, sig):
        '''Get index of grid point closest to reset value'''
                
        v_r = self.model_param['v_r']
        
        if self.model == 'LIF':
        
            x_r = np.sqrt(2)*(v_r - mu)/sig        
            idx_r = np.argmin(np.abs(x_arr - x_r))
        
            if x_arr[idx_r] > x_r:
                idx_r -= 1
        
        return idx_r

    def eigeneq_backwards_integration(self, lam, mu, sig):        
        '''Integrate eigenfuncions backward from the threshold for given initial condition of the probability current
        :param lam: eigenvalue
        :param mu: mean input
        :param sig: input fluctuations                
        :return real/complex arr phi: eigenfunction approximated at each grid point
        :return real/complex arr J: flux approximated at each grid point
        :return arr v_arr: memebrane potential grid'''
                                                         
        if self.solver_param['integrate_in'] == 'fortran':
           
            N = self.N
            
            J_th  = self.solver_param['J_th']
            v_min = self.model_param['v_min']
            v_th  = self.model_param['v_th']
            
            v_arr = np.linspace(v_min, v_th, N)
            x_arr = self.get_x_arr(v_arr, mu, sig)
 
            # find index of reset point                
            r_idx = self.get_reset_idx(x_arr, mu, sig)
                    
            if np.iscomplexobj(lam):                                
                J, phi = real_backward_integration(len(x_arr), lam, J_th, x_arr, r_idx)
            else:
                J, phi = complex_backward_integration(len(x_arr), lam, J_th, x_arr, r_idx)            
        
        elif self.solver_param['integrate_in'] == 'python':
            J, phi, v_arr = self.backward_integration(lam, mu, sig, J_th)
           
        return phi, J, v_arr
                                                  
    def calc_J_lb(self, lam, mu, sig):
        '''Caculate flux at left boundary for given mu and sigma by integrating eigenfuncioneq backwards
        :param complex/float lam: eigenvalue
        :param float mu: mean input
        :param float sig: input fluctuations  
        :param complex/float J_th: flux at threshold defaults to 1.0                
        :return float/complex J_lb: flux at the left boundary'''        
        
        if self.solver_param['integrate_in'] == 'fortran':        
            
            N     = self.solver_param['N'] 
            J_th  = self.solver_param['J_th']
            v_min = self.model_param['v_min']
            v_th  = self.model_param['v_th']
            v_arr = np.linspace(v_min, v_th, N)
                    
            x_arr = self.get_x_arr(v_arr, mu, sig)
            r_idx = self.get_reset_idx(x_arr, mu, sig)
                                    
            if np.iscomplexobj(lam):
                J_lb = calc_j_lb_complex(lam, J_th, x_arr, r_idx)
            else:
                J_lb = calc_j_lb_real(lam, J_th, x_arr, r_idx)
        
        elif self.solver_param['integrate_in'] == 'python':
                                                
            J_lb = self.backward_integration(lam, mu, sig, output = 'J_lb')
            
        return J_lb
                                                        
    def backward_integration(self, lam, mu, sig, output = 'all'):        
        '''Integrate eigenfuncions backward from the threshold for given initial condition of the probability current
        :param lam: eigenvalue
        :param mu: mean input
        :param sig: input fluctuations     
        :param str output: controls output can be one of ['full', 'J_lb']          
        '''                
        N     = self.solver_param['N']
        J_th  = self.solver_param['J_th']
        v_min = self.model_param['v_min']
        v_th  = self.model_param['v_th']
                
        v_arr = np.linspace(v_min, v_th, N)
        x_arr = self.get_x_arr(v_arr, mu, sig)
                    
        # find index of reset point                
        r_idx = self.get_reset_idx(x_arr, mu, sig)
                    
        if np.iscomplexobj(lam): # lam complex
            lam_type = np.complex128
            get_expAdx = self.get_expXdx_complex        
        else: # lam real
            lam_type = np.float
            get_expAdx = self.get_expXdx_real 

        A = np.array([[0, lam], [1, 0]], dtype = lam_type)
        
        if output == 'full':    
            phi = np.zeros_like(x_arr, dtype = lam_type) # eigenfunction
            J   = np.zeros_like(x_arr, dtype = lam_type) # probability current associated with eigenfunction                        
            phi[-1] = 0.0  # absorbing boundary at threshold
            J[-1]   = J_th # arbitrary complex scaling of the                                     
        
            for k in range(N-2, -1, -1): # k= Nv-2 , ... , 0
            
                dx   = x_arr[k+1]-x_arr[k]
                
                x = 0.5*(x_arr[k+1]+x_arr[k])
                
                A[1,1] = x
                
                expAdx = get_expAdx(A, dx)
                            
                J[k]   = expAdx[0,0]*J[k+1] + expAdx[0,1]*phi[k+1]
                phi[k] = expAdx[1,0]*J[k+1] + expAdx[1,1]*phi[k+1]
                
                # reinjection condition
                if k == r_idx:
                    J[k] = J[k] - J[N-1] 
    
            return phi, J, v_arr        
                
        elif output == 'J_lb':
            
            phi = 0.0
            J   = J_th
                                                                                                
            for k in range(N-2, -1, -1): # k= Nv-2 , ... , 0
            
                dx   = x_arr[k+1]-x_arr[k]
                
                x = 0.5*(x_arr[k+1]+x_arr[k])
                
                A[1,1] = x
                
                expAdx = get_expAdx(A, dx)
                            
                J   = expAdx[0,0]*J + expAdx[0,1]*phi
                phi = expAdx[1,0]*J + expAdx[1,1]*phi
                
                # reinjection condition
                if k == r_idx:
                    J = J - J_th
                    
            return J
        
        return

    def get_B(self, lam, x):
        '''Get matrix A(x) for given model, x can be the membrane potential v or some variabel
        of v'''
                
        if self.model == 'PIF':
            pass
        elif self.model == 'QIF':
            pass
        elif self.model == 'LIF':
            
            B = np.array([[lam, 1], [lam, x]], dtype = np.complex128)                        
        elif self.model == 'EIF':
            Delta = 0.
            x_th  = 0.
            f_x = x - Delta*np.exp((x-x_th)/Delta)
            B = np.array([[f_x, lam], [1, 0]], dtype = np.complex128)                        
                    
        return B

    def adjoint_eigeneq_forwards_magnus_general(self, lam, mu, sig, phi_tilde_min):
        
        N = int(1e4) # number of grid points

        v_min = self.model_param['v_min']
        v_th  = self.model_param['v_th']

        v_arr = np.linspace(v_min, v_th, N)
        x_arr = self.get_x_arr(v_arr, mu, sig)        
        
        # initialization
        phi_tilde = np.zeros_like(x_arr, dtype = np.complex128) # eigenfunction
        J_tilde   = np.zeros_like(x_arr, dtype = np.complex128) # probability current associated with eigenfunction
                
        phi_tilde[0] = phi_tilde_min + 0.0j # linear eq = arbitrary complex scaling
        J_tilde[0] = 0.0 + 0.0j # left boundary condition of adjoint operator
                        
        dx = x_arr[1] - x_arr[0]
                        
        # exponential magnus integration
        for k in range(1, N):

            # grid spacing, allowing non-uniform grids
            dx   = x_arr[k]-x_arr[k-1]
            x = 0.5*(x_arr[k]+x_arr[k-1])

            B = self.get_B(lam, x)
                        
            expBdx = self.get_expXdx(B, dx) # expBdx contains now exp(A*dV)
            # (numba-njit-compatible) equivalent to expBdx = scipy.linalg.expm(A*dV)
    
            # the following computes the matrix vector product exp(A*dV) * [phi_tilde, J_tilde]^T
            phi_tilde[k] = expBdx[0,0]*phi_tilde[k-1] + expBdx[0,1]*J_tilde[k-1]
            J_tilde[k]   = expBdx[1,0]*phi_tilde[k-1] + expBdx[1,1]*J_tilde[k-1]
            # that's it. no discontinuous reinjection => instead in characteristic eq.
        
        return phi_tilde, J_tilde, v_arr

def solver_parameter():

    J_th         = np.sqrt(2.0/np.pi) # initial value for flux at threshold
    N_lam        = 20 # number of grid points on which the solver looks for eigenvalues on the real line between [0, real_lam_min] 

    param = {}
    param['J_th']         = J_th
    param['N_lam']        = N_lam
        
    return param
    
def model_parameter():

    #===============================================================================
    # neuron model
    v_th    = 1.0 # threshold
    v_reset = 0.0 # reset   
    v_rev   = 0.0 # reversal
    v_min   = -4*v_th # minimum
    #===============================================================================

    param = {}
    param['model']   = 'LIF'
    param['v_th']    = v_th 
    param['v_r']     = v_reset
    param['v_rev']   = v_rev
    param['v_min']   = v_min
    
    return param


def inner_product(mu, sig, J_th, phi_tilde_min):
    
    import matplotlib.pyplot as plt
    
    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    mu  = file.attrs['mu']
    sig = file.attrs['sig']
    
    ev_complex = file['complex'][:]
    ev_real = file['real'][:]

    param = model_parameter()
    
    integrator = IF_Backward_Integrator(param)

#     diffusion_param = param.copy()
#     diffusion_param['mu']  = mu
#     diffusion_param['sig'] = sig
#     
#     theory = LIF_Theory(diffusion_param = diffusion_param
    M = 5

    phi_tilde_arr = []
    phi_arr   = []

    for i, lam in enumerate(ev_complex[0:M]):
                    
        phi_tilde, J, v_arr  = integrator.adjoint_eigeneq_forwards_magnus_general(lam, mu, sig, phi_tilde_min)
        phi, J, v_arr  = integrator.eigeneq_backwards_magnus_general(lam, mu, sig, J_th)
        
        phi_tilde_arr.append(phi_tilde)
        phi_arr.append(phi)

        
    dv = v_arr[1] - v_arr[0]
        
    ip = np.zeros((M,M), dtype = np.complex128)
        
    for i in range(M):
        for j in range(M):            
            phi_tilde = phi_tilde_arr[i]
            phi       = phi_arr[j]
            
            ip[i,j] = np.sum(phi_tilde*phi)*dv

    plt.imshow(np.abs(ip))
    plt.colorbar()
    plt.show()
        
    return
            

def plot_phi_tilde(mu, sig, phi_tilde_min):

    import matplotlib.pyplot as plt
    
    fz = 18
    
    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    mu  = file.attrs['mu']
    sig = file.attrs['sig']
    
    ev_complex = file['complex'][:]
    ev_real = file['real'][:]

    param = model_parameter()
    
    integrator = IF_Backward_Integrator(param)

#     diffusion_param = param.copy()
#     diffusion_param['mu']  = mu
#     diffusion_param['sig'] = sig
#     
#     theory = LIF_Theory(diffusion_param = diffusion_param
    M = 5

    gs = plt.GridSpec(M, 2)

    for i, lam in enumerate(ev_complex[0:M]):
    

        ax_i0 = plt.subplot(gs[i, 0])
                
        phi_tilde_0, _, v_arr  = integrator.adjoint_eigeneq_forwards_magnus_general(lam, mu, sig, phi_tilde_min)
#         phi1 = theory.phi_n(i+1, 'complex', v_arr)
        
        ax_i0.plot(v_arr, phi_tilde_0.real, ls = '-', c = 'k', label = 'theory')
#         ax_i0.plot(v_arr, phi1.real, ls = '--', c = 'r', label = 'numeric') 

        ax_i0.set_ylabel(f'Real$(\phi_{i+1})$', fontsize = fz)

        ax_i1 = plt.subplot(gs[i, 1])

        ax_i1.plot(v_arr, phi_tilde_0.imag, ls = '-', c = 'k', label = 'theory')
#         ax_i1.plot(v_arr, phi1.imag, ls = '--', c = 'r', label = 'numeric') 

        ax_i1.set_ylabel(f'Imag$(\phi_{i+1})$', fontsize = fz)

        if i == M - 1:
            ax_i0.set_xlabel(f'$v$', fontsize = fz)
            ax_i1.set_xlabel(f'$v$', fontsize = fz)

    plt.show()



def scan_real_ev(mu,sig):

    import matplotlib.pyplot as plt
    
    fz = 18
    
    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    mu  = file.attrs['mu']
    sig = file.attrs['sig']
    
    ev_real = file['real'][1:]

    solver_param = solver_parameter()
    model_param  = model_parameter()
    
    spectral_solver = Spectral_Solver(solver_param, model_param)
    spectral_solver.mu  = mu
    spectral_solver.sig = sig
    
    eps = 0.01
    
    
    lam_arr = np.linspace(-0.1, ev_real[-1], 500)    
    spectral_solver.compute_real_ev(lam_arr)
    
def compare_phi_to_theory_real(mu, sig, J_th):

    import matplotlib.pyplot as plt
    
    fz = 18
    
    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    mu  = file.attrs['mu']
    sig = file.attrs['sig']
    
    ev_complex = file['complex'][:]
    ev_real = file['real'][1:]

    solver_param = solver_parameter()
    model_param  = model_parameter()
    integrator = Integrator(solver_param, model_param)

    diffusion_param = model_param.copy()
    diffusion_param['mu']  = mu
    diffusion_param['sig'] = sig
    
    theory = LIF_Theory(diffusion_param = diffusion_param)

    # plot first M eigenfunctions
    M = 2

    gs = plt.GridSpec(M, 2)

    for i, lam in enumerate(ev_real[0:M]):
    
        ax_i0 = plt.subplot(gs[i, 0])
                
        phi0, J, v_arr  = integrator.eigeneq_backwards_magnus_general(lam, mu, sig)
        print(J[0])
        print(integrator.eigeneq_backwards_magnus_general_fast(lam, mu, sig))
        
        phi1 = theory.phi_n(i+1, 'real', v_arr)
        
        ax_i0.plot(v_arr, phi0.real, ls = '-', c = 'k', label = 'theory')
        ax_i0.plot(v_arr, phi1.real, ls = '--', c = 'r', label = 'numeric') 

        ax_i0.set_ylabel(f'Real$(\phi_{i+1})$', fontsize = fz)

        ax_i1 = plt.subplot(gs[i, 1])

        ax_i1.plot(v_arr, phi0.imag, ls = '-', c = 'k', label = 'theory')
        ax_i1.plot(v_arr, phi1.imag, ls = '--', c = 'r', label = 'numeric') 

        ax_i1.set_ylabel(f'Imag$(\phi_{i+1})$', fontsize = fz)

        if i == M - 1:
            ax_i0.set_xlabel(f'$v$', fontsize = fz)
            ax_i1.set_xlabel(f'$v$', fontsize = fz)
            
            ax_i0.legend()
        
#         ax.plot(ev_complex.real, ev_complex.imag, 'o', ms = 5.0, mfc = 'k', mec = 'k')
#         ax.plot(ev_complex.real, -ev_complex.imag, 'o', ms = 5.0, mfc = 'k', mec = 'k')
#         ax.plot(ev_real, np.zeros_like(ev_real), 'o', ms = 5.0, mfc = 'k', mec = 'k')
#     
#         ax.set_ylabel(u'Imag($\lambda$)', fontsize = 11)
#         ax.set_xlabel(u'Re($\lambda$)', fontsize = 11)
#         ax.grid(True)
    
    plt.show()


def compare_real_ev_to_theory(mu, sig):

    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    M = 20

    lam_real_theory = file['real'][1:M]
        
    model_param  = model_parameter()
       
    solver = Spectral_Solver(model_param)
    solver.set_solver_parameter(N_lam = M-1)
    solver.set_integrator_parameter(J_lb = np.sqrt(2./np.pi))
                                    
    lam_fortran, _ = solver.compute_real_ev(mu, sig)
    
    solver.set_integrator_parameter(integrate_in = 'python')

    lam_python, _ = solver.compute_real_ev(mu, sig)

    n_arr = np.arange(1, M)

    gs = plt.GridSpec(2, 1)
    fz  = 18
    lfz = 16
        
    # compare eigenvalues to theory
    ax0 = plt.subplot(gs[0])
    ax0.plot(n_arr, lam_real_theory, 'o', label = 'Theory')
    ax0.plot(n_arr, lam_python, 'x', label = 'python')
    ax0.plot(n_arr, lam_fortran, '*', label = 'fortran')
    ax0.set_ylabel('$\lambda$', fontsize = fz)
    
    ax0.legend(fontsize = lfz)

    # plot error
    ax1 = plt.subplot(gs[1])    
    
    err_python  =  lam_real_theory - lam_python
    err_fortran =  lam_real_theory - lam_fortran

    ax1.plot(n_arr, err_python, label = 'theory - python')
    ax1.plot(n_arr, err_fortran, label = 'theory - fortran')

    ax1.set_ylabel('error', fontsize = fz)
    ax1.set_xlabel('$n$', fontsize = fz)
    ax1.legend(fontsize = lfz)
    
    plt.show()
    
    return

def text_complex_ev():
    
    model_param  = model_parameter()
       
    solver = Spectral_Solver(model_param)
    solver.set_integrator_parameter(J_lb = np.sqrt(2./np.pi))
        
    mu_min = 0.2
    sig    = 0.2
    mu_max = 0.8
        
    solver.compute_complex_ev_mu_arr_sig(mu_min, mu_max, sig)
    
def plot_spectrum_theory():
    
    mu  = 0.64
    sig = 0.2
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'       
    data_path = './data/eigenvalues/' 
    
    file = h5.File(data_path + filename, 'r')
    
    ev_real    = file['real'][:]
    ev_complex = file['complex'][:]
    
    plt.plot(ev_real, np.zeros_like(ev_real), 'o', mfc = 'k', mec = 'k')
    plt.plot(ev_complex.real, ev_complex.imag, 'o', mfc = 'k', mec = 'k')

    mu_min = 0.2
    sig    = 0.2
    mu_max = 0.8

    filename = f'EV_mu_min_{mu_min:.2f}_mu_max_{mu_max:.2f}_sig_{sig:.2f}.h5'        
    
    file = h5.File(data_path + filename, 'r')
    mu_arr = file['mu_arr'][:]
    
    idx = np.argmin(np.abs(mu_arr - mu))
    
    ev = file['ev'][:, idx]
    
    plt.plot(ev.real, ev.imag, 'x')
    
    plt.show()
    
def scan_J_lb():    

    model_param  = model_parameter()
    
    solver = Spectral_Solver(model_param)
    solver.set_integrator_parameter(J_lb = np.sqrt(2./np.pi))
    
    mu  = 0.64
    sig = 0.2
    
    lam_real_arr = np.linspace(-15, -0.1, int(6e2))
    lam_imag_arr = np.linspace(0, 5, int(2e2))
    
    solver.scan_J_lb(mu, sig, lam_real_arr, lam_imag_arr)
    
    
    
def plot_complex_ev(mu_min, sig, mu_max):

    filename = f'EV_mu_min_{mu_min:.2f}_mu_max_{mu_max:.2f}_sig_{sig:.2f}.h5'        
    data_path = './data/eigenvalues/' 
    
    f = h5.File(data_path + filename)

    mu_arr  = f['mu_arr'][:]
    lam_mat = f['ev'][:]    

    N = 3
    
    lam_mat = lam_mat[:N, :]

    gs = plt.GridSpec(3, 2)
    
    for i in range(N):
        
        axi0 = plt.subplot(gs[i, 0])
        axi0.plot(mu_arr, np.real(lam_mat[i, :]), 'o')
        axi1 = plt.subplot(gs[i, 1])        
        axi1.plot(mu_arr, np.imag(lam_mat[i, :]), 'o')

    plt.show()
    
    return
    
def compare_complex_ev_to_theory(mu, sig):

    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    ev0 = file['complex'][:]
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_numeric.h5'
    file = h5.File(data_path + filename, 'r')

    ev1 = file['complex'][:]
    
    ev1 = ev1[ev1.imag != 0]
    
    ev1 = ev1[np.argsort(np.abs(ev1.real))]
    
    
    gs = plt.GridSpec(3, 1)
    
    ax0 = plt.subplot(gs[0])
    
    ax0.plot(ev0.real, ev0.imag, 'o')
    ax0.plot(ev1.real, ev1.imag, 'x')

    n0 = np.arange(1, len(ev0)+0.1)
    n1 = np.arange(1, len(ev1)+0.1)

    f = lambda n, p0, p1: p1*n**2 + p0    
    popt, _ = curve_fit(f, n1, -ev1.real)
    
    ax1 = plt.subplot(gs[1])
    x = np.linspace(0, n1[-1], int(1e3))    
    ax1.plot(x, f(x, popt[0], popt[1]), '--')
    
    ax1.plot(n0, -ev0.real, 'o')
    ax1.plot(n1, -ev1.real, 'x')
    
    
    ax2 = plt.subplot(gs[2])
    ax2.plot(n0, ev0.imag, 'o')
    ax2.plot(n1, ev1.imag, 'x')
    
    plt.show()

def theory_real_ev(mu, sig):
    
    model_param  = model_parameter()

    diffusion_param = model_param.copy()
    diffusion_param['mu']  = mu
    diffusion_param['sig'] = sig
    
    theory = LIF_Theory(diffusion_param = diffusion_param)
    
    min_real = -30.
    
    theory.EV_brute_force(min_real, comp = False)
    
    return


def plot_real_ev(mu, sig):

    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')

    ev_real = file['real'][1:]
    n_arr = np.arange(1, len(ev_real)+0.1)

    plt.plot(n_arr, ev_real, 'o')

    f = lambda n, a, b: a*n+b

    popt, _ = curve_fit(f, n_arr[:4], ev_real[:4])

    a0 = popt[0]
    b0 = popt[1]
    
    x_arr = np.linspace(1, n_arr[-1], int(1e3))
    
    plt.plot(x_arr, f(x_arr, a0, b0), '-')
    
    plt.show()

def compare_phi_to_theory(mu, sig, J_th):

    
    fz = 18
    
    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    mu  = file.attrs['mu']
    sig = file.attrs['sig']
    
    ev_complex = file['complex'][:]
    ev_real = file['real'][:]

    param = model_parameter()
    integrator = IF_Backward_Integrator(param)

    diffusion_param = param.copy()
    diffusion_param['mu']  = mu
    diffusion_param['sig'] = sig
    
    theory = LIF_Theory(diffusion_param = diffusion_param)

    # plot first M eigenfunctions
    M = 5

    gs = plt.GridSpec(M, 2)

    for i, lam in enumerate(ev_complex[0:M]):
    
        ax_i0 = plt.subplot(gs[i, 0])
                
        phi0, J, v_arr  = integrator.eigeneq_backwards_magnus_general(lam, mu, sig, J_th)
        phi1 = theory.phi_n(i+1, 'complex', v_arr)
        
        ax_i0.plot(v_arr, phi0.real, ls = '-', c = 'k', label = 'theory')
        ax_i0.plot(v_arr, phi1.real, ls = '--', c = 'r', label = 'numeric') 

        ax_i0.set_ylabel(f'Real$(\phi_{i+1})$', fontsize = fz)

        ax_i1 = plt.subplot(gs[i, 1])

        ax_i1.plot(v_arr, phi0.imag, ls = '-', c = 'k', label = 'theory')
        ax_i1.plot(v_arr, phi1.imag, ls = '--', c = 'r', label = 'numeric') 

        ax_i1.set_ylabel(f'Imag$(\phi_{i+1})$', fontsize = fz)

        if i == M - 1:
            ax_i0.set_xlabel(f'$v$', fontsize = fz)
            ax_i1.set_xlabel(f'$v$', fontsize = fz)
            
            ax_i0.legend()
        
#         ax.plot(ev_complex.real, ev_complex.imag, 'o', ms = 5.0, mfc = 'k', mec = 'k')
#         ax.plot(ev_complex.real, -ev_complex.imag, 'o', ms = 5.0, mfc = 'k', mec = 'k')
#         ax.plot(ev_real, np.zeros_like(ev_real), 'o', ms = 5.0, mfc = 'k', mec = 'k')
#     
#         ax.set_ylabel(u'Imag($\lambda$)', fontsize = 11)
#         ax.set_xlabel(u'Re($\lambda$)', fontsize = 11)
#         ax.grid(True)
    
    plt.show()

def plot_ev_numeric(mu_min, mu_max, sig):

    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_min_{mu_min:.2f}_mu_max_{mu_max:.2f}_sig_{sig:.2f}.h5'
    
    file = h5.File(data_path + filename, 'r')

    mu_arr = file['mu_arr'][:]    
    ev     = file['ev'][:]
    
    # differen mu's
    M = 100
    # first N modes
    N = 5

    gs  = plt.GridSpec(2,1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
                                 
    for j in range(N):
    
        ax0.plot(mu_arr[:M], ev[j, :M].real, 'o')
        ax1.plot(mu_arr[:M], ev[j, :M].imag, 'o')
                
    plt.show()
            
    return
        
def test_spectral_solver(mu, sig):
    
    model_param = model_parameter()
    solver_param = solver_parameter()
    
    ss = Spectral_Solver(solver_param, model_param)
    
    ss.mu  = mu 
    ss.sig = sig
    ss.compute_real_ev()
    
def check_fortran():    

    mu  = 1.2
    sig = 0.2
    lam = 0.0

    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    ev_complex = file['complex'][:]
    
    mu  = file.attrs['mu']
    sig = file.attrs['sig']

    sovler_param = solver_parameter()
    model_param  = model_parameter()
    
    integrator   = Integrator(sovler_param, model_param)

    M = 4
    
    gs = plt.GridSpec(M, 2)
    
    fz  = 18
    lfz = 16
    
    for i, lam in enumerate(ev_complex[:M]):

        phi_f, _, v_arr_f = integrator.eigeneq_backwards_magnus_general_fortran(lam, mu, sig)
        phi, _, v_arr = integrator.eigeneq_backwards_magnus_general(lam, mu, sig)
        
        ax_i0 = plt.subplot(gs[i,0])
        ax_i1 = plt.subplot(gs[i,1])
                
        ax_i0.plot(v_arr_f, phi_f.real, c = 'k', label = 'fortran')
        ax_i0.plot(v_arr, phi.real, c = 'r', ls = '--', label = 'python')
        ax_i0.set_ylabel('Real($\phi$)', fontsize = fz)

        ax_i1.plot(v_arr_f, phi_f.imag, c = 'k', label = 'fortran')
        ax_i1.plot(v_arr, phi.imag, c = 'r', ls = '--', label = 'python')
    
        ax_i1.set_ylabel('Imag($\phi$)', fontsize = fz)
    
        if i == M-1:
            ax_i0.set_xlabel('v')
            ax_i1.set_xlabel('v')
            ax_i0.legend(fontsize = lfz) 
            
    plt.legend()
    plt.show()

def time_fortran():

    sovler_param = solver_parameter()
    model_param = model_parameter()
    integrator = Integrator(sovler_param, model_param)

    setup = '''from __main__ import Integrator, solver_parameter, model_parameter
sovler_param = solver_parameter()
model_param = model_parameter()
integrator = Integrator(sovler_param, model_param)

mu  = 1.2
sig = 0.2
lam = 0.0 + 0.0j    
'''

    mycode = '''integrator.eigeneq_backwards_magnus_general_fortran(lam, mu, sig)'''
    
    N = 100

    print('Average run time fortran')    
    print(timeit(stmt = mycode, setup = setup, number  = N)/N)

    mycode = '''integrator.eigeneq_backwards_magnus_general(lam, mu, sig)'''

    print('Average run time python')    
    print(timeit(stmt = mycode, setup = setup, number  = N)/N)
    
if __name__ == '__main__':
    
#     mu   = 1.2
#     sig  = 0.20    
#     J_th = np.sqrt(2./np.pi) 
#     phi_tilde_min = 0.1
#     plot_phi_tilde(mu, sig, phi_tilde_min)
#     compare_phi_to_theory(mu, sig, J_th)
#     inner_product(mu, sig, J_th, phi_tilde_min)
#     test_spectral_solver()
#     compare_phi_to_theory_real(mu, sig, J_th)
#     scan_real_ev(mu, sig)
#     mu  = 0.2
#     sig = 0.2
#     test_spectral_solver(mu, sig)
#     theory_real_ev(mu, sig)
#     plot_real_ev(mu, sig)
#     compare_real_ev_to_theory(mu, sig)
#     text_complex_ev()

#     mu_min = 0.2
#     sig    = 0.2
#     mu_max = 0.8

#     plot_complex_ev(mu_min, sig, mu_max)
#     plot_ev_numeric(mu_min, mu_max, sig)
#     check_fortran()
#     wrap_fortran()
#     time_fortran()
#     check_fortran()
#     plot_spectrum_theory()
    
#     scan_J_lb()
    mu  = 0.64
    sig = 0.2
    compare_complex_ev_to_theory(mu, sig)
    
    
    print('Finished')

