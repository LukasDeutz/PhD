import os
import numpy as np
from scipy.optimize import brentq, curve_fit, root, minimize, brute
from scipy.ndimage import gaussian_filter
import h5py as h5
import multiprocessing as mp
import functools
import tqdm

# fortran for speed up
from backward_integration import complex_backward_integration, real_backward_integration
from calculate_J_lb import calc_j_lb_complex, calc_j_lb_real


class Spectral_Solver():
    '''Finds eigenfunctions and eigenvalues for the Fokker-Planck operator of a given one dimensional integrate-and-fire neuron.'''
    
    
    def __init__(self, model_param):
        
        self.model_param  = model_param
        self.solver_param = self.default_solver_parameter()
        self.root_finder  = self.default_root_finder()
#         self.init_J_lb_funcs()
                
        self.integrator = Integrator(self.model_param)


    def default_solver_parameter(self):
                
        lam_max  = -0.1 # maximal real part of eigenvalues 
        dlam     = 0.05 # stepsize to find eigenvalues on real line
        dmu      = 5e-3 # grid spacing on mu axis
        dsig     = 5e-3 # grid spacing on sig axis
        N_lam    = 10 # number of eigenvalues to be found                
        lam_real_min = -30
        eps_imag = 1e-4
                                                           
        param = {}
        
        param['lam_real_min'] = lam_real_min # find eigenvalues with larger real part 
        param['N_lam'] = N_lam
                                                                 
        param['dlam']     = dlam
        param['lam_max']  = lam_max
        param['dmu']      = dmu
        param['dsig']     = dsig    
        param['eps_imag'] = eps_imag 


                
#         N_lam_bf = 3 #solver looks for the first 3 eigenvalues on the real line brute force        
#         param['N_lam_bf']   = N_lam_bf
#         param['N_lam_comp'] = N_lam
        
        return param
        
    def default_root_finder(self):
        
        # root finder
        root_finder = {}
        root_finder['method'] = 'SLSQP' # minimization method        
        root_finder['kwargs'] = {}
        root_finder['param']  = {'bounding_box': [0.1, 0.1]} # bounding box around initial value
        
        return root_finder
    
    def set_solver_parameter(self, **kwargs):
        '''Overwrites default parameters
        :param dict solver_param: dictionary with parameters'''
        
        for key, value in kwargs.items():
            
            self.solver_param[key] = value
        
        return

    def log_abs_J_lb(self, lam, mu, sig):

        return np.log(np.abs(self.integrator.calc_J_lb(np.complex(lam[0], lam[1]), mu, sig)))
        
    def J_lb_comp(self, lam, mu, sig):
        
        J_lb = self.integrator.calc_J_lb(lam, mu, sig)
        
        return np.array([J_lb.real, J_lb.imag])
            
#     def init_J_lb_funcs(self):
#         
#         self.J_lb_real = lambda lam, mu, sig: self.integrator.calc_J_lb(lam, mu, sig)        
#         self.J_lb_abs  = lambda lam, mu, sig: np.abs(self.integrator.calc_J_lb(lam, mu, sig))        
#         self.log_abs_J_lb = lambda lam, mu, sig: np.log(np.abs(self.integrator.calc_J_lb(np.complex(lam[0], lam[1]), mu, sig)))
#                                 
#         def J_lb_comp(lam, mu, sig):
#             
#             J_lb = self.integrator.calc_J_lb(lam, mu, sig)
#             
#             return np.array([J_lb.real, J_lb.imag])
#         
#         self.J_lb_comp = J_lb_comp
# 
#         return
        
    def set_root_finder(self, **kwargs):
        '''Overwrites default of the root finder
        :param dict solver_param: dictionary with parameters'''

        for key, value in kwargs.items():
            
            self.root_finder[key] = value
        
        return
        
    def set_integrator_parameter(self, **kwargs):
        '''Overwrites default parameters of the integrator object
        :param dict solver_param: dictionary with parameters'''
        
        self.integrator.set_parameter(**kwargs)
        
        return
                                    
    def compute_real_ev(self, mu, sig, N_lam = None):
        '''Caculate eigenvalues on the real line given sig and mu. It is assumed that mu is 
        choosen small enough to ensure that the spectrum of the FP operator is real.
        :param float mu: mu
        :param float sig: sigma
        :param int N: stops after N eigenvalues are found'''
        
        #------------------------------------------------------------------------------ 
        # Find first eigenvalues on real line brute-force  
        if N_lam is None:
            N_lam   = self.solver_param['N_lam']               
        lam     = self.solver_param['lam_max'] # start value
        dlam    = self.solver_param['dlam'] # step size
        
        # initial flux at left boundary
        J_lb = self.integrator.calc_J_lb(lam, mu, sig)
        sign_J = np.sign(J_lb)

        # bracket zero crossings of J_lb as a function of lambda
        a_arr = np.zeros(N_lam) # left boundary
        b_arr = np.zeros(N_lam) # right boundary
          
        k = 0                                
        while k < N_lam:
            
            lam = lam - dlam
            J = self.integrator.calc_J_lb(lam, mu, sig)
            
            if np.sign(J) != sign_J:
                a_arr[k] = lam
                b_arr[k] = lam + dlam                                
                k +=1
            
            sign_J = np.sign(J)
            
        #------------------------------------------------------------------------------ 
        # Determine eigenvalues by finding the roots in the brackets [a_arr, b_arr]        
        lam_arr = np.zeros(len(a_arr))
                
        for i, (a, b) in enumerate(zip(a_arr, b_arr)):
            
            # determine eigenvalu  by root finding root in (a,b)
            lam = brentq(self.integrator.calc_J_lb, a, b, args = (mu, sig))
            lam_arr[i]  = lam
#             print(f'Found eigenvalue {i+1} on the real line by brute-force')
        
        return lam_arr


    def scan_J_lb(self, 
                  mu, 
                  sig, 
                  lam0,
                  lam1,
                  lam_real_min, 
                  lam_real_max,
                  lam_imag_min,
                  lam_imag_max,
                  dlam = None):    
        
        if dlam is None:
            dlam = 0.01

        eps = 1e-8
                
        lam_real_arr = np.arange(lam_real_min, lam_real_max + 0.1*dlam, dlam)
        lam_imag_arr = np.arange(lam_imag_min, lam_imag_max + 0.1*dlam, dlam)
        
        J_lb = np.zeros((len(lam_imag_arr), len(lam_real_arr)), dtype = np.complex128)
        
        import matplotlib.pyplot as plt
        
        for i, lam_imag in enumerate(lam_imag_arr):
            for j, lam_real in enumerate(lam_real_arr):
                
                if np.abs(lam_imag) < eps:
#                     lam = lam_real
                    lam = np.complex(lam_real, 0)
                else:
                    lam = np.complex(lam_real, lam_imag)            
                J_lb[i,j] = self.integrator.calc_J_lb(lam, mu, sig)
                
            if np.abs(lam_imag) < eps:
                print('Test')

                
        fig_path = 'figures/ev_real_2_complex/test/energy/'

        
        fig = plt.figure()
        plt.imshow(np.log(np.abs(J_lb)), extent=[lam_real_min,lam_real_max,lam_imag_min, lam_imag_max])
        plt.colorbar()
        
        plt.plot(lam0.real, lam0.imag, 'x')
        plt.plot(lam1.real, lam1.imag, 'x')
        
        plt.savefig(fig_path + f'energy_lam_real_max_{lam_real_max:.2f}_lam_real_min_{lam_real_min:.2f}_mu_{mu:.4f}.png')
                
        plt.close(fig)
                
        return

    def save_lam_mat(self, lam_mat, mu_arr, sig_arr):
        '''Save matrix with eigenvalues'''
                                
        N_lam = np.size(lam_mat, 2)                
        
        data_path = './data/eigenvalues/'
        filename = f'ev_mu_min_{mu_arr[0]:.2f}_mu_max_{mu_arr[-1]:.2f}_sig_min_{sig_arr[0]:.2f}_sig_max_{sig_arr[-1]:.2f}_N_{N_lam}.h5' 
                                         
                                                            
        with h5.File(data_path + filename, mode = 'w') as file:                   
        
            file['lam_mat'] = lam_mat
            file['mu_arr']  = mu_arr
            file['sig_arr'] = sig_arr
            file.attrs['mu_min']  = mu_arr[0]
            file.attrs['mu_max']  = mu_arr[-1]
            file.attrs['sig_min'] = sig_arr[0]
            file.attrs['sig_max'] = sig_arr[-1]
            file.attrs['N_lam']   = N_lam
            
        return

    def get_lam_mat(self, N_lam, mu_min, mu_max, sig_min, sig_max):

        data_path = './data/eigenvalues/'
        filename = f'ev_mu_min_{mu_min:.2f}_mu_max_{mu_max:.2f}_sig_min_{sig_min:.2f}_sig_max_{sig_max}.h5'
                                            
        if os.path.isfile(data_path + filename):
            f = h5.File(data_path + filename)
            if N_lam > f.attrs['N_lam']:
                self.compute_real_2_complex_ev_parallel(N_lam, mu_min, mu_max, sig_min, sig_max, N_skip = f.attrs['N_lam'])
                

    def compute_real_2_complex_ev_parallel(self, N_lam, mu_min, mu_max, sig_min, sig_max, N_skip = 0):
        '''Computes the first N eigenvalues for all grid points on a 2 dimensional grid  
        between [mu_min, mu_max] and [mu_min, mu_max]. 
        :param int N_lam: number of eigenvalues to be found
        :param float mu_min: minimum mean input. Must be choosen small enough to ensure that all eigenvalues are real for all sigma
        :param float mu_max: maximum mean input
        :param float sig_min: minimum standard diviation of the input
        :param float sig_max: maximum stabdard diviation of the input
        :param int N_skip: First N_skip eigenvalues are skipped. Defaults to zero. Should be used to extend an existing file with eigenvalues                 
        
        :return arr lam_mat: matrix with eigenvalues at each grid point'''

        dmu  = self.solver_param['dmu'] # step size mu
        dsig = self.solver_param['dsig'] # step size sig

        mu_arr = np.arange(mu_min, mu_max + 0.1*dmu, dmu)
        sig_arr = np.arange(sig_min, sig_max + 0.1*dsig, dsig)

#         N_sig = len(sig_arr)
#         N_mu  = len(mu_arr)

        args = [(N_lam, mu_arr, sig, N_skip) for i, sig in enumerate(sig_arr)]

        with mp.Pool(mp.cpu_count()) as p:

            lam_mat = list(tqdm.tqdm(p.imap(self.wrapper, args), total=len(args)))
            lam_mat = np.array(lam_mat)

            self.save_lam_mat(lam_mat, mu_arr, sig_arr)


#         print(f'Number of jobs: {len(sig_arr)}')        
#         lam_mat = np.vstack(lam_mat)
                                                      
#         lam_mat = np.zeros((N_sig, N_mu, N_lam - N_skip), dtype = np.complex)
#                                 
#         for i, sig in enumerate(sig_arr):
#             
#             lam_mat[i, :, :] = self.compute_real_2_complex_ev_sig(N_lam, mu_arr, sig, N_skip)
#                 
                
        return
    
    def wrapper(self, tup):
        
        return self.compute_real_2_complex_ev_sig(*tup)
    
    def compute_real_2_complex_ev_sig(self, N_lam, mu_arr, sig, N_skip):
        '''Compute the first N_lam eigenvalues for all mu in mu_arr given while sigma is fixed
        :param int N_lam: number of eigenvalues
        :param arr mu_arr: mean input 
        :param sig: variance of the input''' 
        
#         if i is not None:
#             print(mp.current_process())
#             print(f'Job {i} running')
        
        N_lam = N_lam + 1
                        
        # initial eigenvalues 
        lam_arr = self.compute_real_ev(mu_arr[0], sig, N_lam) # real part         
        lam_arr = lam_arr.astype(np.complex) 
        
        if N_skip != 0:
            lam_arr = lam_arr[N_skip-1:]
            N_lam = len(lam_arr)
                
        lam_mat = np.zeros((len(mu_arr), N_lam), dtype = np.complex) # eigenvalues 
                                        
        # Eigenvalues start on the real line and team up to form complex conjugate pairs
        # when mu increases. This means that as long as eigenvalues are far apart, we can 
        # be sure they stay on the real line if mu is increasesed by one step. Hence, we
        # only look for zero crossings of J_lb as a function of lambda on the real line by 
        # default. As soon as a pair of eigenvalues becomes closer than epsilon, we start 
        # to monitor this particular pair to catch the moment when its transitions away 
        # from the real line and becomes a complex conjugate pair.
        is_trans_arr = np.zeros(N_lam, dtype = np.bool) # if True then ev is transitioning
        is_real_arr  = np.ones(N_lam, dtype = np.bool) # if True ev is real
        
        eps = 0.15
                                                     
        for i, mu in enumerate(mu_arr):

            lam_mat[i, :] = lam_arr
            
            j = 0
            
            while j < N_lam: 
                
                lam_init = lam_arr[j]

                if is_trans_arr[j]: # pair of ev [j, j+1] is teaming up to form a complex conjugate pair
                    
                    # check if there are still two zero crossings on the real line
                    dlam = self.solver_param['dlam']
                    
                    lam_max =  lam_arr[j].real   + dlam
                    lam_min =  lam_arr[j+1].real - dlam
                                        
                    a_arr, b_arr = self.find_zero_crossings_on_real_line(mu, sig, lam_min, lam_max, dlam)                                                           
                    
                    if len(a_arr) == 2: # eigenvalue j and ev j+1 are still on the real line

                        lam_arr[j+1] = brentq(self.integrator.calc_J_lb, a_arr[0], b_arr[0], args = (mu, sig))
                        lam_arr[j]   = brentq(self.integrator.calc_J_lb, a_arr[1], b_arr[1], args = (mu, sig))
                                                
                    elif len(a_arr) == 0: # eigenvalue j and ev j+1 teamed up to complex conjugate pair
                        
                        is_real_arr[j]   = False
                        is_real_arr[j+1] = False                        
                        is_trans_arr[j]  = False 

                        lam_real = 0.5*(lam_arr[j].real + lam_arr[j+1].real)                                                                                        
                        # Guess imaginary part based distance on real line
                        lam_imag = np.abs(0.5*(lam_arr[j].real - lam_arr[j+1].real))
                                                                    
                        lam = self.find_complex_ev(np.complex(lam_real, lam_imag), mu, sig, method = 'Nelder-Mead')

                        if lam.imag < 0:
                            lam = np.conjugate(lam)
                        if lam.imag == 0:
                            print('Something went wrong!')
                        lam_arr[j] = lam
                        lam_arr[j+1] = np.conjugate(lam)
                                        
                    else:
                        print('Something went wrong!')
                                                                
                    j += 1                

                elif not is_real_arr[j]: # ev is complex                                
                    lam = self.find_complex_ev(lam_init, mu, sig, method = 'Nelder-Mead')                
                    lam_arr[j]   = lam
                    lam_arr[j+1] = np.conjugate(lam)
                    j += 1                                                
                
                else: # ev is real                    
                    lam_arr[j]  = self.find_real_ev(lam_init.real, mu, sig)
                    
                j += 1
                        
            # Caculate the difference between neighbouring ev which are still on the real 
            # to identify those how teaming up to form complex conjugate pairs            
            nn_dist = np.abs(np.diff(lam_arr[is_real_arr]))                                       
                                    
            # if distance is smaller than eps, then we expect the pair of ev 
            # to become a complex conjugate pair for larger mu 
            idx_arr = np.arange(N_lam)[is_real_arr] 
            idx_arr = idx_arr[:-1][nn_dist < eps]            
            is_trans_arr[idx_arr] = True 
                        
#             print(f'Fitted eigenvalues for mu_{i+1}, {len(mu_arr)-i-1} left to go')
                                                                        
        return lam_mat[:, :-1]
                        
    def compute_real_init_ev(self, mu, sig_min, sig_max):
        '''Caculates the ev values for all sigma between sig_min and sig_max for a given value mu_min
        :param float mu: mean input
        :param sig_min: minimum standard deviation of the input
        :param sig_max: maximum standard deviation of the input'''
                
        N_lam   = self.solver_param['N_lam'] # number of eigenvalues to be found
        dsig    = self.solver_param['dsig'] # step size
        
        sig_arr = np.arange(sig_min, sig_max + 0.1*dsig, dsig)
        M = len(sig_arr)
                        
        lam_mat = np.zeros((len(sig_arr), N_lam))
                
        # We need do find the first intital eigenvalues by scanning the real line brute force
        lam_init_arr = self.compute_real_ev(mu, sig_min) # real part                 
        lam_mat[0, :] = lam_init_arr 
                
        # for the remaining sigma's, we use the eigenvalues caculated for the previous sigma 
        # as an initial guess for the next sigma
        for i, sig in enumerate(sig_arr[1:], 1):

            lam_arr = np.zeros(N_lam)
                                
            for j, lam_init in enumerate(lam_init_arr):
                
                lam_arr[j] = self.find_real_ev(lam_init, mu, sig)
                
            lam_mat[i, :] = lam_arr                
            lam_init_arr  = lam_arr
            
            print(f'Caculated eigenvalues for sig_{i}, {M-i} left to go!')
            
        return lam_mat, sig_arr
            
    def find_zero_crossings_on_real_line(self, mu, sig, lam_min, lam_max, dlam = None):
        '''Find zero crossings on the real line
        :param float mu: mean input
        :param float sig: variance input
        :param float lam_min: left boundary
        :param float lam_max: right boundary'''
        
        if dlam is None:        
            dlam = self.solver_param['dlam'] # step size


        # we start at lam_min + dlam because we need to caculate lam_min before the for loop
        lam_arr = np.arange(lam_min +dlam, lam_max + 0.1*dlam, dlam)        
        
        J_lb = self.integrator.calc_J_lb(lam_min, mu, sig)        
        sign_J_lb = np.sign(J_lb) 

        J_lb_arr = np.zeros_like(lam_arr)
                
        a_arr = [] 
        b_arr = []
                
        for i, lam in enumerate(lam_arr):
            
            J_lb = self.integrator.calc_J_lb(lam, mu, sig)
            
            J_lb_arr[i] = J_lb
                        
            if np.sign(J_lb) != sign_J_lb:
                a_arr.append(lam - dlam)
                b_arr.append(lam)
                
            sign_J_lb = np.sign(J_lb)                                
                            
        return np.array(a_arr), np.array(b_arr)
        
    def compute_real_2_complex_ev(self, mu_min, mu_max, sig):
        
        mu_arr = np.arange(mu_min, mu_max, self.solver_param['dmu'])
                
        # initial eigenvalues 
        lam_arr = self.compute_real_ev(mu_arr[0], sig) # real part         
        lam_arr = lam_arr.astype(np.complex)
         
        N_lam = len(lam_arr)
        
        lam_mat = np.zeros((N_lam, len(mu_arr)), dtype = np.complex) # real part eigenvalues 
                                        
        # we need to keep track when eigenvalues transition from the real line to
        # complex plane                 
        is_real_arr = np.ones(N_lam, dtype = np.bool) # real line
        trans_arr   = np.zeros(N_lam, dtype = np.bool) # transition state
        
        eps = 0.35

        import matplotlib.pyplot as plt

        fig_path = 'figures/ev_real_2_complex/test/'

        for the_file in os.listdir(fig_path):
            file_path = os.path.join(fig_path, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)        
        
        for the_file in os.listdir(fig_path + 'energy/'):
            file_path = os.path.join(fig_path + 'energy/', the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                         
        lam_real_max = np.zeros(N_lam)
        lam_real_min = np.zeros(N_lam)
                            
        for i, mu in enumerate(mu_arr):

            lam_mat[:, i] = lam_arr
            
            j = 0

            fig = plt.figure()
            ax0 = plt.subplot(111)
            
            while j < N_lam: 
                
                lam_init = lam_arr[j]

                if trans_arr[j]: # pair of ev [j, j+1] is teaming up to form a complex conjugate pair
                    
                    # check if there are still two zero crossings
                    M = 100
                    dlam = (lam_arr[j].real - lam_arr[j+1].real)/M
                    
                    lam_max =  lam_arr[j].real   + dlam
                    lam_min =  lam_arr[j+1].real - dlam
                                        
                    a_arr, b_arr, J_lb_arr, lam_zc_arr = self.find_zero_crossings_on_real_line(mu, sig, lam_min, lam_max, dlam)

#                     lam0 = self.find_real_ev(lam_init.real, mu, sig)
#                     lam1 = self.find_real_ev(lam_arr[j+1].real, mu, sig)

                    if lam_real_max[j] == 0:
                        lam_real_max[j] = lam_mat[j, i].real
                    if lam_real_min[j] == 0:
                        lam_real_min[j] = lam_mat[j+1, i].real
                                                           
                    if len(a_arr) == 2: # pair of ev is still on the real line

                        lam_arr[j+1] = brentq(self.integrator.calc_J_lb, a_arr[0], b_arr[0], args = (mu, sig))
                        lam_arr[j]   = brentq(self.integrator.calc_J_lb, a_arr[1], b_arr[1], args = (mu, sig))
                                                
                    elif len(a_arr) == 0: # if eigenvalue j is complex, then ev j+1 must be complex conjugate pair
                        
                        is_real_arr[j]   = False
                        is_real_arr[j+1] = False                        
                        trans_arr[j]     = False 

#                         rrange = (slice(lam_real - 0.05, lam_real + 0.05, 0.001), slice(0.04, 0.06, 0.001))                        
#                         resbrute = brute(self.log_abs_J_lb, rrange, args=(mu,sig))
#                         lam1 = np.complex(resbrute[0], resbrute[1]) 

                        lam_real = 0.5*(lam_arr[j].real + lam_arr[j+1].real)                                                                                        
                        lam = self.find_complex_ev(np.complex(lam_real, 0), mu, sig, method = 'Nelder-Mead')

                        if lam.imag < 0:
                            lam = np.conjugate(lam)
                        if lam.imag == 0:
                            print('Something went wrong!')
                        lam_arr[j] = lam
                        lam_arr[j+1] = np.conjugate(lam)

#                         self.scan_J_lb(mu, sig, lam1, lam1, lam_real - 0.1, lam_real + 0.1, 0.04, 0.06, dlam = 0.001)                    
#                         self.scan_J_lb(mu, sig, lam_arr[j], lam_arr[j+1], lam_real_min[j], lam_real_max[j], -0.2, 0.2)
                                        
                    else:
                        print('Something went wrong!')
                    
#                     self.scan_J_lb(mu, sig, lam_arr[j], lam_arr[j+1], lam_real_min[j], lam_real_max[j], -0.2, 0.2)
                    ax0.plot(lam_arr[j].real, lam_arr[j].imag, 'o', c = 'r')                                        
                    ax0.plot(lam_arr[j+1].real, lam_arr[j+1].imag, 'o', c = 'r')                                        
                                            
                    j += 1                
                
                elif is_real_arr[j]: # ev is real
                    
                    lam_arr[j]  = self.find_real_ev(lam_init.real, mu, sig)
                    ax0.plot(lam_arr[j].real, lam_arr[j].imag, 'o', c = 'k')                                                                        
                else: # ev is complex                                
                    lam = self.find_complex_ev(lam_init, mu, sig, method = 'Nelder-Mead')                
                    lam_arr[j]   = lam
                    lam_arr[j+1] = np.conjugate(lam)
#                     self.scan_J_lb(mu, sig, lam_arr[j], lam_arr[j+1], lam_real_min[j], lam_real_max[j], -0.2, 0.2)
                    ax0.plot(lam_arr[j].real, lam_arr[j].imag, 'o', c = 'k')                                        
                    ax0.plot(lam_arr[j+1].real, lam_arr[j+1].imag, 'o', c = 'k')                                        
                                        
                    j += 1                                                
                    

                j += 1
            
            plt.savefig(fig_path + f'ev_mu_{mu:.3f}.png')
            plt.close(fig)

            # difference between neighbouring real ev 
            idx_arr = np.arange(N_lam)[is_real_arr]            
            nn_dist = np.abs(np.diff(lam_arr[idx_arr]))                                       
            idx_arr = idx_arr[:-1][nn_dist < eps]
                        
            # if distance is smaller than eps, then we expect the pair of ev 
            # to become a complex conjugate pair for larger mu 
            trans_arr[idx_arr] = True 
                        
#             lam_arr = 2*lam_arr - lam_mat[:, i] # improve guess by adding error made in last iteration
            print(f'Fitted eigenvalues for mu_{i+1}, {len(mu_arr)-i-1} left to go')
                   
        self.save_lam_mat(lam_mat)
                                                     
        return lam_mat, mu_arr

    def find_real_ev(self, lam_init, mu, sig, method = 'brentq'):
        '''Find real eigenvalue by minimizing the flux at the left boundary. For this method to work 
        it is crucial that a good initial guess for the position of the eigenvalue is provided
        
        :param comp lam_init: initial guess for complex eigenvalue
        :param float mu: mu
        :param float sig: sigma
        :return comp lam: eigenvalue'''

        if method == 'brentq':
        
            dlam = 0.05        
            a,b = self.bracket_zc(lam_init, dlam, mu, sig)
            
#             if np.sign(self.integrator.calc_J_lb(a, mu, sig)) == np.sign(self.integrator.calc_J_lb(b, mu, sig)):                
#                 print('')
#                 a,b = self.bracket_zc(lam_init, dlam + 0.001, mu, sig)
#                 print('Something went wrong')
            
            lam = brentq(self.integrator.calc_J_lb, a, b, args = (mu, sig), xtol = 1e-20)            
            
        if method == 'Nelder-Mead':
                                                                                             
            fit = minimize(self.integrator.calc_J_lb,
                       np.array([lam_init.real, lam_init.imag]),  
                       args = (mu, sig),
                       method = 'Nelder-Mead')                                 
            
            lam = fit.x
                
        return lam
                
    def find_complex_ev(self, lam_init, mu, sig, method = 'hybr'):
        '''Find complex eigenvalue by minimizing the flux at the left boundary. For this method to work 
        it is crucial that a good initial guess for the position of the eigenvalue is provided
        
        :param comp lam_init: initial guess for complex eigenvalue
        :param float mu: mu
        :param float sig: sigma
        :return comp lam: eigenvalue'''


        if method == 'hybr': 
            
            kwargs = self.root_finder['kwargs']
                                                                     
            fit = root(self.J_lb_comp,
                       np.array([lam_init.real, lam_init.imag]),  
                       args = (mu, sig),
                       **kwargs)                                 
                    
        elif method == 'SLSQP':
                                          
            dx, dy = self.root_finder['param']['bounding_box']
            kwargs = self.root_finder['kwargs']
                                  
            bounds = [(lam_init.real - dx, lam_init.real + dx), 
                      (lam_init.imag - dy, lam_init.imag + dy)]          
        
            fit = minimize(self.J_lb_abs,
                           np.array([lam_init.real, lam_init.imag]),  
                           args = (mu, sig), 
                           method = 'SLSQP',
                           bounds = bounds,
                           **kwargs)
        
        elif method == 'Nelder-Mead':
            
            kwargs = self.root_finder['kwargs']
            
            fit = minimize(self.log_abs_J_lb,
                           np.array([lam_init.real, lam_init.imag]),  
                           args = (mu, sig), 
                           method = 'Nelder-Mead',
                           **kwargs)
            
         
        lam = fit.x
                                                            
        if np.abs(lam[1]) < self.solver_param['eps_imag']:
            lam[1] = 0.0
                        
        return np.complex(lam[0], lam[1])
            
    def compute_ev_mu_arr_sig_arr(self, N, mu_arr, sig_arr):
        
                
        lam_arr = self.get_lam_arr_init(N, mu_arr[0], sig_arr[0])
                       
        lam_mat = np.zeros((len(mu_arr), N), dtype = np.complex128)
                        
        for i, (mu,sig) in enumerate(zip(mu_arr, sig_arr)):
                        
            lam_mat[i, :] = lam_arr            
            lam_arr, _ = self.compute_complex_ev_mu_sig(lam_arr, mu, sig)
            
        lam_mat[-1, :] = lam_arr
        

        filename  = f'EV_feed_forward_2.h5'
        data_path = './data/eigenvalues/'
        
        with h5.File(data_path + filename, mode = 'w') as hf5:                   
        
            hf5['ev']      = lam_mat
            hf5['mu_arr']  = mu_arr
            hf5['sig_arr'] = sig_arr

    
    def compute_real_ev_by_extrapolation(self, N, lam_arr, mu, sig):

        # Allocate output arrays            
        N_lam    = N
        N_lam_bf = self.solver_param['N_lam_bf']          
        dlam     = self.solver_param['dlam']          
        
        full_lam_arr  = np.zeros(N_lam)
        full_J_lb_arr = np.zeros(N_lam)

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
                        
        return full_lam_arr
                    
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
                    
                    # catch rounding errors
                    if np.sign(self.integrator.calc_J_lb(b, mu, sig)) == np.sign(self.integrator.calc_J_lb(b, mu, sig)):
                        b = b + 1e-3                    
                else:
                    b = lam
                    a = lam - dlam 

                    # catch rounding errors
                    if np.sign(self.integrator.calc_J_lb(b, mu, sig)) == np.sign(self.integrator.calc_J_lb(b, mu, sig)):
                        a = a - 1e-3                    
                                                    
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
            lam_arr[i] = np.complex(lam[0], np.abs(lam[1]))
            err_lam_arr[i] = lam_arr[i] - lam_n_init 
            print(f'Found eigenvalue {i+1} using 2d root finder')                 
        
        return lam_arr, err_lam_arr


    def get_lam_arr_init(self, N, mu, sig):
        
        filename  = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_init.h5'
        data_path = './data/eigenvalues/'
        
        f = h5.File(data_path + filename, 'r+')                    
        lam_arr = f['ev'][:N]  

        return lam_arr
        
    def compute_ev_mu_sig(self, mu_min, mu_max, sig_min, sig_max):
        
        dmu  = self.solver_param['dmu']
        dsig = self.solver_param['dsig']

        mu_arr  = np.arange(mu_min, mu_max+0.1*dmu, dmu)
        sig_arr = np.arange(sig_min, sig_max+0.1*dsig, dsig)

        N_mu  = len(mu_arr)
        N_sig = len(sig_arr)
                                                                                                
        lam_arr = self.get_lam_arr_init(mu_max, sig_min)
        
        lam_mat = np.zeros((N_sig, N_mu, len(lam_arr)), dtype = np.complex128)
        lam_mat[0, -1, :] = lam_arr
    
        for i, sig in enumerate(sig_arr[1:]):
            
            lam_arr, _ = self.compute_complex_ev_mu_sig(lam_arr, mu_max, sig)        
            lam_mat[i+1, -1, :] = lam_arr
            
        for i, sig in enumerate(sig_arr):
            
            lam_arr = lam_mat[i, -1, :]
                        
            for j in range(N_mu-2, 0, -1):
                                
                mu = mu_arr[j]
                lam_arr, _ = self.compute_complex_ev_mu_sig(lam_arr, mu, sig)
                lam_mat[i,j,:] = lam_arr
        
        filename  = f'EV_mat_mu_min_{mu_min:.2f}_mu_max_{mu_max:.2f}_sig_min{sig_min:.2f}_sig_max{sig_max:.2f}.h5'
        data_path = './data/eigenvalues/'
        
        with h5.File(data_path + filename) as f:
            
            f['ev_mat']  = lam_mat
            f['mu_arr']  = mu_arr
            f['sig_arr'] = sig_arr
            
        return
    
    def compute_ev_neighbour(self, lam_arr_init, mu, sig):
                
        lam_arr = np.zeros_like(lam_arr_init, dtype = np.complex128)
                
        for n, lam in enumerate(lam_arr):
            
            lam_arr[n] = self.find_complex_ev(lam, mu, sig)
            
        return
            
                        
    def compute_ev_mu_arr_sig(self, mu_min, mu_max, sig):
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
                
    def compute_complex_ev_bf(self, mu, sig, lam_real_arr, lam_imag_arr):
        '''Find complex eigenvalue by brute force scanning the complex plane
        :param float mu: mean input
        :param float sig: sigma
        :param arr lam_real_arr: grid points real axis
        :param arr lam_imag_arr: grid points imaginary axis
        :return complex arr: eigenvalues'''
                                
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
        for k, (i,j) in enumerate(root_idx_arr):
        
            lam_arr_init[k] = np.complex(lam_real_arr[j], lam_imag_arr[i])
            
        # sort eigenvalues by real part
        lam_arr_init = lam_arr_init[np.argsort(np.abs(lam_arr_init))]
                    
        lam_arr = np.zeros_like(lam_arr_init)
                        
        # refine eigenvalues by minimizing flux at left boundary
        for i, lam_n_init in enumerate(lam_arr_init):
            
            lam_arr[i] = self.find_complex_ev(lam_n_init, mu, sig)
                        
        # get rid of real eigenvalues
        eps = 1.e-2
        lam_arr = lam_arr[np.abs(lam_arr.imag) > eps]
                        
        return lam_arr
        
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

    def compute_complex_ev_by_extrapolation(self, init_lam_arr, mu, sig):
        '''Computes complex eigenvalues for a given mu and sigma by extrapolation. The method can only be used 
        if the first 3 (ideally more) complex eigenvalues are already known. The method makes an initial guess for 
        the next eigenvalues by fitting the real and imaginary part as function of the mode number
        
        :param complex arr init_lam_arr: array with initial complex eigenvalues
        :param float mu: mean input
        :param float sig: sigma
        :return lam_arr: lam_arr'''

        M = len(init_lam_arr)                
        N_max = 1000
                
        lam_arr = np.zeros(N_max, dtype = np.complex128)
                                            
        # We assume that the real part is quadratic and imaginary part are 
        # quadratic functions of the mode number
        f_real = lambda n, p0, p1: p1*n**2+p0        
        f_imag = lambda n, p0, p1, p2: p2*n**2+p1*n+p0                
                
        for n in range(M+1, N_max+1):
            
            n_arr = np.arange(1, n)
            lam_arr = lam_arr[:n-1]
            
            optp_real, _ = curve_fit(f_real, n_arr, lam_arr.real)
            optp_imag, _ = curve_fit(f_imag, n_arr, lam_arr.imag)
            lam_n_real = f_real(n, optp_real[0], optp_real[1])
            lam_n_imag = f_imag(n, optp_imag[0], optp_imag[1], optp_imag[2])
            
            lam_n_init = np.complex(lam_n_real, lam_n_imag)
            
            lam = self.find_complex_ev(lam_n_init, mu, sig)
            
            if lam.real < self.solver_param['lam_real_min']:
                break
                                    
            lam_arr[n-1] = lam
            
        return lam_arr

    def compute_real_ev_init(self, mu, sig):
        '''Caculate first N real eigenvalues for given mu and sigma
        :param int N: number of eigenvalues
        :param float mu: mu
        :param float sig: sigma'''
        
        data_path = './data/eigenvalues/'
        filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_init.h5'
                
        if os.path.isfile(data_path + filename):
            file = h5.File(data_path + filename, 'r+')
            if 'real' in list(file.keys()):                            
                lam_arr = file.pop('real')[:]                                            
                if len(lam_arr) < N:                    
                    lam_arr = self.compute_real_ev(mu, sig, N)        
                file['real'] = lam_arr
            else:
                lam_arr = self.compute_real_ev(mu, sig, N)            
#                 lam_arr = self.compute_real_ev_by_extrapolation(N, lam_arr, mu, sig)
                file['real']      = lam_arr                                
        else:
            lam_arr = self.compute_real_ev(mu, sig, N)            
#             lam_arr = self.compute_real_ev_by_extrapolation(N, lam_arr, mu, sig)
            
            with h5.File(data_path + filename, mode = 'r+') as f:                               
                f['real']      = lam_arr                
                f.attrs['mu']  = mu
                f.attrs['sig'] = sig
            
        return lam_arr
    
    def compute_complex_ev_init(self, mu, sig):
        '''Caculate first N complex eigenvalues for given mu and sigma
        :param int N: number of eigenvalues
        :param float mu: mu
        :param float sig: sigma'''
        
        data_path = './data/eigenvalues/'
        filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_init.h5'
        
        dlam = self.param['dlam']
        
        lam_real_arr = np.arange(-30, -0.1, dlam)
        lam_imag_arr = np.arange(0, 20, dlam)
                
        lam_arr = self.compute_complex_ev_bf(mu, sig, lam_real_arr, lam_imag_arr)
        lam_arr = self.compute_complex_ev_by_extrapolation(lam_arr, N, mu, sig)
                
        if os.path.isfile(data_path + filename):
            file = h5.File(data_path + filename, 'r')
            lam_arr = file['complex'][:]
            if len(lam_arr) < N:
                lam_arr = self.compute_complex_ev_by_extrapolation(lam_arr, N, mu, sig)                
                
                file.close()
                
                file = h5.File(data_path + filename, 'w')                
                file['complex'] = lam_arr                                        
        else:
            dlam = 0.05

            with h5.File(data_path + filename, mode = 'w') as f:                   
            
                f['complex']   = lam_arr
                f.attrs['mu']  = mu                
                f.attrs['sig'] = sig
            
        return lam_arr

    def compute_ev_init(self, mu, sig):
                
        lam_arr_comp = self.compute_complex_ev_init(N_comp, mu, sig)
        lam_arr_real = self.compute_real_ev_init(N_real, mu, sig)
        
        data_path = './data/eigenvalues/'
        filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_init.h5'        
        
        lam_arr = np.concatenate((lam_arr_real, lam_arr_comp))
        
        idx_arr = np.argsort(np.abs(lam_arr.real))

        lam_arr = lam_arr[idx_arr]

        f = h5.File(data_path + filename, 'r+')
        
        if 'ev' in list(f.keys()):
            if len(lam_arr) < N_comp + N_real:            
                f.pop('ev')
                f['ev'] = lam_arr
        else:
            f['ev'] = lam_arr
                
        return f['ev'][:]

    def compute_complex_ev_mu_arr_sig(self, N, mu_min, mu_max, sig):
        '''Caculate eigenvalues for given mu in mu_arr and given sig
        :param int N: number of eigenvalues
        :param float mu_min: start value mu 
        :param float mu_max: end value mu
        :param float sig: sigma'''
                                                        
        lam_init_arr = self.compute_complex_ev_init(N, mu_min, sig)        
        mu_arr = np.arange(mu_min, mu_max, self.solver_param['dmu'])
        
        # allocate empty matrix to save eigenvalues
        lam_mat = np.zeros((N, len(mu_arr)), dtype = np.complex)                                
        
        # computate eigenvalues for mu_min assuming that spectrum is real
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
        
            hf5['complex']     = lam_mat
            hf5['mu_arr'] = mu_arr
            hf5.attrs['sig'] = sig
        
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
           
            N     = self.solver_param['N']            
            J_th  = self.solver_param['J_th']
            v_min = self.model_param['v_min']
            v_th  = self.model_param['v_th']
            
            v_arr = np.linspace(v_min, v_th, N)
            x_arr = self.get_x_arr(v_arr, mu, sig)
            
            M = len(x_arr)
            
            # find index of reset point                
            r_idx = self.get_reset_idx(x_arr, mu, sig)
                    
            if np.iscomplexobj(lam):                                
                J, phi = complex_backward_integration(M, lam, J_th, x_arr, r_idx)            
            else:
                J, phi = real_backward_integration(len(x_arr), lam, J_th, x_arr, r_idx)
        
        elif self.solver_param['integrate_in'] == 'python':
            
            phi, J, v_arr = self.backward_integration(lam, mu, sig)
           
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
                #J_lb = calc_j_lb_real(lam, J_th, x_arr, r_idx)
                J_lb = calc_j_lb_complex(lam, J_th, x_arr, r_idx)
                J_lb = J_lb.real
        
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
        
        if output == 'all':    
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