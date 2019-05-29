'''
Created on 20 May 2019

@author: lukas
'''
import cmath
import h5py as h5
from LIF_response import LIF_Theory
import matplotlib.pyplot as plt
from scipy.optimize import brentq, curve_fit, root
import itertools as it 
from timeit import timeit 
import numpy as np
import matplotlib.pyplot as plt
from spectral_solver import Spectral_Solver, Integrator

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
    integrator.set_parameter(J_th = np.sqrt(2./np.pi))

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
    
def compare_phi_to_theory_complex(mu, sig):
        

    
    data_path = 'data/eigenvalues/'
    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    mu  = file.attrs['mu']
    sig = file.attrs['sig']
    
    ev_complex = file['complex'][:]

    model_param  = model_parameter()
    model_param['v_min'] = -1.0
    integrator = Integrator(model_param)
    integrator.set_parameter(J_th = np.sqrt(2./np.pi), 
                             N = int(2e4),
                             integrate_in = 'fortran')

    diffusion_param = model_param.copy()
    diffusion_param['mu']  = mu
    diffusion_param['sig'] = sig
    
    theory = LIF_Theory(diffusion_param = diffusion_param)

    # plot first M eigenfunctions
    M = 3

    v_arr_1 = np.linspace(model_param['v_min'], model_param['v_th'], int(1e3))
    fig = plt.figure(figsize = (10./2.54, 7./2.54))
    gs = plt.GridSpec(M, 2, wspace = 0.55)

    fz = 8
    lfz = 4

    for i, lam in enumerate(ev_complex[0:M]):
    
        ax_i0 = plt.subplot(gs[i, 0])
                
        phi0, _, v_arr  = integrator.eigeneq_backwards_integration(lam, mu, sig)
        
        phi1 = theory.phi_n(i+1, 'complex', v_arr_1)
        
        ax_i0.plot(v_arr_1, phi1.real, ls = '-', c = 'k', label = 'theory')
        ax_i0.plot(v_arr, phi0.real, ls = '--', c = 'r', label = 'numeric') 

        ax_i0.set_ylabel(f'Real$(\phi_{i+1})$', fontsize = fz)
        ax_i0.set_yticks([-1, 1])
        
        if i != M-1:        
            ax_i0.set_xticklabels([])

        ax_i1 = plt.subplot(gs[i, 1])

        ax_i1.plot(v_arr_1, phi1.imag, ls = '-', c = 'k', label = 'theory')
        ax_i1.plot(v_arr, phi0.imag, ls = '--', c = 'r', label = 'numeric') 

        ax_i1.set_ylabel(f'Imag$(\phi_{i+1})$', fontsize = fz)
        ax_i1.set_yticks([-1, 1])
        
        if i != M-1:        
            ax_i0.set_xticklabels([])

        if i == 0:
            ax_i0.set_xlabel(f'$v$', fontsize = fz)
            ax_i1.set_xlabel(f'$v$', fontsize = fz)
            
            ax_i0.legend(fontsize = lfz, loc = 'lower left')

    
    fig_path = '../../transfer_presentation/figures/'            
    fig.savefig(fig_path + 'LIF_ef_theory_numeric.svg', format = 'svg')

    
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
    
def scan_J_lb(mu, sig):    

    model_param  = model_parameter()
    
    integrator = Integrator(model_param)
    integrator.set_parameter(J_lb = np.sqrt(2./np.pi))

    dlam = 0.05
    lam_real_arr = np.arange(-15, -0.1, dlam)
    lam_imag_arr = np.arange(-10, 10, dlam)
    

    J_lb = np.zeros((len(lam_imag_arr), len(lam_real_arr)), dtype = np.complex128)
    
    for i, lam_imag in enumerate(lam_imag_arr):
        for j, lam_real in enumerate(lam_real_arr):
        
            lam = np.complex(lam_real, lam_imag)            
            J_lb[i,j] = integrator.calc_J_lb(lam, mu, sig)

        
    # save
    filename = f'J_lb_mu_{mu:.2f}_sig_{sig:.2f}.h5'        
    data_path = './data/eigenvalues/' 
    
    with h5.File(data_path + filename, mode = 'w') as hf5:                   
    
        hf5['J_lb']   = J_lb
        hf5.attrs['mu']  = mu
        hf5.attrs['sig'] = sig
    
    return
    
    
    
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
    
def test_compute_complex_ev_by_extrapolation(mu, sig):
    
    data_path = 'data/eigenvalues/'    
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}.h5'
    file = h5.File(data_path + filename, 'r')
    
    mu  = file.attrs['mu']
    sig = file.attrs['sig']

    lam_arr = file['complex'][:]

    model_param = model_parameter()
    solver = Spectral_Solver(model_param)
    
    N_complex_lam = 20
        
    lam_arr = solver.compute_complex_ev_by_extrapolation(lam_arr, N_complex_lam, mu, sig)

    n_arr = np.arange(1, len(lam_arr)+1)

    gs = plt.GridSpec(2,1)
    ax0 = plt.subplot(gs[0])
    ax0.plot(n_arr, -lam_arr.real, 'o')
    ax0 = plt.subplot(gs[1])
    ax0.plot(n_arr, lam_arr.imag, 'o')
    
    plt.show()

def complex_ev_scan_mu(mu_min, mu_max, sig):

    model_param = model_parameter()
    solver = Spectral_Solver(model_param)
    
    N = 15
                        
    solver.compute_complex_ev_mu_arr_sig(N, mu_min, mu_max, sig)

def plot_complex_ev_mu(mu_min, mu_max, sig):
    
    filename = f'EV_mu_min_{mu_min:.2f}_mu_max_{mu_max:.2f}_sig_{sig:.2f}.h5'        
    data_path = './data/eigenvalues/' 
    
    file = h5.File(data_path + filename, 'r')
    
    lam_mat = file['complex'][:,:]
    mu_arr  = file['mu_arr'][:]
    
    gs = plt.GridSpec(3,1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    
    M = 15
    K = 5
    
    lam_mat = lam_mat[:M, ]
    
    for i, lam_arr in enumerate(lam_mat.T):
        mu = mu_arr[i]
        ax0.plot(lam_arr.real, lam_arr.imag, 'o', label = f'$\mu$ = {mu:.2f}')
        ax1.plot(mu*np.ones_like(lam_arr), lam_arr.real, 'o')
        ax2.plot(mu*np.ones_like(lam_arr), lam_arr.imag, 'o')
    
    plt.show()

def ceq_mat(mu, sig):
        
    param = model_parameter()

    diffusion_param = param.copy()
    diffusion_param['mu']  = mu
    diffusion_param['sig'] = sig
    
    theory = LIF_Theory(diffusion_param = diffusion_param)
    
    theory.EV_ceq()
    
    print('Finished')

def plot_ceq(mu, sig):
    
    filename = f'ceq_mat_mu_{mu:.2f}_sig_{sig:.2f}.h5'        
    data_path = './data/eigenvalues/' 

    f = h5.File(data_path + filename, 'r')
    
    mu  = f.attrs['mu']
    sig = f.attrs['sig']
    ceq = f['ceq_mat'][:, :]

    inch = 2.54
    fig = plt.figure() # figsize = (10/inch, 4.5/inch)
    ax = plt.subplot(111)
        
    plt.imshow(np.abs(ceq))    
    ax.set_xlabel('RE($\lambda$)', fontsize = 12)
    ax.set_ylabel('Im($\lambda$)', fontsize = 12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.colorbar()
            
    fig_path  = '../../transfer_presentation/figures/'
    fig.savefig(fig_path + 'ceq.svg', format = 'svg')

    fig = plt.figure() # figsize = (10/inch, 4.5/inch)
    ax = plt.subplot(111)
        
    plt.imshow(np.log(np.abs(ceq)))    
    ax.set_xlabel('RE($\lambda$)', fontsize = 12)
    ax.set_ylabel('Im($\lambda$)', fontsize = 12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.colorbar()
            
    fig_path  = '../../transfer_presentation/figures/'
    fig.savefig(fig_path + 'ceq_log.svg', format = 'svg')

    fig = plt.figure() # figsize = (10/inch, 4.5/inch)
    ax = plt.subplot(111)
        
    B = np.log(np.abs(ceq.imag))        
    
    for i in range(np.size(B, 1)):
    
        B[200, i] = 0.5*(B[199, i]+B[201, i])
    
    plt.imshow(np.log(np.abs(ceq.real))- B)    
    ax.set_xlabel('RE($\lambda$)', fontsize = 12)
    ax.set_ylabel('Im($\lambda$)', fontsize = 12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.colorbar()
            
    fig_path  = '../../transfer_presentation/figures/'
    fig.savefig(fig_path + 'ceq_real_imag.svg', format = 'svg')

def plot_J_lb(mu, sig):
    
    filename = f'J_lb_mu_{mu:.2f}_sig_{sig:.2f}.h5'        
    data_path = './data/eigenvalues/' 

    f = h5.File(data_path + filename, 'r')
    
    mu  = f.attrs['mu']
    sig = f.attrs['sig']
    ceq = f['ceq_mat'][:, :]

    inch = 2.54
    fig = plt.figure() # figsize = (10/inch, 4.5/inch)
    ax = plt.subplot(111)
        
    plt.imshow(np.abs(ceq))    
    ax.set_xlabel('RE($\lambda$)', fontsize = 12)
    ax.set_ylabel('Im($\lambda$)', fontsize = 12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.colorbar()
            
    fig_path  = '../../transfer_presentation/figures/'
    fig.savefig(fig_path + 'J_lb.svg', format = 'svg')

    fig = plt.figure() # figsize = (10/inch, 4.5/inch)
    ax = plt.subplot(111)
        
    plt.imshow(np.log(np.abs(ceq)))    
    ax.set_xlabel('RE($\lambda$)', fontsize = 12)
    ax.set_ylabel('Im($\lambda$)', fontsize = 12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.colorbar()
            
    fig_path  = '../../transfer_presentation/figures/'
    fig.savefig(fig_path + 'ln_J_lb.svg', format = 'svg')

    fig = plt.figure() # figsize = (10/inch, 4.5/inch)
    ax = plt.subplot(111)
        
    B = np.log(np.abs(ceq.imag))        
    
    for i in range(np.size(B, 1)):
    
        B[200, i] = 0.5*(B[199, i]+B[201, i])
    
    plt.imshow(np.log(np.abs(ceq.real))- B)    
    ax.set_xlabel('RE($\lambda$)', fontsize = 12)
    ax.set_ylabel('Im($\lambda$)', fontsize = 12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.colorbar()
            
    fig_path  = '../../transfer_presentation/figures/'
    fig.savefig(fig_path + 'J_lb_real_imag.svg', format = 'svg')
    
def scan_mu_sig_ev():
        
    sig_min = 0.2
    sig_max = 0.6
    mu_min = 0.5
    mu_max = 1.5
    
    model_param  = model_parameter()       
    solver = Spectral_Solver(model_param)
    solver.set_integrator_parameter(J_lb = np.sqrt(2./np.pi))
    solver.compute_ev_mu_sig(mu_min, mu_max, sig_min, sig_max)


#     solver.compute_complex_ev_init(20, mu_max, sig_min)
    
    
def plot_init():
     
    mu  = 1.5
    sig = 0.2
     
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_init.h5'
    data_path = './data/eigenvalues/'
     
    f = h5.File(data_path + filename)                    
    lam_arr = f['complex'][:]
     
    lam_arr_real = f['real'][:]

    fig = plt.figure(figsize = (10/2.54, 7.5/2.54))
    ax = plt.subplot(111)     
    ax.plot(lam_arr.real, lam_arr.imag, 'o')
    ax.plot(lam_arr_real, np.zeros_like(lam_arr_real), 'o', ms = 1.0) 
    ax.grid(True)
    ax.set_xlabel('Re($\lambda$)', fontsize = 12)
    ax.set_ylabel('Im($\lambda$)', fontsize = 12)
    
    fig_path  = '../../transfer_presentation/figures/'
    fig.savefig(fig_path + 'ev_init.svg', format = 'svg')
    
   
def delete_h5():

    mu  = 1.5
    sig = 0.2
     
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_init.h5'
    data_path = './data/eigenvalues/'

    f = h5.File(data_path + filename)                    
    f.pop('real')


def prepare_init_h5():

    mu  = 1.5
    sig = 0.2
     
    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_init.h5'
    data_path = './data/eigenvalues/'

    f = h5.File(data_path + filename, 'r+')                    
    
    lam_real    = f['real']
    lam_complex = f['complex'] 

    lam = np.concatenate((lam_real, lam_complex))    
    idx = np.argsort(np.abs(lam.real))
    
    lam = lam[idx]
    
    f['ev'] = lam


def feed_forward():

    mu  = 0.8 
    sig = 0.2
    
    tau = 10*1e-3

    K = 5000
    
    g = 4

    KE = g*K/(g+1)
    KI = K/(g+1)
    
    
    hE = 0.01
    hI = -g*hE
    

    A = np.array([[KE*hE*tau, KI*hI*tau],
                  [KE*hE**2*tau, KI*hI**2*tau]])
    
    r = np.linalg.solve(A, [mu, sig**2])
    
    rE = r[0]
    rI = r[1]
    
    amp = 0.25*rE
    w   = 5 # in units of tau
    
    T = 50 # simulation time in units of tau
    
    ds = 0.01
    
    s = np.arange(0, T+0.1*ds, ds)    
    s_mp = s + 0.5*ds
    
    mu_s  = mu + amp*np.sin(w*s)*KE*hE*tau
    sig_s = np.sqrt(sig**2 + amp*np.sin(w*s)*KE*hE**2*tau)
    
    model_param = model_parameter()
    solver = Spectral_Solver(model_param)

    solver.set_integrator_parameter(J_lb = np.sqrt(2./np.pi))
#     solver.compute_ev_init(20, 120, mu_s[0], sig_s[0])

    solver.compute_ev_mu_arr_sig_arr(20, 120, mu_s, sig_s)

def feed_forward_plot():

    mu  = 0.8 
    sig = 0.2

    filename = f'EV_mu_{mu:.2f}_sig_{sig:.2f}_init.h5'   
    data_path = './data/eigenvalues/'

    f = h5.File(data_path + filename, 'r+')

    lam_arr = f['ev'][:]
    
    plt.plot(lam_arr.real, lam_arr.imag, 'o')
    plt.show() 


if __name__ == '__main__':
    
#     mu   = 1.2
#     sig  = 0.20    0
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
#     scan_J_lb()                f = 10                f = 10


#     mu  = 0.64
#     sig = 0.2
#     compare_complex_ev_to_theory(mu, sig)
#     test_compute_complex_ev_by_extrapolation(mu, sig)
#     mu_min = 0.64
#     mu_max = 1.5
#     sig = 0.2
#     complex_ev_scan_mu(mu_min, mu_max, sig)
#     plot_complex_ev_mu(mu_min, mu_max, sig)
#     mu  = 0.8
#     sig = 0.2
#     ceq_mat(mu, sig)    
#     plot_ceq(mu, sig)
#     mu  = 0.8
#     sig = 0.2
#     compare_phi_to_theory_complex(mu, sig)
#     scan_J_lb(mu, sig)    
#     plot_J_lb(mu, sig)
#     scan_mu_sig_ev()
#     plot_init()
#     delete_h5()
#     prepare_init_h5()    
#     scan_mu_sig_ev()    
    
    feed_forward()    
#     feed_forward_plot()

        
    print('Finished')