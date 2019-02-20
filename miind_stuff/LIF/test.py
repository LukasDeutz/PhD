import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import nest

from LIF_response import LIF_Response, LIF_Theory
from nest import pynestkernel

def test_nest_0():
        
    neuron = nest.Create('iaf_psc_alpha')
#     print(nest.GetStatus(neuron))
    nest.SetStatus(neuron, {'I_e': 376.})    
    
    mm = nest.Create('multimeter')
    nest.SetStatus(mm, {"withtime": True, 'record_from':['V_m']})
                
    sd = nest.Create('spike_detector', params={"withgid": True, "withtime": True})
    
    nest.Connect(mm, neuron)
    nest.Connect(neuron, sd)
    
    nest.Simulate(1000.0)

    data_mm = nest.GetStatus(mm, keys = 'events')[0]
    Vms = data_mm['V_m']
    ts  = data_mm['times']
    
    plt.plot(ts, Vms)
    
    data_sd = nest.GetStatus(sd, keys = 'events')[0] 
    
#     evs = data_sd['senders']
    ts  = data_sd['times']
    
    v_th = nest.GetStatus(neuron, 'V_th')
            
    plt.plot(ts, v_th*np.ones_like(ts), 'o')
    
    plt.show()
    
    
def test_nest_1():
    
    # number of trials 
        
    nest.SetKernelStatus({'resolution': 0.01})
    
    N = 1000
        
    model_param = {}
    
    model_param['V_reset'] = 0.0
    model_param['V_th']    = 1.
    model_param['tau_m']   = 10.0
#     model_param['tau_ref'] = 0.0
    model_param['E_L']     = 0.0
    model_param['V_m']     = 0.0
    model_param['C_m']     = 1.0
                
    neurons = nest.Create('iaf_psc_delta', 1000, params = model_param)
    
    KE = 1.0
    KE = 1.0
     
    mu  = 0.8
    sig = 0.15
    
    hE =  0.01
    hI = -0.01
    
    KE = 1.
    KI = 1.

    rE, rI = get_rE_rI(mu, sig, model_param['tau_m']*1.e-3, KE, hE, KI, hI)
                            
 
    noise_ex = nest.Create("poisson_generator")
    noise_in = nest.Create("poisson_generator")
     
    nest.SetStatus(noise_ex, {'rate': rE})
    nest.SetStatus(noise_in, {'rate': rI})    
    
    nest.Connect(noise_ex, neurons, syn_spec = {'weight': hE})
    nest.Connect(noise_in, neurons, syn_spec = {'weight': hI})


    mm = nest.Create('multimeter')
    nest.SetStatus(mm, {"withtime": True, 'record_from':['V_m']})
                
    sd = nest.Create('spike_detector', params={"withgid": True, "withtime": True})
    
    nest.Connect(mm, neurons)
    nest.Connect(neurons, sd)
    
    t_end = 1000.0
    
    nest.Simulate(t_end)

    data_mm = nest.GetStatus(mm, keys = 'events')[0]
    Vms = data_mm['V_m']
    ts  = range(1, 10000)
    
    Vms = np.reshape(Vms, (int(t_end-1), N))

    return

def get_rE_rI(mu, sig, tau, KE, hE, KI, hI):

    A = tau*np.array([[KE*hE, KI*hI],
                      [KE*hE**2, KI*hI**2]])
    
    rE, rI = np.linalg.solve(A, [mu, sig**2])
    
    return rE, rI

def mc_parameter(model_param):
    
    dt = 1.e-4 # time step
    N_trial = int(1e5) # number of trials
    bins_v  = np.linspace(model_param['v_min'], model_param['v_th'], 100, endpoint = True)
    
    mc_param = {}
    mc_param['dt']       = dt
    mc_param['N_trial']  = N_trial
    mc_param['bins_v']   = bins_v
    mc_param['t_report'] = 1.e-3
    mc_param['t_end']    = 1.0
     
    return mc_param

def miind_parameter():
    
    #===========================================================================
    # miind parameter
    submit = 'response' # 
    basename    = 'LIF' # 
        
    uac         = True # use geometric method to caclculate transition matrix
    N_mc        = 1000 # precision of transition matrix
    dt          = 1e-4 # time step
    t_report    = 1e-3 # report time 
    N_grid      = 150  # number of cells in strip     
    w_strip     = 0.01 # width of bin size for the artificial second dimension
    t_end       = 0.1
    #===========================================================================    

    param = {}
    
    param['submit']   = submit
    param['basename'] = basename

    param['uac']            = uac    
    param['N_mc']           = N_mc
    param['dt']             = dt
    param['t_report']       = t_report  
    param['t_state_report'] = t_report      
    param['N_grid']         = N_grid  
    param['w_strip']        = w_strip
    param['t_end']          = t_end

    return param

def nest_parameter():

    #===========================================================================
    # miind parameter
    
    N_trial   = int(1.e4) # number of trials
    dt        = 1.e-4 # time step
    dt_report = 1.e-3 # report time
    t_end     = 1.0   # end of simulation
    n_vbins   = 100 # number bins density
    
    #===========================================================================    

    param = {}
    
    param['N_trial']   = N_trial
    param['dt']        = dt
    param['dt_report'] = dt_report
    param['t_end']     = t_end
    param['n_vbins']   = n_vbins
    
    return param

def model_parameter(mu, sig):

    #===============================================================================
    # neuron model
    v_th    = 1.0 # threshold
    v_reset = 0.0 # reset   
    v_rev   = 0.0 # reversal
    v_min   = -v_th # minimum
    tau     = 1e-2 # membrane time constant
    tau_ref = 0 # refractiory period        
    #===============================================================================
                
    #===============================================================================
    # input     
    N_inp = 2 # number of inputs
    g = 4. # ratio between excitatory and inhibitory inputs    
    K = 10000 # total number of indegrees
    
    KE = g/(g+1)*K # number of excitatory indegrees
    KI = K/(g+1) # number of inhibitory indegrees
            
    hE = 0.0001 # efficacy of excitatory inputs
    hI = -g*hE # efficacy of inhibitory inputs
        
    # diffusion approximation: caculate input rate for given mu and sig 
    r_ext_E, r_ext_I = get_rE_rI(mu, sig, tau, KE, hE, KI, hI)    
    #===============================================================================
    
    param = {}
    
    param['v_th']    = v_th
    param['v_reset'] = v_reset
    param['v_min']   = v_min
    param['v_rev']   = v_rev
    param['tau']     = tau
    param['tau_ref'] = tau_ref
        
    param['N_inp']   = N_inp   
    param['hE']      = hE
    param['KE']      = KE
    param['r_ext_E'] = r_ext_E    
    param['r_ext_I'] = r_ext_I        
    param['hI']      = hI
    param['KI']      = KI
    param['r_inh_I'] = r_ext_I
    param['h_arr'] = np.array([hE, hI])
    param['K_arr'] = np.array([KE, KI])
    param['r_ext_arr'] = np.array([r_ext_E, r_ext_I])
    
    return param


def plot_rate(filename, mc = False, miind = False, nest = False):

    #===============================================================================
    # Load data
    h5f = h5.File(filename)            
    h5f['model_parameter']
    
    model_param = {}
    
    for key in h5f['model_parameter'].keys():    
        model_param[key] = h5f['model_parameter/' + key].value

    r_max_0 = 0
    r_max_1 = 0
    r_max_2 = 0    
    t_end_0 = 0
    t_end_1 = 0
    t_end_2 = 0
    
    if miind:
        r0 = h5f['miind/rate/r'][:] 
        t0 = h5f['miind/rate/t'][:]            
        r_max_0 = np.max(r0)
        t_end_0 = t0[-1]

    if mc:
        r1 = h5f['mc/rate/r'][:] 
        t1 = h5f['mc/rate/t'][:]            
        r_max_1 = np.max(r1)
        t_end_1 = t1[-1]

    if nest:
        r2 = h5f['nest/rate/r'][:] 
        t2 = h5f['nest/rate/t'][:]            
        r_max_2 = np.max(r2)
        t_end_2 = t2[-1]

    t_start = 0.0
    t_end = np.max([t_end_0, t_end_1, t_end_2])    
    
    #===============================================================================
    
    #===========================================================================
    # theory 
    theory = LIF_Theory(model_param)
    r_stat = theory.get_r0()
    
    r_max = np.max([r_max_0, r_max_1, r_max_2, r_stat])

    #===========================================================================
    
    #=======================================================================
    # Plot 
    fz  = 18
    lfz = 14. # legend fontsize
    
    fig = plt.figure(figsize = (8, 12))
    gs  = plt.GridSpec(1,1)    
     
    # Compare rate against theoretical expectation
    ax0 = plt.subplot(gs[0])
 
    if miind:
        ax0.plot(t0, r0, ls = '-', c = 'r', label = r'miind')    
    if mc:
        ax0.plot(t1, r1, ls = '-', c = 'g', label = r'mc')
    if nest:
        ax0.plot(t2, r2, ls = '-', c = 'b', label = r'nest')
        
    ax0.plot([t_start, t_end], [r_stat, r_stat], ls = '--', c = 'k', label = r'$r_0$')
    
    if model_param['v_th'] < theory.mu:
        
        r_stat_approx = theory.r0_approx
        ax0.plot([t_start, t_end], [r_stat_approx, r_stat_approx], ls = ':', c = 'k', label = r'$r_0$ approx')
        
     
    ax0.set_xlim([t_start, t_end])
    ax0.set_ylim([0,  1.05*r_max])
    ax0.legend(fontsize = lfz)
    ax0.grid(True)
    ax0.set_ylabel(r'rate [Hz]', fontsize = fz)
    
    plt.show()
    #=======================================================================
    
    return

def plot_stationary_density(filename, mc = True, miind = True, nest = True):
    
    #===============================================================================
    # Load data
    h5f = h5.File(filename)            
    h5f['model_parameter']
    
    model_param = {}
        
    for key in h5f['model_parameter'].keys():    
        model_param[key] = h5f['model_parameter/' + key].value

    v_min = model_param['v_min']
    v_th  = model_param['v_th']


    if miind:
        rho0_0 = h5f['miind/density']['rho'][:] 
        v0     = h5f['miind/density']['v'][:] 
        
    if mc:
        rho0_1 = h5f['mc/density']['rho'][-3, :] 
        v1     = h5f['mc/density']['v'][:] 
        
    if nest:
        rho0_2 = h5f['nest/density']['rho'][-1, :] 
        v2     = h5f['nest/density']['v'][:]                 
    #===============================================================================
    
    #===========================================================================
    # theory 
    theory = LIF_Theory(model_param)
    v3 = np.linspace(v_min, v_th, 100)
    rho0_3  = theory.get_phi_0(v3)
    #===========================================================================
    
    
    #=======================================================================
    # Plot 
    fz  = 18
    lfz = 14. # legend fontsize
    
    fig = plt.figure(figsize = (8, 12))
    gs  = plt.GridSpec(1,1)    
     
    # Compare rate against theoretical expectation
    ax0 = plt.subplot(gs[0])
 
    if miind:
        ax0.plot(v0, rho0_0, ls = ':', c = 'k', label = r'miind')    
    if mc:
        ax0.plot(v1, rho0_1, ls = '--', c = 'k', label = r'mc')
    if nest:
        ax0.plot(v2, rho0_2, ls = '--', c = 'k', label = r'nest')
    
    ax0.plot(v3, rho0_3, ls = '-', c = 'r', label = r'theory')    

    ax0.set_xlim([v_min, v_th])
#     ax0.set_ylim([0,  1.05*r_max])
    ax0.legend(fontsize = lfz)
    ax0.grid(True)
    ax0.set_ylabel(r'density', fontsize = fz)
    
    plt.show()
    #=======================================================================
    
    return

def run(filename, miind = False, nest = False, mc = False):
    
    mu  = 0.7
    sig = 0.2 
 
    model_param = model_parameter(mu, sig)
    
    LIF = LIF_Response(model_param)
    
    if miind:
        miind_param = miind_parameter()
        LIF.run_mind(miind_param)            
    if nest:
        nest_param = nest_parameter()
        LIF.run_nest(nest_param)
    if mc: 
        mc_param = mc_parameter(model_param)
        LIF.run_mc(mc_param)
                
    LIF.save(filename)   
    
    return
    
def test_theory_stationary_rate():

    import mpmath as mp
    
    mu_arr = np.linspace(0, 15, 50)
    sig  = 5.
    v_r  = 0.
    v_th = 15.
       
    tau = 20*1.e-3

    r_arr = np.zeros_like(mu_arr)

    for i, mu in enumerate(mu_arr): 
                
        a = (v_r  - mu)/sig
        b = (v_th - mu)/sig
                                           
        r0 = 1./(tau*mp.sqrt(mp.pi)*mp.quad(lambda x: (1.+mp.erf(x))*mp.exp(x**2), [a, b]))
        r_arr[i] = r0
        
    gs = plt.GridSpec(1, 2)
    ax0 = plt.subplot(gs[0])
    ax0.plot(mu_arr, r_arr)
    
    mu = 10
    sig_arr = np.linspace(0, 10, 50)

    r_arr = np.zeros_like(sig_arr)

    
    for i, sig in enumerate(sig_arr): 
                
        a = (v_r  - mu)/sig
        b = (v_th - mu)/sig
                                           
        r0 = 1./(tau*mp.sqrt(mp.pi)*mp.quad(lambda x: (1.+mp.erf(x))*mp.exp(x**2), [a, b]))
        r_arr[i] = r0

    ax1 = plt.subplot(gs[1])
    ax1.plot(sig_arr, r_arr)
        
    plt.show()


def test_theory():
        
    mu  = 0.8
    sig = 0.1
    
    model_param = model_parameter(mu, sig)    
    
    theory = LIF_Theory(model_param)
    r_stat = theory.r0()

    pass
            
if __name__ == '__main__':
    
    hE = 0.01
    
    filename = f'./data/test_hE_{hE}.h5'    


#     run(filename, nest = False, mc = True)

    plot_rate(filename, nest=False, mc = True, miind = False)    
#     plot_stationary_density(filename, miind = False)    
#     plot_stationary_density(filename, miind=False, mc=True, nest=False)    
#     test_theory_stationary_rate()
    
    print('Finished')
    
    
    