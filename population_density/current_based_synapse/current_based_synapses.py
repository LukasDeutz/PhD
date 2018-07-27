import numpy as np

def poisson_spike_train(lam, T):
    '''Draw random spike times for a Poisson process with constant rate within the given time interval 
    :param lam: rate
    :param T: recording time  
    :return spike_train: spike times 
    '''
    n = np.random.poisson(T*lam)
    spike_train = np.random.uniform(0, T, n)

    return spike_train

def exp_decaying_current(t, spike_train, tau, h):
    '''Caculate current trace at time points t for incoming spike train
    :param t: time points
    :param spike_train: spike_times   
    :param tau: synaptic time constant
    :param h: synaptic efficacy
    :return I: current trace'''
    
    I = np.zeros_like(t)
    
    for t_spike in spike_train:
                    
        H_t = 0.5*(1 + np.sign(t-t_spike)) # heaviside function centered around spike time                    
        I += h*np.exp(-(t-t_spike)/tau)*H_t # exponential decay 

    return I 

def generate_exp_decaying_current_trials(N, T, t, tau, h, lam, I0):
    '''Generate N current traces for randomly drawn Poisson spike trains
    :param N: number of trials
    :param t: time points 
    :param tau: synaptic time constant
    :param h: synaptic efficacy
    :param I0: initial condition 
    :return I_mat: N x len(t) matrix with current traces
    '''

    I_mat = np.tile(np.exp(-t/tau), (N,1))

    # initial value
    if np.isscalar(I0):
        I_mat = I0*I_mat         
    else:
        for i, I0_i in enumerate(I0):
            I_mat[i,:] *= I0_i
                       
    for i in xrange(N):
        
        spike_train = poisson_spike_train(lam, T)
        I_mat[i, :] += exp_decaying_current(t, spike_train, tau, h)

    return I_mat

def generate_voltage_trials(N, t, tau, V0, I_mat):
    '''Generate N voltage traces for N different current inputs unsing euler method
    :param N: number of trials
    :param t: time points 
    :param tau: membrane time constant
    :param V0: initial value
    :param I_mat: N x len(t) matrix with current traces
    :return V_mat: N x len(t) matrix with voltage traces
    '''
    M = len(t) # number of time steps
    dt = t[1] - t[0] # step width 
        
    V_mat = np.tile(np.exp(-t/tau), (N,1))

    # initial value
    if np.isscalar(V0):
        V_mat = V0*V_mat         
    else:
        for i, V0_i in enumerate(V0):
            V_mat[i,:] *= V0_i  

    delta_t = t[1] - t[0]

    kernel = np.ones((M, M))
    kernel = np.triu(kernel, 0) # columns are heaviside functions for sliding t
             
    for i, ti in  enumerate(t):         
        kernel[:, i] =  kernel [:, i]*np.exp(-(ti-t)/tau)

    V_mat = np.matmul(I_mat, kernel)*dt
                                                                                  
    return V_mat    
    
    
    
    
    









