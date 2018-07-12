import numpy as np
import matplotlib.pyplot as plt

def exp_decaying_current(t, spike_times, tau, h):
    '''Model synaptic current as a simply exponential decay'''
    
    I = np.zeros_like(t)
    
    for t_spike in spike_times:
                    
        H_t = 0.5*(1 + np.sign(t-t_spike)) # heaviside function centered around spike time                    
        I += h/tau*np.exp(-(t-t_spike)/tau)*H_t # exponential decay 

    return I 

def poisson_spike_train(lam, T):
    
    n = np.random.poisson(T*lam)
    spike_times = np.random.uniform(0, T, n)

    return spike_times

# Model incoming spikes as poisson input 
N = 100 # number of trials
lam = 10. # rate
T = 10. # recording time

#------------------------------------------------------------------------------ 
# Synaptic current

tau = 2. # intrinsic synapse time constant
h = 1. # efficacy 

delta_t = 0.001*tau # time step
M = int(np.ceil(T/delta_t)) # temporal resolutioon

t = np.linspace(0, T, M)
spike_times = poisson_spike_train(lam, T)

I = exp_decaying_current(t, spike_times, tau, h)

plt.plot(t, I, '-', color = 'k')
plt.plot(spike_times, np.zeros_like(spike_times), 'o', color = 'blue')

#------------------------------------------------------------------------------ 
# Diffusion approximation

lam = 10. # rate
tau = 1. # intrinsic synapse time constant
h = 1. # efficacy 

mu = tau*lam*h 
sig2 = tau**2*lam*h**2

N = 100 # number of trials
T = 10 # recording time 

delta_t = 0.001*tau # time step

M = int(np.ceil(T/delta_t)) # number of time points

t = np.linspace(0, T, M) # time points 

# I0 = mu*np.exp(-t/tau) # initial value 
# I_mat = np.tile(I0, (N,1))

I_mat = np.zeros((N,M))

for i in xrange(N):
    
    spike_train = poisson_spike_train(lam, T)
    I_mat[i, :] += exp_decaying_current(t, spike_train, tau, h)

plt.figure()

for I in I_mat:
    
    plt.plot(t, I, '-', linewidth = 0.1)


plt.ylim(0, max(I_mat.flatten()))
plt.xlim(0, t[-1])

plt.hlines(mu, 0, t[-1], colors = 'k', linestyles = '-', label = '$\mu$')
plt.hlines([mu - 2*np.sqrt(sig2), mu + 2*np.sqrt(sig2)], 0, t[-1], colors = 'k', linestyles = '--', label = ['$\mu - \sigma$', '$\mu + \sigma$'])
# plt.legend()
plt.xlabel('time', fontsize = 18)
plt.ylabel('I(t)', fontsize = 18)


plt.show()
# 
# I_trial_mat = np.zeros((N, M))
# 
# for i in xrange(N):
#     
#     spike_times = poisson_spike_train(lam, T)
#     I_trial_mat[i, :] = exp_decaying_current(t, spike_times, tau, h)
# 
# 
# #------------------------------------------------------------------------------ 
# # Plot trials 
# 
# for I in I_trial_mat:
#     
#     plt.plot(t, I)
#     
# plt.show()    
    
    