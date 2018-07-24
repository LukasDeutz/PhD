import numpy as np
import matplotlib.pyplot as plt

from current_based_synapses import generate_exp_decaying_current_trials, generate_voltage_trials

#------------------------------------------------------------------------------ 
# parameters

# excitatory synapse
lamE = 50. # rate
tauE = 1 # intrinsic synapse time constant
hE = 0.1 # efficacy 

muE = tauE*lamE*hE 
sigE2 = 0.5*tauE*lamE*hE**2

# inhibitory synapse
lamI = 5. # rate
tauI = 2 # intrinsic synapse time constant
hI = -0.25 # efficacy 

# stationary limit 
muI = tauI*lamI*hI 
sigI2 = 0.5*tauI*lamI*hI**2

mu = muE + muI # expectation value
sig2 = sigE2 + sigI2 # variance

N = 10 # number of trials
T = 100 # recording time 

delta_t = 0.01*min(tauE, tauI) # time step

t = np.arange(0, T + delta_t, delta_t)

#------------------------------------------------------------------------------ 
# current traces

# sharp initial condition I(0)=0  
I0_E = 0.
I0_I = 0.

I_mat_E = generate_exp_decaying_current_trials(N, T, t, tauE, hE, lamE, I0_E)
I_mat_I = generate_exp_decaying_current_trials(N, T, t, tauI, hI, lamI, I0_I)
        
I_mat = I_mat_E + I_mat_I

#------------------------------------------------------------------------------ 
# membrane potential

tau_m = 10 # membrane time constant

V0 = 0 # sharpe initial condition 

V_mat = generate_voltage_trials(N, t, tau_m, V0, I_mat)


# stationary limit
mu_V = tau_m*mu # expectation value  
sig2_V = tauE*tau_m**2/(tauE + tau_m)*sigE2 + tauI*tau_m**2/(tauI + tau_m)*sigI2 # variance

#------------------------------------------------------------------------------ 
# Plotting

plt.figure(figsize = (14, 10))
 
gs = plt.GridSpec(2,1)
 
ax0 = plt.subplot(gs[0])
 
for I in I_mat:
     
    ax0.plot(t, I, '-', linewidth = 0.1)  
 
y_max = max(I_mat.flatten()) 
y_min = min(I_mat.flatten())
 
plt.hlines(mu, 0, t[-1], colors = 'k', linestyles = '-', label = r'$\mu$')
plt.hlines(mu - 2*np.sqrt(sig2), 0, t[-1], colors = 'k', linestyles = '--', label = r'$\mu - 2\sigma$')
plt.hlines(mu + 2*np.sqrt(sig2), 0, t[-1], colors = 'k', linestyles = '--', label = r'$\mu + 2\sigma$')
 
ax0.set_title(r'Sharp initial condition I(O)=I', fontsize = 18)    
plt.xlim(0, t[-1])
plt.ylim(y_min, y_max)
plt.legend(loc = 'upper right', fontsize = 16)    
 
ax1 = plt.subplot(gs[1])
 
for V in V_mat:
     
    ax1.plot(t, V, '-', linewidth = 0.1)  
 
y_max = max(V_mat.flatten()) 
y_min = min(V_mat.flatten())
 
plt.hlines(mu_V, 0, t[-1], colors = 'k', linestyles = '-', label = r'$\mu$')
plt.hlines(mu_V - 2*np.sqrt(sig2_V), 0, t[-1], colors = 'k', linestyles = '--', label = r'$\mu - 2\sigma$')
plt.hlines(mu_V + 2*np.sqrt(sig2_V), 0, t[-1], colors = 'k', linestyles = '--', label = r'$\mu + 2\sigma$')
 
ax1.set_title(r'Sharp initial condition V(0)=V', fontsize = 18)    
plt.xlim(0, t[-1])
plt.ylim(y_min, y_max)
plt.legend(loc = 'upper right', fontsize = 16)    
     
plt.show()    