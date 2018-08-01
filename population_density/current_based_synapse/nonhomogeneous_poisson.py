import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial
from scipy.optimize import root 
from scipy.integrate import quad
from scipy.stats import poisson

def generate_nonhomogeneous_poisson_0(r, T, r_max):
    '''Generate spike train from nonhomogeneous Poisson process with rate function r_t within recording time T
    param r: rate function
    param T: recording time
    param r_max: max(r) for t in [0,T]
    return spike_times: array with spike times'''
    
    N = np.random.poisson(r_max*T)
    
    # draw spike for homogeneous Poisson process with rate r_max
    hom_spike_times = np.sort(np.random.uniform(0, T, N)) 

    #thinning
    p = r(hom_spike_times)/r_max # probability to accept spike
    u = np.random.uniform(size = N)        
    
    spike_times = hom_spike_times[u <= p]
    
    return spike_times
    

#------------------------------------------------------------------------------ 
# parameter
    
tau = 1. # membrane time constant 

# define oscillating rate function 
T = 10.*tau # choose period large compared to the membrane time constannt
w = 2.*np.pi/T

A  = 5. # amplitude of oscillations 
r0 = 20. # stationary rate 
r_max = r0 + A

r = lambda t: A*np.sin(w*t) + r0 # rate function

Tr = 2*T # recording time

#------------------------------------------------------------------------------ 
# generate ensemble of spike trains and check statistics  

N = 100000 # number of trials

# select random time window

t0 = np.random.uniform(0, Tr)
t1 = np.random.uniform(0, Tr)

if t0 < t1:
    pass
else:
    t = t0
    t0 = t1
    t1 = t
    
N_arr = np.zeros(N) 
 
for i in xrange(N):

    spike_train = generate_nonhomogeneous_poisson_0(r, Tr, r0 + A)
    
    hits = np.logical_and(t0 <= spike_train, spike_train <= t1)
    count = np.sum(hits)
    N_arr[i] = count
    
N_min = min(N_arr)
N_max = max(N_arr)

n_arr = np.arange(N_min, N_max)

p_arr_0 = np.zeros_like(n_arr)

# theoretical prediction for spike count statistics 
lam = quad(r, t0, t1)[0] # rate integral 
p_arr_1 = np.zeros_like(n_arr)

for i,n in enumerate(n_arr):
    
    count = np.sum(N_arr == n)
    p = float(count)/N
    p_arr_0[i] = p
    # calculated expected prop. from Poisson distribution
    p_arr_1[i] = poisson.pmf(n, lam)
    
#------------------------------------------------------------------------------ 
# plotting 

gs = plt.GridSpec(1,1)

ax0 = plt.subplot(gs[0])
ax0.bar(n_arr, p_arr_1, align = 'center', color = 'blue')
ax0.plot(n_arr, p_arr_0, 'x', color = 'red')

plt.show()









