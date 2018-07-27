import numpy as np
from scipy.optimize import root 
import matplotlib.pyplot as plt
from scipy.misc import factorial


def generate_nonhomogeneous_poisson_0(r, T, r0):
    
    N = np.random.poisson(r0*T)
    
    spike_times = np.sort(T*np.random.uniform(size = N))

    #thinning
    
    u = np.random.uniform(size = N)
    
    spike_times_0 = []
    
    for i, t in enumerate(spike_times):
        
        p = 1. - r(t)/r0
        if u[i] > p:
            spike_times_0.append(t)
    
    return np.array(spike_times_0)
    
def generate_nonhomogeneous_poisson_1(r_t, T):
    '''Generate spike train from nonhomogeneous Poisson process with rate function r_t for a recording time T
    param r_t: rate function
    param T: recording time
    return spike_times: array with spike times:'''

    t = 0 # initial time
    
    spike_times = []
            
    while t<T:
        
        # uniform random number
        u = np.random.uniform()
        
        # invert cdf to draw ISI 
        isi = root(lambda y: 1. - np.exp(-r_t(t+y) + r_t(t)) - u, 1./r_t(t)).x 
        t = isi + t # new spike time previous plus ISI
        spike_times.append(t)
    
    return np.array(spike_times)


#------------------------------------------------------------------------------ 
# parameter
    
tau = 1. # membrane time constant 

# define oscillating rate function 
T = 10.*tau # choose period large compared to the membrane time constannt
w = 2.*np.pi/T

A  = 2. # amplitude of oscillations 
r0 = 5.# stationary rate 

r = lambda t: A*np.sin(w*t) + r0 # rate function

Tr = 10000*T # recording time

#------------------------------------------------------------------------------ 
# generate spike trains and check statistics  

spike_train = generate_nonhomogeneous_poisson_0(r, Tr, r0 + A)

t0 = 0
t1  = 0.5*T

N_arr = []

while t1 <= Tr:
                        
    idx = np.logical_and(t0 < spike_train, spike_train <= t1)
    
    N_arr.append(len(spike_train[idx]))
    
    t0 = t0 + T
    t1 = t1 + T 

ax0 = plt.subplot(gs[0])

t = np.linspace(0, Tr, 100*Tr)
    
# ax0.hist(spike_train, 60)
ax0.plot(t, r(t), '-', color = 'red')
ax0.plot(spike_train, r0*np.ones_like(spike_train), 'o', color = 'blue', markersize = 0.5)

ax1 = plt.subplot(gs[1])

lam = A/w+r0*0.5*T

bins = np.arange(0.5, 5*lam, 1)
ax1.hist(N_arr, bins, density = True)

n = np.arange(0, round(5*lam), 1)
y = lam**(n)/factorial(n)*np.exp(-lam)

ax1.plot(n, y)

plt.show()









