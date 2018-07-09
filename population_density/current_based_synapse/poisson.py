import numpy as np
import matplotlib.pyplot as plt

N = 1000 # number of trials
T = 100 # recording time  
lam = 1 # rate

trials = []

# record spike times for N trials
for i in xrange(N):

    t = np.random.exponential(1./lam) # occurance of first spike
    
    spike_times = []

    while t < T: # record spike times until t > T
    
        spike_times.append(t)
        t += np.random.exponential(1./lam) 

    trials.append(spike_times)

# plot spike trains and number of spikes in each trial
fig = plt.figure()
gs = plt.GridSpec(1,2)
ax0 = plt.subplot(gs[0])  

spike_counts = np.zeros(N)

for i, trial in enumerate(trials):

    ax0.plot(trial, i*np.ones_like(trial), 'o', color = 'b', markersize = 0.5)
    spike_counts[i] = np.sum(trial)
    
ax0.set_xlabel('t')
ax0.set_ylabel('trial')

ax1 = plt.subplot(gs[1]) 
ax1.fill_betweenx(np.arange(N), 0, spike_counts, color = 'b')
ax1.set_xlabel('count')
ax0.set_ylabel('trial')


#------------------------------------------------------------------------------ 
# Distribution of counts 

plt.figure()

ax = plt.subplot(111)

ax.hist(spike_counts)

# Trial statistics
avg_count = np.mean(spike_counts)
std_count = np.std(spike_counts)

# Theory
exp_count_theory = lam*T
std_count_theory = np.sqrt(exp_count_theory)

print np.abs(avg_count - exp_count_theory)/exp_count_theory
print np.abs(std_count - std_count_theory)/std_count_theory


plt.show()

