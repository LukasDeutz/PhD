import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------ 
# Interspike interval (ISI) 

lam = 2. # rate
N = 10000 #  

t = np.random.exponential(1./lam, N) # draw N ISI's

avg_t = np.mean(t) # ISI trial average
std_t = np.std(t) # ISI trial std

exp_t = 1./lam # expected ISI assuming a Poisson process
var_t = 1./lam**2 # variance ISI assuming a Poisson process
exp_std_t = np.sqrt(var_t)


print 'ISI drawn from exponential distribution'
print 'ISI trial average: %.3f' % avg_t
print 'ISI expactation value: %.3f' % exp_t
print 'Relative deviation: %.3f' % (np.abs(avg_t - exp_t)/exp_t)
print 
print 'ISI trial std: %.3f' % std_t
print 'ISI expected std: %.3f' % exp_std_t
print 'Relative deviation: %.3f' % (np.abs(std_t - exp_std_t)/exp_std_t)

plt.figure()

h, _, _ = plt.hist(t, bins = 100, density = True, facecolor = 'b', edgecolor = 'b')
plt.vlines([exp_t, exp_t - exp_std_t, exp_t + exp_std_t], 0, np.max(h), linestyles = ['-', '--', '--'], label = ['exp', 'avg'])

#------------------------------------------------------------------------------ 
# Spike counts
   
N = 10000 # number of trials
T = 100 # recording time  
 
trials = []
 
# generate spike times for N trials using that ISI's are exponentially distributed
for i in xrange(N):
 
    t = np.random.exponential(1./lam) # occurance of first spike
     
    spike_times = []
 
    while t < T: # record spike times until t > T
     
        spike_times.append(t)
        t += np.random.exponential(1./lam) 
 
    trials.append(spike_times)

spike_counts = np.zeros(N)

# plot spike trains
fig = plt.figure()
gs = plt.GridSpec(1,2)
ax0 = plt.subplot(gs[0])  

for i, trial in enumerate(trials):
 
    ax0.plot(trial, i*np.ones_like(trial), 'o', color = 'b', markersize = 0.1)
    spike_counts[i] = len(trial) # spike count

avg_n = np.mean(spike_counts) # spike count trial average
std_n = np.std(spike_counts) # spike count trial std

exp_n = T*lam # expected number of spikes
exp_std_n = np.sqrt(T*lam) # expected number of spikes

print
print 'Spike count trial average: %.3f' % avg_n
print 'Spike count expactation value: %.3f' % exp_n
print 'Relative deviation: %.3f' % (np.abs(avg_n - exp_n)/exp_n)
print 
print 'Spike count std: %.3f' % std_n
print 'Spike count expected std: %.3f' % exp_std_n
print 'Relative deviation: %.3f' % (np.abs(std_n - exp_std_n)/exp_std_n)

ax0.set_xlabel('t')
ax0.set_ylabel('trial')
 
ax1 = plt.subplot(gs[1]) 
ax1.fill_betweenx(np.arange(N), 0, spike_counts, color = 'b')
ax1.set_xlabel('count')
 
#------------------------------------------------------------------------------ 
# Plot distribution of counts 
 
plt.figure()
 
ax = plt.subplot(111)
h, _, _ = ax.hist(spike_counts, bins = 100)
plt.vlines([exp_n, exp_n - exp_std_n, exp_n + exp_std_n], 0, np.max(h), linestyles = ['-', '--', '--'], label = ['exp', 'avg'])

#------------------------------------------------------------------------------ 
# Alterantively, just draw spike counts from Poisson distributed and distribute spike times uniformly

T = 5000 # choose larger recording time to match figure 1


n = np.random.poisson(lam*T) # spike count

spike_times = np.sort(np.random.uniform(0, T, n))

t = np.diff(spike_times)

print
print 'Uniformly distributed spike times'
print 'ISI trial average: %.3f' % avg_t
print 'ISI expactation value: %.3f' % exp_t
print 'Relative deviation: %.3f' % (np.abs(avg_t - exp_t)/exp_t)
print 
print 'ISI trial std: %.3f' % std_t
print 'ISI expected std: %.3f' % exp_std_t
print 'Relative deviation: %.3f' % (np.abs(std_t - exp_std_t)/exp_std_t)

avg_t = np.mean(t) # ISI trial average
std_t = np.std(t) # ISI trial std

plt.figure()

h, _, _ = plt.hist(t, bins = 100, density = True, facecolor = 'b', edgecolor = 'b')
plt.vlines([exp_t, exp_t - exp_std_t, exp_t + exp_std_t], 0, np.max(h), linestyles = ['-', '--', '--'], label = ['exp', 'avg'])

plt.show()

