import matplotlib.pyplot as plt
import numpy as np 


def synaptic_current(t, J, tau_s, tau_r):
    
    I = np.zeros_like(t)
    
    idx_arr = t > 0
    
    I[idx_arr] = (np.exp(-t[idx_arr]/tau_r) - np.exp(-t[idx_arr]/tau_s))/(tau_r - tau_s) 

    return I


if __name__ == '__main__':

    fig_path = '../figures/'

    fz = 18
    lfz = 16

    J = 0.5 
    tau_s = 2 
    tau_r_arr = [0.0, 0.5, 1.0, 1.99]

    t = np.linspace(-2.0, 10.0, 1000)
    
    ax = plt.subplot(111)
    
    for tau_r in tau_r_arr:
        
        I = synaptic_current(t, J, tau_s, tau_r)
    
        ax.plot(t, I, label = u'$\tau_{r} = $'  + '%.1f' % tau_r)
    
    ax.grid(True)
    ax.set_xlabel('$t$ in ms', fontsize = 18)
    ax.set_ylabel('$I(t)$ in mV', fontsize = 18)
    ax.legend(fontsize = lfz)
    ax.set_xlim([-2.0, 10.0])
    ax.set_ylim([0.0, 0.55])
    
    plt.savefig(fig_path + 'current_based_synapse.png')
    
    plt.show()