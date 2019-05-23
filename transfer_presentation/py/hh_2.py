import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint

# Average potassium channel conductance per unit area (mS/cm^2)
gK = 36.0

# Average sodoum channel conductance per unit area (mS/cm^2)
gNa = 120.0

# Average leak channel conductance per unit area (mS/cm^2)
gL = 0.3

# Membrane capacitance per unit area (uF/cm^2)
Cm = 1.0

# Potassium potential (mV)
VK = -12.0

# Sodium potential (mV)
VNa = 115.0

# Leak potential (mV)
Vl = 10.613

# Potassium ion-channel rate functions

def alpha_n(Vm):
    return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)

def beta_n(Vm):
    return 0.125 * np.exp(-Vm / 80.0)

# Sodium ion-channel rate functions

def alpha_m(Vm):
    return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)

def beta_m(Vm):
    return 4.0 * np.exp(-Vm / 18.0)

def alpha_h(Vm):
    return 0.07 * np.exp(-Vm / 20.0)

def beta_h(Vm):
    return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)
  
# n, m, and h steady-state values

def n_inf(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))

def m_inf(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))

def h_inf(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))
  

# Compute derivatives
def compute_derivatives(y, t0, I0):
    dy = np.zeros((4,))
    
    if 5 < t0 < 6:
        I = I0
    else:
        I = 0.
            
    Vm = y[0]
    n = y[1]
    m = y[2]
    h = y[3]
    
    # dVm/dt
    GK = (gK / Cm) * np.power(n, 4.0)
    GNa = (gNa / Cm) * np.power(m, 3.0) * h
    GL = gL / Cm
    
    dy[0] = (I / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl))
    
    # dn/dt
    dy[1] = (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)
    
    # dm/dt
    dy[2] = (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)
    
    # dh/dt
    dy[3] = (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)
    
    return dy
  
def plot_threshold():  

    # Start and end time (in milliseconds)
    tmin = 0.0
    tmax = 50

    # Time values
    t = np.linspace(tmin, tmax, 10000)

    # State (Vm, n, m, h)    
    v0 = np.array([0.0, n_inf(), m_inf(), h_inf()])
    # Input current amplitude
    
    I0_arr = np.arange(4, 8, 1)
    
    inch = 2.54
    fig = plt.figure(figsize = ())
    ax  = plt.subplot(111)
    
    for I0 in I0_arr:
        
        v_mat = odeint(compute_derivatives, v0, t, args = (I0,))
        V = v_mat[:, 0]
        
        ax.plot(t, V)


    t_min = 4
    t_max = 20
    v_th = 5.5

    ax.plot([t_min, t_max], [v_th,v_th], '--', c = 'k')        
    ax.plot([t_min, t_max], [0.0,0.0], '-', c = 'k')            
    
    ax.set_xlim([t_min, t_max])
    ax.set_ylim(-15, 20)
    ax.set_xlabel('time [ms]', fontsize = 16)
    ax.set_ylabel('membrane potential [mV]', fontsize = 16)
    
    fig_path = '../figures/'
    figname  = 'hh_threshold.svg'
        
    plt.savefig(figname)
        
    return
  
def plot_action_potential():
    
    # Start and end time (in milliseconds)
    tmin = 0.0
    tmax = 50

    # Time values
    T = np.linspace(tmin, tmax, 10000)

    # State (Vm, n, m, h)    
    Y = np.array([0.0, n_inf(), m_inf(), h_inf()])
    # Input current amplitude
    I0 = 50

    v_arr = odeint(compute_derivatives, Y, T, args = (I0,))

    def Id(t):
        if 5 < t < 6:
            return I0
        return 0.0

    # Input stimulus
    Idv = [Id(t) for t in T]
    
    plt.figure(figsize=(12, 7))
    gs = plt.GridSpec(4,1)    

    fz = 12
    lfz = 12
    
    ax0 = plt.subplot(gs[0])
    ax0.plot(T, Idv)
    ax0.set_ylabel(r'$I$ [pA]', fontsize = fz)
    ax0.set_xticks([])
    ax0.grid(True)
    ax0.set_xlim(T[0], T[-1])

    V = v_arr[:, 0]

    ax1 = plt.subplot(gs[1])
    ax1.plot(T, V, c = 'k')
    ax1.set_ylabel('Vm (mV)', fontsize = fz)
    ax1.grid(True)
    ax1.set_xlim(T[0], T[-1])
    ax1.plot([T[0], T[-1]], [0, 0], ls = '--', c = 'k')
    ax1.grid(True)
    
    n = v_arr[:, 1]
    m = v_arr[:, 2]
    h = v_arr[:, 3]
    
    ax2 = plt.subplot(gs[2])    
    ax2.plot(T, n**4, c = 'g', label = u'$n^4$')
    ax2.plot(T, m**3*h, c = 'y', label = u'$m^3 h$')
    ax2.set_xlim(T[0], T[-1])
    ax2.set_ylabel('gating variables', fontsize = fz)
    ax2.legend(fontsize = lfz)
    ax2.grid(True)

    I_K = n**4*gNa*(V - VNa)
    I_NA = m**3*h*gK*(V - VK)

    ax3 = plt.subplot(gs[3])    
    ax3.plot(T, I_K, c = 'g', label = u'$I_{\text{K}}$')    
    ax3.plot(T, I_NA, c = 'y', label = u'$I_{\text{Na}}$')
    ax3.legend(fontsize = lfz)
    ax3.set_ylabel('$I_{\text{K}}/I_{text{Na}}$', fontsize = fz)
    ax3.set_xlim(T[0], T[-1])
    ax3.set_xlabel('time [ms]', fontsize = fz)
    ax3.grid(True)

    plt.show()

if __name__ == '__main__':  
  
#     plot_action_potential()
    plot_threshold()






