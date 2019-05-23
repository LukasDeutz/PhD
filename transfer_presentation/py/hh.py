import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


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

def compute_derivatives(y, t, I0):
        
    # Input current (pA)
    def Id(t):
        if 1.0 < t < 2.0:
            return I0
        return 0.0
        
    dy = np.zeros((4,))
    
    Vm = y[0]
    n = y[1]
    m = y[2]
    h = y[3]
    
    # dVm/dt
    GK  = gK/Cm * np.power(n, 4.0)
    GNa = gNa/Cm * np.power(m, 3.0) * h
    GL  = gL/Cm
    
    dy[0] = Id(t)/Cm  - GK*(Vm-VK) - GNa*(Vm-VNa) - GL*(Vm-VL)
    
    # dn/dt
    dy[1] = alpha_n(Vm) * (1.0 - n) - (beta_n(Vm) * n)
    
    # dm/dt
    dy[2] = alpha_m(Vm) * (1.0 - m) - (beta_m(Vm) * m)
    
    # dh/dt
    dy[3] = alpha_h(Vm) * (1.0 - h) - beta_h(Vm) * h
    
    return dy

def plot_V(y_arr, T):
    
    # Input current (pA)
    def Id(t, I):
        if 1.0 < t < 2.0:
            return I
        return 0.0
    
    plt.figure(figsize = (14, 8))
    gs = plt.GridSpec(3,1)    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    
    for i, I in enumerate(I_arr):
    
        # Input stimulus
        Idv = [Id(t, I) for t in T]
                
        ax0.plot(T, Idv)
        
        v_arr = y_arr[i]
        
        ax1.plot(T, v_arr[:, 0])
        ax2.plot(T, v_arr[:, 0])
        
    ax0.set_xlabel('Time (ms)')
    ax0.set_ylabel(r'Current density (uA/$cm^2$)')
    ax0.grid(True)
            
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Vm (mV)')
    plt.grid(True)    
    
    ax2.set_xlim([0.5, 2.0])
    ax2.set_ylim([0, 30])  
    
    plt.show()  

if __name__ == '__main__':

    # model parameter    
    gK = 36.0 # Average potassium channel conductance per unit area (mS/cm^2)
    gNa = 120.0 # Average sodoum channel conductance per unit area (mS/cm^2)
    gL = 0.3 # Average leak channel conductance per unit area (mS/cm^2)
    Cm = 1.0 # Membrane capacitance per unit area (uF/cm^2)
    VK = -12.0 # Potassium potential (mV)
    VNa = 115.0 # Sodium potential (mV)
    VL = 10.613 # Leak potential (mV) 

    # Start and end time (in milliseconds)
    tmin = 0.0
    tmax = 50.0
    T = np.linspace(tmin, tmax, 10000)
      
    # State (Vm, n, m, h)
    Y = np.array([VL, n_inf(0), m_inf(0), h_inf(0)])
    
    # amplitudes of input current 
    I_arr = np.array([0])
    
    # caculate response 
    y_arr = []
    
    for I in I_arr:
        
        y_arr.append(odeint(compute_derivatives, Y, T, args = (I,)))
        
    plot_V(y_arr, T)
    
