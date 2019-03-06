import numpy as np
import matplotlib.pyplot as plt

from mpmath import pcfu, pcfv, pcfw, exp
from scipy.special import pbvv, pbwa


# a = -0.5
# N = 1000
# x_arr = np.linspace(-9, 3, N)
# 
# v_arr = np.zeros_like(x_arr)
# u_arr = np.zeros_like(x_arr)
# 
# for i,x in enumerate(x_arr):
#     
#     v_arr[i] = exp(-0.25*x**2)*pcfv(a, x)
#     u_arr[i] = exp(-0.25*x**2)*pcfu(a, x)
# 
# plt.plot(x_arr, v_arr, label = '$V(x)$') 
# plt.plot(x_arr, u_arr, label = '$U(x)$')
# 
# plt.legend()
# plt.show()


a_arr = np.array([-0.5, -2., -3.5, -5, -6.5, -8.0])


gs = plt.GridSpec(2, 2)

ax0 = plt.subplot(gs[0])
ax0.set_title('U(x)')
ax0.set_xlim([-10, 10])
ax0.set_ylim([-6, 6])

ax1 = plt.subplot(gs[1])
ax1.set_title('V(x)')
ax1.set_xlim([-10, 10])
ax1.set_ylim([-6, 6])

ax2 = plt.subplot(gs[2])
ax2.set_title('f(x)U(x)')
ax2.set_xlim([-10, 10])
ax2.set_ylim([-6, 6])

ax3 = plt.subplot(gs[3])
ax3.set_title('f(x)V(x)')
ax3.set_xlim([-10, 10])
ax3.set_ylim([-6, 6])

N = 1000
x_arr = np.linspace(-20, 20, N)
 
for a in a_arr:
    
    v_arr = np.zeros_like(x_arr)
    u_arr = np.zeros_like(x_arr)
 
    for i,x in enumerate(x_arr):
     
        v_arr[i] = pcfv(a, x)
        u_arr[i] = pcfu(a, x)
    
    ax0.plot(x_arr, u_arr, label = 'a=%.1f' % a)
    ax0.legend()
    ax1.plot(x_arr, v_arr, label = 'a=%.1f' % a)
    ax1.legend()
        
for a in a_arr:
    
    v_arr = np.zeros_like(x_arr)
    u_arr = np.zeros_like(x_arr)
 
    for i,x in enumerate(x_arr):
     
        v_arr[i] = exp(-0.25*x**2)*pcfv(a, x)
        u_arr[i] = exp(-0.25*x**2)*pcfu(a, x)
    
    ax2.plot(x_arr, u_arr, label = 'a=%.1f' % a)
    ax2.legend()
    ax3.plot(x_arr, v_arr, label = 'a=%.1f' % a)
    ax3.legend()

plt.show()    



# pbvv_arr = np.zeros_like(x_arr, dtype = np.complex)
# 
# for i,x in enumerate(x_arr):
# 
#     pbvv_arr[i] = pbvv(a, x)

# pbvv_arr = pbwa(x_arr, a)
# 
# plt.plot(x_arr, pbvv_arr[0], label = 'scipy')
# plt.plot(x_arr, pcfv_arr, ls = '--', label = 'mpmath')
# 
# plt.legend()
# plt.show()

