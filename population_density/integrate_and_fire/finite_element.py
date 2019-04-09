import numpy as np
import matplotlib.pyplot as plt


def finite_element_2(N):
        
    h = 1./(N-1)
    
    A = np.diag(2./3*np.ones(N))
    A[0,0] = 1./3
    A[-1,-1] = 1./3
        
    A = A + np.diag(1./6*np.ones(N-1), k = 1)
    A = A + np.diag(1./6*np.ones(N-1), k = -1)
    A = h*A
    
    B = np.zeros((N, N-1))    
    np.fill_diagonal(B, -1)
    np.fill_diagonal(B[1:, :], 1)
    
    C = B.T
    
    D = np.zeros((N-1,N-1))
    
    Block = np.bmat([[A, B], [C, D]])
    
    f = np.zeros(2*N - 1)
    f[N:] = -h
    
    x = np.linalg.solve(Block, f)
    sig = x[:N]
    u   = x[N:]
        
    x_u   = h*np.arange(0.5, N-1)
    x_sig = h*np.arange(0, N)
        
    return u, x_u, sig, x_sig


def finite_element(n):

    h = 1./n # mesh size 
    
    # create diagonal matrix
    K = np.diag(2.*np.ones(n-1))
    # fill off diagonals
    K = K + np.diag(-1.*np.ones(n-2), k = 1)
    K = K + np.diag(-1.*np.ones(n-2), k = -1)
    
    K = K/h
        
    # Force vector if f=1                           
    F = h*np.ones(n-1)
    
    x = h*np.arange(1, n)
    u = np.linalg.solve(K, F)

    return x, u

def analytical_result(x):
        
    u = -0.5*x**2 + 0.5*x
    
    return u
    

def plot_fe_1():
    
    n_arr = [10, 50, 100]
        
    ax0 = plt.subplot(111)

    x_theo = np.linspace(0, 1, 1e3)    
    u_theo = analytical_result(x_theo)
    
    gs = plt.GridSpec(len(n_arr), 2)
         
    for i, n in enumerate(n_arr):

        # Plot function
        axi0 = plt.subplot(gs[i, 0])
        
        x, u = finite_element(n)        
        axi0.plot(x_theo, u_theo, label = 'analytics', ls = '-', c = 'k')
        axi0.plot(x, u, 'o--', label = '%s' % str(n), c = 'r', mfc = 'None', ms = 5, mec = 'k')
        axi0.grid(True)
        axi0.set_ylabel('u', fontsize = '20')
        axi0.legend()
        
        # Plot error        
        axi1 = plt.subplot(gs[i, 1])
        err  = u - analytical_result(x)
        axi1.plot(x, err, 'k')
        axi1.set_xlabel('x', fontsize = 20)
        axi1.set_ylabel('err', fontsize = 20)

    axi0.set_xlim([0, 1])
    axi0.set_xlabel('x', fontsize = '20')
    
    axi1.set_xlim([0, 1])
    axi1.set_xlabel('x', fontsize = '20')
    
    plt.show()    
    

def plot_fe_2():

    N = 20
    
    u, x_u, sig, x_sig = finite_element_2(N)

    x_theo = np.linspace(0, 1, 1e3)    
    u_theo = analytical_result(x_theo)
    
    plt.plot(x_theo, u_theo, '-', c = 'k')
    plt.plot(x_u, u, '--o', c = 'r')
    
    plt.show()
    
if __name__ == '__main__':
    
    plot_fe_2()


