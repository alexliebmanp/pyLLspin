import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import numba


# Runge-Kutta
# solves df/dx = G(x, f(x))
#@numba.jit
def rkode(G, x0, f0, dx, Ns):

    f = np.zeros((Ns, len(f0)))
    x = np.zeros(Ns)
    xn=x0
    fn=f0
    f[0,:] = f0
    x[0] = x0
    for i in np.arange(1, Ns):

        ### change for various modes here ###

        #### 4th order RK ###
        k1 = dx*G(xn, fn)
        k2 = dx*G(xn + dx/2, fn + k1/2)
        k3 = dx*G(xn + dx/2, fn + k2/2)
        k4 = dx*G(xn + dx, fn + k3)
        fn = fn + (1/3)*(k1/2 + k2 + k3 + k4/2)
        xn = xn + dx
        f[i,:] = fn
        x[i] = xn

        #####################################

    return x, f


# Runge-Kutta
# solves df/dx = G(x, f) for specific LLG code driving force G(*coupling_constants, *f, alpha)
@numba.jit(nopython=False)
def rkode_llg_numba(G, x0, f0, dx, Ns, coupling_constants, alpha):

    f = np.zeros((Ns, len(f0)))
    x = np.zeros(Ns)
    xn=x0
    fn=f0
    f[0,:] = f0
    x[0] = x0
    coupling_constants = tuple(coupling_constants)
    for i in np.arange(1, Ns):

        ### change for various modes here ###

        #### 4th order RK ###
        k1 = dx*G(*coupling_constants, *tuple(fn), alpha)
        k2 = dx*G(*coupling_constants, *tuple(fn + k1/2), alpha)
        k3 = dx*G(*coupling_constants, *tuple(fn + k2/2), alpha)
        k4 = dx*G(*coupling_constants, *tuple(fn + k3), alpha)
        fn = fn + (1/3)*(k1/2 + k2 + k3 + k4/2)
        xn = xn + dx
        f[i,:] = fn
        x[i] = xn

        #####################################

    return x, f


# solves df/dx = G(x, f(x)) - (does it really?) modify for everything
#@numba.jit(nopython=True)
def finite_difference(z0, dt, dx, Nt, random_noise, h0, g0, m0):

    nx = len(z0)
    z = np.zeros((Nt, nx))
    t = np.zeros(Nt)

    z[0] = z0
    z[1] = z0
    t[0] = 0
    nstart = 1
    for n in range(nstart,Nt):
        t[n] = t[n-1] + dt
        for i in range(nx):


            # periodic boundary conditions
            ip = np.mod(i+1,nx)
            im = i-1
            rn = g0*random_noise_calc(i, z[n-1,i], random_noise)

            #z[n, i] = z[n-1, i] + dt*((z[n-1,ip] - 2*z[n-1,i] + z[n-1, im])/dx**2) + dt*ac_driving((n-1)*dt,h0) + g0*dt*random_noise_calc(i, z[n-1,i], random_noise)

            z[n, i] = (1/(m0+dt))*((2*m0+dt)*z[n-1,i] - m0*z[n-2,i] + dt**2*(z[n-1,ip] - 2*z[n-1,i] + z[n-1, im])/dx**2 + dt**2*ac_driving((n-1)*dt,h0) + (dt**2)*rn)

            #print(z[n,i])


        # pin x=0 to z=0
        #z[n, int(nx/2)] = 0

    return t, z
