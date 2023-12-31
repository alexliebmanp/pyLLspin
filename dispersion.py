'''
this file contains functions for numerically calculating disperion relations and converting from symbolic to numerical functions
'''
import numpy as np
import sympy as sp
import scipy.linalg as lin
from pyLLspin.groundstates import *
from pyLLspin.analytical_lleq import *
from pyLLspin.numerical import *

def compute_lswt(H_single, H_sum, coupling_constants,  coupling_constants_n, k_vect, num_spins, num_neighbors, M_num=None, M=None, groundstate=None, method='Minimize', x0=None, extra_args=['Method -> {"SimulatedAnnealing"}']):
    '''
    macro function to carry out full computation of dispersion relation for a spin Hamiltonian within linear spin wave theory.

    Issues: all of the sympy substitution functions are slooooow. As such, I should redesign this to use them as little as possible. This is highly doable and will greatly improve performance.

    NEEDS WORK

    args:
        - 
    returns:
        - 
    '''
    # obatain ground state if necessary
    if groundstate is None:
        #print('computing ground state')
        groundstate = find_ground_state_mathematica(H_sum, coupling_constants, coupling_constants_n, num_spins, num_neighbors, method=method, x0=x0, extra_args=extra_args)

    # compute analytical matrix if necessary
    if M is None:
        print('computing M')
        M, transf = get_analytical_ll_matrix(H_single, num_spins, num_neighbors)

    #t0 = time.time()
    if M_num is None:
        M_num = get_numerical_ll_matrix(M, coupling_constants, num_spins)
    Mk = lambda k: M_num(*coupling_constants_n, *groundstate.flatten(), k)

    dispersion, eigenvects = get_dispersion(k_vect, Mk)

    return dispersion, eigenvects, groundstate

def get_dispersion(k_vect, Mk):
    '''
    Given a matrix function Mk computes real eigenvalues and associated eigenvectors. returns in order (mode #, k, spin #)

    args:
        - k_vect:   array of k values for which to compute spectrum of Mk
        - Mk:   function that returns M(k)

    returns:
        - energies: eigenvalues of M(k)
        - vects:    eigenvectors of M(k)
    '''

    n_num = Mk(0).shape[0]
    energies = 1j*np.zeros((n_num,len(k_vect)))
    eigenvectors = 1j*np.zeros((n_num, len(k_vect), n_num))
    #energies = []
    #eigenvectors = []
    for i, k_num in enumerate(k_vect):
        m = Mk(k_num)
        eigen_system = lin.eig(m)
        eigen = eigen_system[0]
        vectors = eigen_system[1]
        sort_ind = np.argsort(np.real(eigen))
        vects = vectors[:,sort_ind].transpose()
        ens = eigen[sort_ind]
        #energies.append(ens)
        #eigenvectors.append(vects)
        energies[:,i] = ens
        eigenvectors[:,i,:] = vects
        #for ii in range(n_num):
        #    eigenvectors[ii,i,:] = vects[:,ii]
    return energies, eigenvectors

def transv_flucs_to_local_coords(mode):
    '''
    Given a transverse fluctuation mode, return full vector in local coordinate (ie, add in a local z coordinate equal to 0)
    '''

    num_spins = int(len(mode)/2)
    fullmode = 1j*np.zeros(3*num_spins)
    for ii in range(num_spins):
        fullmode[3*ii] = mode[2*ii]
        fullmode[3*ii+1] = mode[2*ii+1]
    return fullmode

def get_mode_cartesian(mode, transf):
    '''
    Given a mode which only gives the transverse fluctuations and a transformation from local coordinates to cartesian, returns cartesian mode in regular state representation, [[Sx1, Sy1, Sz1],...,[Sxn, Syn, Szn]]
    '''
    fullmode = transv_flucs_to_local_coords(mode)
    return np.matmul(transf, fullmode).reshape((int(len(fullmode)/3),3))
