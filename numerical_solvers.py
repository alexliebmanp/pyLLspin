'''
this file contains functions for numerically calculating disperion relations and converting from symbolic to numerical functions
'''
import numpy as np
import sympy as sp
import scipy.linalg as lin
from groundstates import *
from analytical_solvers import *

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

    n_num = Mk(1).shape[0]
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

def get_numerical_ll_matrix(M, coupling_constants, num_spins):
    '''
    compute a numerical function for M given symbolic M.

    args:
        - M:    symbolic M matrix
        - coupling_constants:   list of symbolic coupling constants
        - num_spins:    number of spins in system.
    
    returns:
        - M_num:  numerical function M with arguments (*coupling_constants, *groundstate)
    '''
    groundstate_vars = []
    for i in range(num_spins):
        groundstate_vars.append(Sx0(i))
        groundstate_vars.append(Sy0(i))
        groundstate_vars.append(Sz0(i))
    M_num = sp.lambdify(coupling_constants+groundstate_vars+[k], M, 'numpy')
    return M_num

def get_numerical_H(H_sum, coupling_constants, num_spins, num_neighbors):
    '''
    Given a spin chain Hamiltonian in the form H = sum(H_sum) where the sum is on spins, and a list of coupling contants, and the number of spins, returns numerical Hamiltonian functions for use with python. 
    args:
        - H_sum:    sympy representation of Hamiltonian/Free energy to solve
        - coupling_constants:   list of coupling constants
        - num_spins:    number of spins in the system
        - num_neighbors:    number of nearest neighbor interactions in the Hamiltonian
    '''
    # obtain numerican Hamiltonian functions (for both use in numpy and mathematica)
    H_tot = 0
    for j in range(num_spins):
        subs_listx = [(Sx(n+i), Sx(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listy = [(Sy(n+i), Sy(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listz = [(Sz(n+i), Sz(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_list = subs_listx+subs_listy+subs_listz
        for element in sp.expand(H_sum).args:
            H_tot = H_tot + element.subs(subs_list)
    subs_list6 = [(Sx(i), Ss(3*i)) for i in range(num_spins)]+[(Sy(i), Ss(3*i+1)) for i in range(num_spins)]+[(Sz(i), Ss(3*i+2)) for i in range(num_spins)]
    H_tot = H_tot.subs(subs_list6)
    func_input = tuple(coupling_constants+[Ss(i) for i in range(num_spins*3)])
    H_python = sp.lambdify(func_input, H_tot, 'numpy')

    return H_python

def get_mathematica_H(H_sum, num_spins, num_neighbors):
    '''
    Given a spin chain Hamiltonian in the form H = sum(H_sum) where the sum is on spins, and a list of coupling contants, and the number of spins, returns numerical Hamiltonian functions for use with Mathematica

    args:
        - H_sum:    sympy representation of Hamiltonian/Free energy to solve
        - coupling_constants:   list of coupling constants
        - num_spins:    number of spins in the system
        - num_neighbors:    number of nearest neighbor interactions in the Hamiltonian
    '''
    # obtain numerican Hamiltonian functions (for both use in numpy and mathematica)
    H_angles = 0
    for j in range(num_spins):
        subs_listx_angles = [(Sx(n+i), sp.sin(theta(sp.Mod(j+i, num_spins)))*sp.cos(phi(sp.Mod(j+i, num_spins)))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listy_angles = [(Sy(n+i), sp.sin(theta(sp.Mod(j+i, num_spins)))*sp.sin(phi(sp.Mod(j+i, num_spins)))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listz_angles = [(Sz(n+i), sp.cos(theta(sp.Mod(j+i, num_spins)))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_list_angles = subs_listx_angles+subs_listy_angles+subs_listz_angles
        for element in sp.expand(H_sum).args:
            H_angles = H_angles + element.subs(subs_list_angles)
        H_mathematica = str(H_angles).replace('**','^').replace('[','').replace(']','').replace('sin', 'Sin').replace('cos','Cos').replace('(','[').replace(')',']')

    return H_mathematica

def angles_to_spin_state(angles):
    '''
    takes state in form [theta1,...,thetan,phi1,...,phin] and puts it into form [[S1x,S1y,S1z],...,[Snx,Sny,Snz]].
    '''

    num_spins = int(len(angles)/2)
    thetas = angles[:num_spins]
    phis = angles[num_spins:]
    return np.array([[np.sin(thetas[i])*np.cos(phis[i]), np.sin(thetas[i])*np.sin(phis[i]), np.cos(thetas[i])] for i in range(num_spins)])

def spin_state_to_angles(state):
    '''
    takes state in form [[S1x,S1y,S1z],...,[Snx,Sny,Snz]] and puts it into angle form [theta1,...,thetan,phi1,...,phin]
    '''

    thetas = []
    phis = []
    for s in state:
        s_norm = s/np.sqrt(np.sum([s[i]**2 for i in range(3)])) # normalize state just in case
        sx = s_norm[0]
        sy = s_norm[1]
        sz = s_norm[2]
        thetas.append(np.arctan2(np.sqrt(sx**2 + sy**2),sz))
        phis.append(np.arctan2(sy,sx))
    return redefine_angles(thetas+phis)