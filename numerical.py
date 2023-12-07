'''
this file contains functions for numerically calculating disperion relations and converting from symbolic to numerical functions
'''
import numpy as np
import sympy as sp
import scipy.linalg as lin
from pyLLspin.groundstates import *
from pyLLspin.analytical_lleq import *
import numba

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
    subs_list2 = [(Sx(i), Ss(3*i)) for i in range(num_spins)]+[(Sy(i), Ss(3*i+1)) for i in range(num_spins)]+[(Sz(i), Ss(3*i+2)) for i in range(num_spins)]
    H_tot = H_tot.subs(subs_list2)
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

def get_numerical_driving_term(driving_term, coupling_constants, num_spins):
    '''
    compute a numerical function for LLG driving term given symbolic driving term.

    args:
        - driving_term:    analytical expression for right hand side of LLG equation
        - coupling_constants:   list of symbolic coupling constants
        - num_spins:    number of spins in system.
    
    returns:
        - driving_term_num:  numerical function driving_term_num with arguments (*coupling_constants, *state, alpha), alpha being the damping constant
    '''
    state_vars = []
    for i in range(num_spins):
        state_vars.append(Sx(i))
        state_vars.append(Sy(i))
        state_vars.append(Sz(i))
    driving_term_num = sp.lambdify(coupling_constants+state_vars+[alpha], driving_term, 'numpy')
    return driving_term_num

def get_numerical_coord_transf(transf, num_spins):
    '''
    Obtain numerical coordinate transformation from analytical one (generated by get_analytical_ll_matrix_transf)

    return function transf(*groundstate) such that to get the coordinate transformation you evaluate transf and then multiply by state to rotate.
    '''
    groundstate_vars = []
    for i in range(num_spins):
        groundstate_vars.append(Sx0(i))
        groundstate_vars.append(Sy0(i))
        groundstate_vars.append(Sz0(i))
    transf_num = sp.lambdify(groundstate_vars, transf, 'numpy')
    return transf_num