'''
This file contains functions for evaluating ground states using both mathematica interface and python
'''

import numpy as np
import sympy as sp
import scipy.optimize as opt
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
from wolframclient.language import wlexpr
from mathematica_interface import *
from helper import *

def find_ground_state_mathematica(H_sum, coupling_constants, coupling_constants_n, num_spins, num_neighbors, method='Minimize', x0=None, extra_args=['Method -> {"SimulatedAnnealing"}']):
    '''
    Given a spin chain Hamiltonian in the form H = sum(H_sum) where the sum is on spins, and a list of coupling contants and their values, and the number of spins, computes the ground state spin configuration. Periodic boundary conditions are applied at the boundaries of spin chain.

    Issue: this function is quite slow as a result of the Mathematica interface - pretty much  all the time is devoted to the one evaluation step. This seems unavoidable if we are to use the Mathematica function.

    args:
        - H_sum:    sympy representation of Hamiltonian/Free energy to solve or a Mathematica appropriate string
        - coupling_constants:   list of symbolic coupling constants
        - coupling_constants_n: list of associated values
        - num_spins:    number of spins in the system
        - num_neighbors:    number of nearest neighbor interactions in the Hamiltonian
        - method:   string, Mathematica function to use for minimization. Should be either Minimize of FindMinimum
        - x0:       initial state for each spin in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn,Szn]], gets converted to [theta1,...,thetan,phi1,...,phin]

    returns:
        - groundstate:     minimum energy state in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn,Szn]]
    '''
    # check valid method
    valid_methods = ['Minimize', 'FindMinimum']
    if method not in valid_methods:
        raise ValueError(f'{method} not a valid method. Please choose from {valid_methods}')
    
    # setup initial state
    if x0 is None:
        x0=[0 for i in range(2*num_spins)]
    else:
        x0 = spin_state_to_angles(x0)
    thetas0 = x0[:num_spins]
    phis0 = x0[num_spins:]

    # obtain Hamiltonian function as Mathematica appropriate string
    if type(H_sum) is not str:
        H_mathematica = get_mathematica_H(H_sum, num_spins, num_neighbors)
    else:
        H_mathematica = H_sum
    for i in range(len(coupling_constants)):
        H_mathematica = H_mathematica.replace(str(coupling_constants[i]), str(coupling_constants_n[i]))

    # obtain input for mathematica minimize function
    if method=='FindMinimum':
        angles_input="{{theta"+str(0)+', '+python_num_to_mathematica(thetas0[0])+"}"
        for i in range(1,num_spins):
            #print(python_num_to_mathematica(thetas0[i]))
            angles_input = angles_input + ", {theta"+str(i)+", "+python_num_to_mathematica(thetas0[i])+"}"
        for i in range(num_spins):
            angles_input = angles_input + ", {phi"+str(i)+", "+python_num_to_mathematica(phis0[i])+"}"
        angles_input = angles_input + "}"
    elif method=='Minimize':
        angles_input = str(tuple([theta(i) for i in range(num_spins)]+[phi(i) for i in range(num_spins)]))
        angles_input = angles_input.replace('[','').replace(']','').replace('(','{').replace(')','}')
    mathematica_input = angles_input
    #print(mathematica_input)
    
    # initiate mathematica sessions and minimize Hamiltonian
    additional_inputs = ''
    for i in extra_args:
        additional_inputs = additional_inputs+f', {i}'
    math_str = f'{method}[{H_mathematica}, {mathematica_input+additional_inputs}]'
    #print(math_str)
    #t0 = time.time()
    mathematica_session=WolframLanguageSession()
    #t1 = time.time()
    opt = mathematica_session.evaluate(wlexpr(math_str))
    #t2 = time.time()
    mathematica_session.terminate()
    #tf = time.time()
    #print(f'Time to open Mathematica connection: {t1-t0}')
    #print(f'Time to evaluate Mathematica: {t2-t1}')
    #print(f'Time to close Mathematica connection: {tf-t2}')
    #print(f'Time to complete mathematica computation: {tf-t0}')

    #print(opt)
    angles = [mathematica_num_to_python(str(opt[1][i][1])) for i in range(2*num_spins)]
    angles = normal_rotate_state(angles)
    angles = redefine_angles(angles)
    #print(angles)
    groundstate = angles_to_spin_state(angles)

    return groundstate

def find_ground_state_python(H_num, coupling_constants, coupling_constants_n, num_spins, num_neighbors, x0=None, args=(), method=None):
    '''
    Given a spin chain Hamiltonian in the form H = sum(H_sum) where the sum is on spins, and a list of coupling contants and their values, and the number of spins, computes the ground state spin configuration. Periodic boundary conditions are applied at the boundaries of spin chain.

    args:
        - H_sum:    sympy representation of Hamiltonian/Free energy to solve or a Mathematica appropriate string
        - coupling_constants:   list of symbolic coupling constants
        - coupling_constants_n: list of associated values
        - num_spins:    number of spins in the system
        - num_neighbors:    number of nearest neighbor interactions in the Hamiltonian
        - x0:       initial state for each spin in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn,Szn]], gets converted to [theta1,...,thetan,phi1,...,phin]

    returns:
        - groundstate:     minimum energy state in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn,Szn]]
    '''
    
    # setup initial state
    if x0 is None:
        x0 = np.array([0 for i in range(2*num_spins)])
    else:
        x0 = spin_state_to_angles(x0)

    # get numerical Hamiltonian as function of angles
    H_angles = lambda angles: H_num(*coupling_constants_n, *angles_to_spin_state(angles).flatten())
    #print(H_angles(x0))

    # minimize H_angles
    res = opt.minimize(H_angles, x0, method=method)

    angles = [i for i in res.x]
    angles = normal_rotate_state(angles)
    angles = redefine_angles(angles)
    groundstate = angles_to_spin_state(angles)

    return groundstate

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