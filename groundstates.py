'''
This file contains functions for evaluating ground states using both mathematica interface and python
'''

import numpy as np
import sympy as sp
import scipy.optimize as opt
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
from wolframclient.language import wlexpr
from pyLLspin.mathematica_interface import *
from pyLLspin.helper import *
from pyLLspin.numerical import *
from pyLLspin.diff_eq_solver import *
import time
import numba

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

def find_ground_state_python(H_num, coupling_constants_n, num_spins, x0=None, method=None, min_kwargs={}):
    '''
    Given a spin chain Hamiltonian in the form H = sum(H_sum) where the sum is on spins, and a list of coupling contants and their values, and the number of spins, computes the ground state spin configuration. Periodic boundary conditions are applied at the boundaries of spin chain.

    args:
        - H_num:    python function to minimize, representing Hamiltonian. Its arguments are of the form (*coupling_constants, Sx1, Sy1, Sz1, ...., Sxn, Syn, Szn).
        - coupling_constants:   list of symbolic coupling constants - included for now just for consistency with Mathematica function
        - coupling_constants_n: list of associated values
        - num_spins:    number of spins in the system
        - num_neighbors:    number of nearest neighbor interactions in the Hamiltonian
        - x0:       initial state for each spin in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn,Szn]], gets converted to [theta1,...,thetan,phi1,...,phin]
        - method:   method to use with opt.minimize

    returns:
        - groundstate:     minimum energy state in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn,Szn]]
    '''
    
    # setup initial state
    if x0 is None:
        x0 = np.array([0 for i in range(2*num_spins)])
    else:
        x0 = spin_state_to_angles(x0)

    # get numerical Hamiltonian as function of angles
    #H_angles = lambda angles: H_num(*coupling_constants_n, *angles_to_spin_state(redefine_angles(normal_rotate_state(angles))).flatten())
    H_angles = lambda angles: H_num(*coupling_constants_n, *angles_to_spin_state(angles).flatten())
    #print(H_angles(x0))

    # minimize H_angles
    bounds = np.array([(0,2*np.pi) for i in range(2*num_spins)])
    if method not in ['basinhopping', 'differential_evolution', 'shgo', 'dual_annealing']:
        res = opt.minimize(H_angles, x0, bounds=bounds, **min_kwargs)
    elif method=='basinhopping':
        res = opt.basinhopping(H_angles, x0, minimizer_kwargs={'bounds':bounds}, **min_kwargs)
    elif method=='differential_evolution':
        res = opt.differential_evolution(H_angles, bounds=bounds, x0=x0, **min_kwargs)
    elif method=='shgo':
        res = opt.shgo(H_angles, bounds, **min_kwargs)
    elif method=='dual_annealing':
        res = opt.dual_annealing(H_angles, bounds=bounds, x0=x0, **min_kwargs)

    angles_opt = [i for i in res.x]
    #angles_opt = normal_rotate_state(angles_opt)
    #angles_opt = redefine_angles(angles_opt)
    groundstate = angles_to_spin_state(angles_opt)

    return groundstate

def find_ground_state_llg(driving_term_num, coupling_constants_n, x0, alpha, dt=0.1, Ns=10000):
    '''
    wrapper function for just finding the ground state via run_llg_simulation.

    args:
        - driving_term_num: function which evaluates RHS of LLG equation with inputs f(*coupling_constants_n, *state, alpha, gamma)
        - coupling_constants_n: numerical coupling constants of free energy
        - x0:  initial state in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn, Szn]]
        - gamma:   gyromagnetic ratio
        - alpha:   phenomenological damping
        - dt:      simulation time step
        - Ns       number of time steps

    returns:
        - final_state: final state in simulation of form [[Sx1, Sy1, Sz1],...,[Sxn, Syn, Szn]]
    '''
    times, states = run_llg_sim(driving_term_num, coupling_constants_n, x0, alpha, dt, Ns)
    final_state = states[-1]
    return final_state

def run_llg_sim(driving_term_num, coupling_constants_n, x0, alpha, dt=0.1, Ns=10000):
    '''
    function of running an LLG simulation using numerical driving term function, ie numerically solve

    dM/dt = -gamma*M\crossB_eff - alpha*M\cross(M\crossB_eff)

    args:
        - driving_term_num: function which evaluates RHS of LLG equation with inputs f(*coupling_constants_n, *state, alpha, gamma)
        - coupling_constants_n: numerical coupling constants of free energy
        - x0:  initial state in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn, Szn]]
        - gamma:   gyromagnetic ratio
        - alpha:   phenomenological damping
        - dt:      simulation time step
        - Ns       number of time steps

    returns:
        - times: simulation time vector
        - states:   (Ns, nspins, 3) array representing state at each time step.
    '''
    nspins = len(x0)
    x0_flat = x0.flatten()
    G = lambda t, state: driving_term_num(*coupling_constants_n, *state, alpha)
    times, states_flat = rkode(G, 0, x0_flat, dt, Ns)
    states = np.reshape(list(states_flat), (Ns, nspins, 3))
    return times, states

@numba.jit(nopython=False)
def find_ground_state_llg_numba(driving_term_num, coupling_constants_n, x0, alpha, dt=0.1, Ns=10000):
    '''
    wrapper function for just finding the ground state via run_llg_simulation.

    args:
        - driving_term_num: function which evaluates RHS of LLG equation with inputs f(*coupling_constants_n, *state, alpha, gamma)
        - coupling_constants_n: numerical coupling constants of free energy
        - x0:  initial state in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn, Szn]]
        - gamma:   gyromagnetic ratio
        - alpha:   phenomenological damping
        - dt:      simulation time step
        - Ns       number of time steps

    returns:
        - final_state: final state in simulation of form [[Sx1, Sy1, Sz1],...,[Sxn, Syn, Szn]]
    '''
    times, states = run_llg_sim_numba(driving_term_num, coupling_constants_n, x0, alpha, dt, Ns)
    final_state = states[-1]
    return final_state

@numba.jit(nopython=False)
def run_llg_sim_numba(driving_term_numba, coupling_constants_n, x0, alpha, dt=0.1, Ns=10000):
    '''
    function of running an LLG simulation using numerical driving term function, ie numerically solve

    dM/dt = -gamma*M\crossB_eff - alpha*M\cross(M\crossB_eff)

    args:
        - driving_term_num: function which evaluates RHS of LLG equation with inputs f(t, state, *args) where args = [*coupling_constants_n, alpha]
        - coupling_constants_n: numerical coupling constants of free energy
        - x0:  initial state in form [[Sx1, Sy1, Sz1],...,[Sxn, Syn, Szn]]
        - gamma:   gyromagnetic ratio
        - alpha:   phenomenological damping
        - dt:      simulation time step
        - Ns       number of time steps

    returns:
        - times: simulation time vector
        - states:   (Ns, nspins, 3) array representing state at each time step.
    '''
    nspins = len(x0)
    x0_flat = x0.flatten()
    times, states_flat = rkode_llg_numba(driving_term_numba, 0, x0_flat, dt, Ns, coupling_constants_n, alpha)
    states = np.reshape(list(states_flat), (Ns, nspins, 3))
    return times, states