'''
The purpose of this folder is to host a variety of helper functions

'''

import numpy as np
import sympy as sp

##############################################
### Define some sympy generating functions ###
##############################################

# constants
N, hbar, t, p, k, omega, lattice_const, alpha, gamma = sp.symbols('N, hbar, t, p, k, omega, a_0, alpha, gamma')
n = sp.symbols('n', cls=sp.Idx)

# indexed variable generators
Sx = lambda i: sp.IndexedBase('S^x', real=True)[i]
Sy = lambda i: sp.IndexedBase('S^y', real=True)[i]
Sz = lambda i: sp.IndexedBase('S^z', real=True)[i]
S = lambda i: sp.Matrix([Sx(i), Sy(i), Sz(i)])
Sx0 = lambda i: sp.IndexedBase('S_0^x', real=True)[i]
Sy0 = lambda i: sp.IndexedBase('S_0^y', real=True)[i]
Sz0 = lambda i: sp.IndexedBase('S_0^z', real=True)[i]
S0 = lambda i: sp.Matrix([Sx0(i), Sy0(i), Sz0(i)])
dSx = lambda i: sp.IndexedBase('\delta S^x')[i]
dSy = lambda i: sp.IndexedBase('\delta S^y')[i]
dSz = lambda i: sp.IndexedBase('\delta S^z')[i]
dS = lambda i: sp.Matrix([dSx(i), dSy(i), dSz(i)])
ux = lambda i: sp.IndexedBase('u^x')[i]
uy = lambda i: sp.IndexedBase('u^y')[i]
uz = lambda i: sp.IndexedBase('u^z')[i]
u = lambda i: sp.Matrix([ux(i), uy(i), uz(i)])
Ss = lambda i: sp.IndexedBase('S')[i]
theta = lambda i: sp.IndexedBase('theta')[i]
phi = lambda i: sp.IndexedBase('phi')[i]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def redefine_angles(angles):
    '''
    redefines angles (thetas,phis) such that theta from 0, pi and phi from 0 to 2pi
    '''

    num_spins = int(len(angles)/2)
    thetas = angles[:num_spins]
    phis = angles[num_spins:]
    for i in range(num_spins):
        theta = thetas[i]%(2*np.pi)
        phi = phis[i]%(2*np.pi)
        if theta > np.pi:
            theta = np.pi-theta%np.pi
            phi = (phi + np.pi)%(2*np.pi)
        thetas[i] = theta
        phis[i] = phi
    return thetas+phis

def normal_rotate_state(angles):
    '''
    rotates states such that first spin has phi = 0
    '''
    num_spins = int(len(angles)/2)
    thetas = angles[:num_spins]
    phis = angles[num_spins:]
    phi0 = phis[0]
    for i in range(num_spins):
        phi = phis[i] - phi0
        phis[i] = phi
    return thetas+phis

def simultaneous_subs(expr, substitutions):
    """
    Custom substitution function that performs all substitutions at once.

    Args:
        expr (sympy.Basic): The expression to substitute variables in.
        substitutions (dict): A dictionary where keys are variables to substitute,
            and values are the substitutions.

    Returns:
        sympy.Basic: The expression with variables simultaneously substituted.
    """
    # Create a temporary dictionary to store the substitutions
    temp_subs = {}
    final_subs = {}

    # Iterate through the substitutions dictionary
    iter = 0
    for var, value in substitutions.items():
        # Replace the variable with a temporary symbol
        #temp_val = sp.symbols(str(value)+'_temp')
        temp_var = sp.IndexedBase(var)[f'_{iter}']
        temp_subs[var] = temp_var
        final_subs[temp_var] = value
        iter+=1

    #print(expr)
    #print(temp_subs)
    #print(final_subs)
    # Replace the temporary symbols back to their original variables
    return expr.subs(temp_subs).subs(final_subs)

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
    takes state in form [[S1x,S1y,S1z],...,[Snx,Sny,Snz]] and puts it into angle form [theta1,...,thetan,phi1,...,phin] where thetas are the azimuthal angle. phi is the polar angle.
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