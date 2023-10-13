'''
This file contains functions for analytically obtaining expressions for ll matrix representing the differential equation

dM/dt = M x Beff

'''
import numpy as np
import sympy as sp
sp.init_printing()
from helper import *

def get_analytical_ll_matrix_transv(H_single, num_spins, num_neighbors):
    '''
    Computes an analytical Landau-Lifshitz coupled linearized matrix equation dS/dt = -(S cross Beff) for harmonic ansatz, where S is the magnetization at each lattice site as a column vector, ie S = (S1x,S1y,S1z,..,Snx,Sny,Snz). Considers only transverse fluctuations.
    
    That is, this function returns the matrix M in the equation M*m = omega*m such that the eigenvalues of M are mode frequencies of Hamiltonian H_single.

    args:
        - H_single: sympy representation of free energy of spin n in the spin chain, explicitly including all interacting neighbors.
        - num_spins:    number of spin for which to compute the matrix
        - num_neighbors:    number of neighors included in the H_single

    return:
        - M:    analytical expression for M such that M*m = omega*m
        - transf:   analytical transformation from local coordinates to xyz
    '''

    # compute effective field in LLG formalism as derivative of H with respect to S in cartesian coordinates
    B_effx = sp.diff(H_single, Sx(n))
    B_effy = sp.diff(H_single, Sy(n))
    B_effz = sp.diff(H_single, Sz(n))
    B_eff = sp.Matrix([B_effx, B_effy, B_effz])

    # from effective field, compute the right hand side of LL equation of motion in cartesian coordinates
    ll = -S(n).cross(B_eff)

    # obtain LL equation in local coordinate system for each spin
    subs_list_rot = {}
    for i in range(-num_neighbors, num_neighbors+1):
        #theta_ni = sp.acos(Sz0(n+i)/sp.sqrt(Sx0(n+i)**2 + Sy0(n+i)**2 + Sz0(n+i)**2))
        #phi_ni = sp.acos(Sx0(n+i)/sp.sqrt(Sx0(n+i)**2 + Sy0(n+i)**2))
        theta_ni = sp.atan2(sp.sqrt(Sx0(n+i)**2 + Sy0(n+i)**2), Sz0(n+i))
        phi_ni = sp.atan2(Sy0(n+i), Sx0(n+i))
        Rot_inv_ni = sp.rot_axis3(-phi_ni)*sp.rot_axis2(-theta_ni)
        Rot_ni = sp.rot_axis2(theta_ni)*sp.rot_axis3(phi_ni)
        S_rot = list(Rot_inv_ni*S(n+i))
        subs_list_rot[Sx(n+i)] = S_rot[0]
        subs_list_rot[Sy(n+i)] = S_rot[1]
        subs_list_rot[Sz(n+i)] = S_rot[2]
    #theta_n = sp.acos(Sz0(n)/sp.sqrt(Sx0(n)**2 + Sy0(n)**2 + Sz0(n)**2))
    #phi_n = sp.acos(Sx0(n)/sp.sqrt(Sx0(n)**2 + Sy0(n)**2))
    theta_n = sp.atan2(sp.sqrt(Sx0(n)**2 + Sy0(n)**2), Sz0(n))
    phi_n = sp.atan2(Sy0(n), Sx0(n))
    Rot_inv_n = sp.rot_axis3(-phi_n)*sp.rot_axis2(-theta_n)
    Rot_n = sp.rot_axis2(theta_n)*sp.rot_axis3(phi_n)
    ll_rot = (-(S(n)).cross(simultaneous_subs(Rot_n*B_eff,subs_list_rot)))
    
    # replace S with (dSx, dSy, 1) where dSx and dSy are small fluctuating component transverse to z (in rotated frame)
    subs_listx = [(Sx(n+i), dSx(n+i)) for i in range(-num_neighbors, num_neighbors+1)]
    subs_listy = [(Sy(n+i), dSy(n+i)) for i in range(-num_neighbors, num_neighbors+1)]
    subs_listz = [(Sz(n+i), 1) for i in range(-num_neighbors, num_neighbors+1)]
    subs_list = subs_listx+subs_listy+subs_listz
    ll_fluc = sp.expand(ll_rot.subs(subs_list))

    # linearize the equations of motion
    var_listx = [dSx(n+i) for i in range(-num_neighbors, num_neighbors+1)]
    var_listy = [dSy(n+i) for i in range(-num_neighbors, num_neighbors+1)]
    var_list = var_listx+var_listy
    ll_lin = vector_linearize(ll_fluc, var_list)

    # plug in harmonic ansatz
    ansatzx = [(dSx(n+i), ux(n+i)*sp.exp(sp.I*i*k*lattice_const)) for i in range(-num_neighbors, num_neighbors+1)]
    ansatzy = [(dSy(n+i), uy(n+i)*sp.exp(sp.I*i*k*lattice_const)) for i in range(-num_neighbors, num_neighbors+1)]
    ansatz = ansatzx+ansatzy
    ll_lin_ansatz = ll_lin.subs(ansatz)*(-sp.I)

    # solve for matrix element of eigenoperator for which we want to diagonalize, taking typical convention for ordering of spin components, ie u = (ux1, uy1, ux2, uy2, ..., uxN, uyN) and periodic boundary conditions
    u_vect = sp.Matrix([element for sublist in [[ux(i), uy(i)] for i in range(num_spins)] for element in sublist])
    ll_lin_vect = []
    for j in range(num_spins):
        subs_listx2 = [(ux(n+i), ux(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listy2 = [(uy(n+i), uy(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_list2 = subs_listx2+subs_listy2
        subs_listx3 = [(Sx0(n+i), Sx0(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listy3 = [(Sy0(n+i), Sy0(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listz3 = [(Sz0(n+i), Sz0(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_list3 = subs_listx3+subs_listy3+subs_listz3
        for ii, element in enumerate(ll_lin_ansatz):
            if ii%3!=2:
                ll_lin_vect.append(element.subs(subs_list2).subs(subs_list3))
    transf = sp.Matrix(np.zeros((3*num_spins, 3*num_spins)))
    for j in range(num_spins):
        subs_list4 = [(Sx0(n), Sx0(sp.Mod(j, num_spins))), (Sy0(n), Sy0(sp.Mod(j, num_spins))), (Sz0(n), Sz0(sp.Mod(j, num_spins)))]
        transf[3*j:3*(j+1),3*j:3*(j+1)] = Rot_inv_n.subs(subs_list4)
    ll_lin_vect = sp.Matrix(ll_lin_vect)
    M = generate_matrix(u_vect, ll_lin_vect)

    return M, transf

def get_analytical_ll_matrix(H_single, num_spins, num_neighbors):
    '''
    Computes an analytical Landau-Lifshitz coupled linearized matrix equation dS/dt = -(S cross Beff) for harmonix ansatz, where S is the magnetization at each lattice site as a column vector, ie S = (S1x,S1y,S1z,..,Snx,Sny,Snz). Note that this explicity keeps longitudinal fluctuations.
    
    That is, this function returns the matrix M in the equation M*m = omega*m such that the eigenvalues of M are mode frequencies of Hamiltonian H_single.

    args:
        - H_single: sympy representation of free energy of spin n in the spin chain, explicitly including all interacting neighbors.
        - num_spins:    number of spin for which to compute the matrix
        - num_neighbors:    number of neighors included in the H_single

    return:
        - M:    analytical expression for M such that M*m = omega*m
    '''

    # compute effective field in LLG formalism as derivative of H with respect to S
    B_effx = sp.diff(H_single, Sx(n))
    B_effy = sp.diff(H_single, Sy(n))
    B_effz = sp.diff(H_single, Sz(n))
    B_eff = sp.Matrix([B_effx, B_effy, B_effz])

    # from effective field, compute the right hand side of LL equation of motion
    ll = -S(n).cross(B_eff)

    # replace S with S0 + dS where dS is some small fluctuating component and S0 is the equilibrium magnetization.
    subs_listx = [(Sx(n+i), Sx0(n+i)+dSx(n+i)) for i in range(-num_neighbors, num_neighbors+1)]
    subs_listy = [(Sy(n+i), Sy0(n+i)+dSy(n+i)) for i in range(-num_neighbors, num_neighbors+1)]
    subs_listz = [(Sz(n+i), Sz0(n+i)+dSz(n+i)) for i in range(-num_neighbors, num_neighbors+1)]
    subs_list = subs_listx+subs_listy+subs_listz
    ll_fluc = sp.expand(ll.subs(subs_list))
    ll_fluc

    # linearize the equations of motion
    var_listx = [dSx(n+i) for i in range(-num_neighbors, num_neighbors+1)]
    var_listy = [dSy(n+i) for i in range(-num_neighbors, num_neighbors+1)]
    var_listz = [dSz(n+i) for i in range(-num_neighbors, num_neighbors+1)]
    var_list = var_listx+var_listy+var_listz
    ll_lin = vector_linearize(ll_fluc, var_list)

    # plug in harmonic ansatz
    ansatzx = [(dSx(n+i), ux(n+i)*sp.exp(sp.I*i*k*lattice_const)) for i in range(-num_neighbors, num_neighbors+1)]
    ansatzy = [(dSy(n+i), uy(n+i)*sp.exp(sp.I*i*k*lattice_const)) for i in range(-num_neighbors, num_neighbors+1)]
    ansatzz = [(dSz(n+i), uz(n+i)*sp.exp(sp.I*i*k*lattice_const)) for i in range(-num_neighbors, num_neighbors+1)]
    ansatz = ansatzx+ansatzy+ansatzz
    ll_lin_ansatz = ll_lin.subs(ansatz)*(-sp.I)

    # solve for matrix element of eigenoperator for which we want to diagonalize, taking typical convention for ordering of spin components, ie u = (ux1, uy1, uz1, ux2, uy2, uz2, ..., uxN, uyN, uzN) and periodic boundary conditions
    u_vect = sp.Matrix([element for sublist in [[ux(i), uy(i), uz(i)] for i in range(num_spins)] for element in sublist])
    ll_lin_vect = []
    for j in range(num_spins):
        subs_listx2 = [(ux(n+i), ux(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listy2 = [(uy(n+i), uy(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listz2 = [(uz(n+i), uz(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_list2 = subs_listx2+subs_listy2+subs_listz2
        subs_listx3 = [(Sx0(n+i), Sx0(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listy3 = [(Sy0(n+i), Sy0(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_listz3 = [(Sz0(n+i), Sz0(sp.Mod(j+i, num_spins))) for i in range(-num_neighbors, num_neighbors+1)]
        subs_list3 = subs_listx3+subs_listy3+subs_listz3
        for element in ll_lin_ansatz:
            ll_lin_vect.append(element.subs(subs_list2).subs(subs_list3))
    ll_lin_vect = sp.Matrix(ll_lin_vect)
    M = generate_matrix(u_vect, ll_lin_vect)

    return M

def generate_matrix(a, b):
    '''
    finds matrix M in equation b = Ma given a and b using sympy.
    '''

    M = sp.Matrix(np.zeros((len(a), len(a))))
    for ii, i in enumerate(a):
        for jj, j in enumerate(a):
            row = sp.expand(b[ii])
            elem = 0
            for arg in row.args:
                if check_proprtional_to(arg, j):
                    elem = elem + arg/j
            M[ii,jj] = elem

    return M

def vector_linearize(expr, variables):
    '''
    convenience function to linearize a vector equation.
    '''
    expr_lin = []
    for exp in expr:
        expr_lin.append(scalar_linearize(exp, variables))
    return sp.Matrix(expr_lin)

def scalar_linearize(expr, variables):
    '''
    linearizes a sympy expression to keep only first order terms in variables. Assumes the expression is a scalar and that the top level expr.func is Add. In the future I can design this to handle top level Mult function.
    '''
    expr = sp.expand(expr)
    if expr.func != sp.core.add.Add:
        raise ValueError('input expression must be an Add object.')
    args = list(expr.args)
    args_lin = []
    for arg in args:
        order = find_order_recurse(arg, variables)
        if order > 1:
            pass
        else:
            args_lin.append(arg)
    return sum(args_lin)

def find_order_recurse(expr, variables):
    '''
    recursively finds order of variables in expr. Basically, we walk down the expression tree until we either find a leaf args=() or we find a member of variables.
    '''
    order = 0
    if expr.func==sp.core.power.Pow:
        power = expr.args[1]
    else:
        power = 1
    for arg in expr.args:
        if arg in variables:
            order = order + 1
        else:
            order = order + find_order_recurse(arg, variables)
    return order*power

def check_proprtional_to(expr, val):
    '''
    checks it expr is proprtional to val. DOES NOT check for linear proportionality so this should always be used for linear expressoins.
    '''
    prop = 0
    for arg in expr.args:
        if val==arg:
            prop = 1
        else:
            prop = prop + check_proprtional_to(arg, val)
    return bool(prop)