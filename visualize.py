'''
This file contains functions for visualizing states, modes, and dispersion relations

'''
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle
from pyLLspin.helper import *

def get_mode(mode, k_num, time, k_vect, energies, eigenvectors):
    '''
    computes spin configuration for mode at k_num and time.

    args:
        - mode:     mode number 1-num_spins
        - k_num:    k at which to evaluate the mode
        - time:     time for which to evaluate the mode
        - k_vect:   k vector over which energies and eigenvectors are evaluated
        - energies: energy spectrum
        - eigenvectors: ...
    '''

    idx = find_nearest(k_vect, k_num)
    omega_num = energies[mode][idx]
    num_modes = len(energies)
    vect_full = eigenvectors[mode][idx]
    n_num = int(len(vect_full)/3)
    vectx = []
    vecty = []
    vectz = []
    for ii, val in enumerate(vect_full):
        if ii%3==0:
            vectx.append(val)
        if ii%3==1:
            vecty.append(val)
        else:
            vectz.append(val)
    Sx_vect = np.real([vectx[i]*np.exp(-1j*omega_num*time) for i in range(n_num)])
    Sy_vect = np.real([vecty[i]*np.exp(-1j*omega_num*time) for i in range(n_num)])
    Sz_vect = np.real([vectz[i]*np.exp(-1j*omega_num*time) for i in range(n_num)])
    Mag = [np.sum(Sx_vect), np.sum(Sy_vect), np.sum(Sz_vect)]
    return Sx_vect, Sy_vect, Sz_vect, Mag

def get_M_vs_t(mode, k_num, t_vect, k_vect, energies, eigenvectors):
    '''
    convenience function to evaluting M vs t. can be extended to output an function of Sx, Sy, Sz as a function of time (such as birefringence..)
    '''

    mx = np.zeros(len(t_vect))
    my = np.zeros(len(t_vect))
    mz = np.zeros(len(t_vect))
    for ii, t in enumerate(t_vect):
        Sx_vect, Sy_vect, Sz_vect, mag = get_mode(mode, k_num, t, k_vect, energies, eigenvectors)
        mx[ii] = mag[0]
        my[ii] = mag[1]
        mz[ii] = mag[2]

    return mx, my, mz

def plot_mode(mode, k_num, time, k_vect, energies, eigenvectors):
    '''
    plots spin configuration for mode at k_num and time.

    args:
        - mode:     mode number 1-num_spins
        - k_num:    k at which to evaluate the mode
        - time:     time for which to evaluate the mode
        - k_vect:   k vector over which energies and eigenvectors are evaluated
        - energies: energy spectrum
        - eigenvectors: ...
    '''

    Sx_vect, Sy_vect, Sz_vect, Mag = get_mode(mode, k_num, time, k_vect, energies, eigenvectors)
    fig, [ax1, ax2, ax3] = plt.subplots(1,3)
    num_atoms = int(len(eigenvectors[0][0])/2)
    p = [i for i in range(num_atoms)]
    ax1.quiver(p, np.ones(len(p)), Sx_vect, Sy_vect, color='red')
    ax2.quiver(p, np.ones(len(p)), Sx_vect, Sy_vect, color='blue')
    ax1.set_ylim(0.75,1.25)
    ax2.set_ylim(0.75,1.25)
    fig.suptitle(f'mode {mode}')
    plt.tight_layout()
    plt.show()

def plot_state(state, lattice_const=0.11, length=0.1, factor=8, elev=7.5, azim=-90):
    '''
    for viewing magnetic states

    args:
        - elev:
        -azim:
    '''
    fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')

    ns = state.shape[0]
    state = length*state
    xx, yy, zz = np.meshgrid(np.arange(0, 1, 1), np.arange(0, 1, 1), np.arange(0, ns*lattice_const, lattice_const))
    uu = state[:,0]
    vv = state[:,1]
    ww = state[:,2]

    ax.set_box_aspect((1/factor,1/factor,1))
    ax.quiver(xx, yy, zz, uu, vv, ww, normalize=False, linewidths=3, color='red')
    ax.axis('off')
    ax.plot([0,0],[0,0],[0,lattice_const*ns], '--', color='black')

    if elev!=90:
        for i in range(ns):
            circle = Circle((0, 0), length, facecolor=(0, 0, 1, 0.25))
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=i*lattice_const, zdir="z")
    if elev==90:
        for ii, s in enumerate(state):
            sx, sy, sz = s
            ax.annotate(ii+1, (sx/3, sy/3))

    ax.view_init(elev, azim)
    plt.show()

def check_fluctuations_transverse(groundstate, eigenvects, vmin=-10e-8, vmax=10e-8):
    '''
    One important check for any solutions to the LL equations which allow longitudinal fluctuations are that the normal modes are mainly transverse to the ground state at each spin site. Formally, if for each site the spin is S = S0 + dS, then we can check this by showing that dS.S0 = 0. Since the normal modes are harmonic in time, it suffices to show it for one time, t=0 being the most convenient. 

    Since we produce the vectors ds directly, all the input that is needed is the ground state and the eigenvectors for each mode.

    output is simply a display of 

    args:
        - groundstate:  list in form [[S1x, S1y, S1z],....,[Snx, Sny, Snz]]
        - dS:           ndarray of eigenvectors with each row corresponding to a different mode.
    returns:
        - dotprod:      ndarray of dS.S0, row indexes mode, column the spin
    '''

    fig, ax = plt.subplots(1)

    num_spins = len(groundstate)

    dotprod = []
    for mode in range(num_spins*3):

        eigenvect = eigenvects[mode].reshape(num_spins,3)
        vals = np.array([np.dot(np.real(eigenvect[i]), groundstate[i]) for i in range(num_spins)])
        dotprod.append(vals)

    dotprod = np.array(dotprod)

    im = ax.imshow(dotprod, aspect=1, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(-0.5, num_spins, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3*num_spins, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    fig.colorbar(im, label=r'$\delta S^i \cdot S^i_0$')

    ax.set(ylabel='Mode', xlabel=r'Spin site $i$')

    fig.tight_layout()

    return dotprod