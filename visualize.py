'''
This file contains functions for visualizing states, modes, and dispersion relations

'''
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle
from pyLLspin.helper import *

def get_mode_snapshot(k, t, freq, mode_vect, equilibrium_vect):
    '''
    computes spin configuration for mode with frequency freq at wavevector k and time t.

    args:
        - k:        wavevector of mode
        - t:        time at which to evaluate spin config
        - freq:     frequency of mode
        - mode_vect:    eigenvector of mode
        - equilibrium_vect:  equilibrium position
    return:
        - spinconfig:   spin configuration at specified time
    '''

    n_num = int(len(equilibrium_vect))
    spinconfig = np.zeros((n_num, 3))
    for n in range(n_num):
        mode = np.real(mode_vect[n]*np.exp(-1j*freq*t))
        equilib = equilibrium_vect[n]
        spinconfig[n,:] = equilib + mode
    return spinconfig

def plot_mode(k, freq, mode_vect, equilibrium_vect, period_fraction=0.25, lattice_const=0.11, length=0.1, factor=8, elev=7.5, azim=-90, fig=None, ax=None):
    '''
    plots spin configuration for mode at k_num and time.

    args:
        - k:        wavevector of mode
        - freq:     frequency of mode
        - mode_vect:    eigenvector of mode
        - equilibrium_vect:  equilibrium position
        - period_faction:        fraction of period over which to display modes
        - elev:
        -azim:
    '''

    if ax is None:
        fig = plt.figure()
        ax = plt.figure().add_subplot(projection='3d')
        showplot=True
    else:
        ax.remove()
        ax = fig.add_subplot(ax.get_subplotspec(), projection='3d')
        showplot=False

    ns = mode_vect.shape[0]
    mode_vect = length*mode_vect
    equilibrium_vect = length*equilibrium_vect
    xx, yy, zz = np.meshgrid(np.arange(0, 1, 1), np.arange(0, 1, 1), np.arange(0, ns*lattice_const, lattice_const))
    vv_eq = equilibrium_vect[:,1]
    ww_eq = equilibrium_vect[:,2]
    uu_eq = equilibrium_vect[:,0]

    lattice_positions = [np.array([0,0,i*lattice_const]) for i in range(ns)]
    period = 2*np.pi/freq
    ts = np.linspace(0,period_fraction*period,100)
    fluctuations = np.zeros((len(ts), ns, 3))
    for ii, t in enumerate(ts):
        spinconfig= get_mode_snapshot(k, t, freq, mode_vect, equilibrium_vect)
        for n in range(ns):
            fluctuations[ii, n] = lattice_positions[n] + spinconfig[n]
    ax.scatter(fluctuations[:,:,0].flatten(), fluctuations[:,:,1].flatten(), fluctuations[:,:,2].flatten(), 'o', s=1, color='blue')

    ax.set_box_aspect((1/factor,1/factor,1))
    ax.quiver(xx, yy, zz, uu_eq, vv_eq, ww_eq, normalize=False, linewidths=3, color='red')
    ax.axis('off')
    ax.plot([0,0],[0,0],[0,lattice_const*ns], '--', color='black')

    if elev!=90:
        for i in range(ns):
            circle = Circle((0, 0), length, facecolor=(0, 0, 1, 0.25))
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=i*lattice_const, zdir="z")
    if elev==90:
        annotation_locs = []
        for ii, s in enumerate(equilibrium_vect):
            sx, sy, sz = s
            s_dir = np.array([sx, sy])/np.sqrt(sx**2 + sy**2)
            fract=0.023
            tol = 1e-2
            if (sx, sy) in annotation_locs:
                sx = sx + fract*s_dir[0]
                sy = sy + fract*s_dir[1]
                annotation_locs.append((sx, sy))
            else:
                annotation_locs.append((sx, sy))
            ax.annotate(ii+1, (sx/3, sy/3))

    ax.view_init(elev, azim)
    if showplot:
        plt.show()
    
def plot_state(state, lattice_const=0.11, length=0.1, factor=8, elev=7.5, azim=-90, fig=None, ax=None):
    '''
    for viewing magnetic states

    args:
        - elev:
        -azim:
    '''
    if ax is None:
        fig = plt.figure()
        ax = plt.figure().add_subplot(projection='3d')
        showplot=True
    else:
        ax.remove()
        ax = fig.add_subplot(ax.get_subplotspec(), projection='3d')
        showplot=False

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
        annotation_locs = []
        for ii, s in enumerate(state):
            sx, sy, sz = s
            s_dir = np.array([sx, sy])/np.sqrt(sx**2 + sy**2)
            fract=0.023
            tol = 1e-2
            if (sx, sy) in annotation_locs:
                sx = sx + fract*s_dir[0]
                sy = sy + fract*s_dir[1]
                annotation_locs.append((sx, sy))
            else:
                annotation_locs.append((sx, sy))
            ax.annotate(ii+1, (sx/3, sy/3))

    ax.view_init(elev, azim)
    if showplot:
        plt.show()

def plot_state_3D(state, lattice_vects=[np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])], lattice_const=0.11, length=0.1, factor=1, elev=7.5, azim=-90, fig=None, ax=None):
    '''
    for viewing magnetic states

    args:
        - lattice_vects:    list of vectors to use to associate lattice coordinat with spatial coordinate
        - elev:
        -azim:
    '''
    if ax is None:
        fig = plt.figure()
        ax = plt.figure().add_subplot(projection='3d')
        showplot=True
    else:
        ax.remove()
        ax = fig.add_subplot(ax.get_subplotspec(), projection='3d')
        showplot=False

    nx, ny, nz = state.shape[:-1]
    state = length*state
    xx = np.zeros((nx,ny,nz))
    yy = np.zeros((nx,ny,nz))
    zz = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pos = lattice_vects[0]*i + lattice_vects[1]*j + lattice_vects[2]*k
                xx[i,j,k] = pos[0]
                yy[i,j,k] = pos[1]
                zz[i,j,k] = pos[2]
    uu = state[:,:,:,0]
    vv = state[:,:,:,1]
    ww = state[:,:,:,2]

    ax.set_box_aspect((1/factor,1/factor,1))
    ax.quiver(xx, yy, zz, uu, vv, ww, normalize=False, linewidths=3, color='red')
    #ax.axis('off')
    '''
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pos = lattice_vects[0]*i + lattice_vects[1]*j + lattice_vects[2]*k

                ax.plot([0,0],[0,0],[0,lattice_const*nz], '--', color='black')
    '''

    '''
    if elev!=90:
        for i in range(ns):
            circle = Circle((0, 0), length, facecolor=(0, 0, 1, 0.25))
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=i*lattice_const, zdir="z")
    if elev==90:
        for ii, s in enumerate(state):
            sx, sy, sz = s
            ax.annotate(ii+1, (sx/3, sy/3))
    '''

    ax.view_init(elev, azim)
    if showplot:
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