'''
This file contains functions for visualizing states, modes, and dispersion relations

'''
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
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

def plot_mode(k, freq, mode_vect, equilibrium_vect, period_fraction=0.5, lattice_const=1, length=1, aspect_factor=4, elev=7.5, azim=-90, fig=None, ax=None, cmap=plt.get_cmap('cet_rainbow'), colorbar=True):
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

    lattice_positions = np.array([np.array([0,0,i*lattice_const]) for i in range(ns)])
    period = 2*np.pi/freq
    ts = np.linspace(0,period_fraction*period,100)
    timestep = period_fraction*np.linspace(0, 1, len(ts))
    fluctuations = np.zeros((ns,len(ts), 3))
    for ii, t in enumerate(ts):
        spinconfig= get_mode_snapshot(k, t, freq, mode_vect, equilibrium_vect)
        for n in range(ns):
            fluctuations[n, ii] = lattice_positions[n] + spinconfig[n]

    x_positions = lattice_positions[:,0]
    y_positions = lattice_positions[:,1]
    z_positions = lattice_positions[:,2]
    for ii, s in enumerate(equilibrium_vect):
        fact = 1.1
        xs = np.array([x_positions[ii], x_positions[ii]+s[0]])
        ys = np.array([y_positions[ii], y_positions[ii]+s[1]])
        zs = np.array([z_positions[ii], z_positions[ii]+s[2]])
        arrow = Arrow3D(xs, ys, zs, 
                lw=2, mutation_scale=5, arrowstyle="-|>", color="r")
        ax.add_artist(arrow)
    ax.plot(np.concatenate([x_positions,[0]]),np.concatenate([y_positions, [0]]),np.concatenate([z_positions,[(ns+1)*lattice_const]]), '--', color='black')

    fact=1
    xmin = np.min(x_positions.flatten())-lattice_const*fact
    xmax = np.max(x_positions.flatten())+lattice_const*fact
    ymin = np.min(y_positions.flatten())-lattice_const*fact
    ymax = np.max(y_positions.flatten())+lattice_const*fact
    zmin = np.min(z_positions.flatten())-lattice_const*fact
    zmax = np.max(z_positions.flatten())+lattice_const*fact
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.set_box_aspect((1/aspect_factor,1/aspect_factor,1))
    ax.axis('off')

    if elev!=90:
        for i in range(ns):
            circle = Circle((0, 0), length, facecolor=(0, 0, 1, 0.25))
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=i*lattice_const, zdir="z")
            scatter = ax.scatter(fluctuations[i,:,0], fluctuations[i,:,1], fluctuations[i,:,2], 'o', c=timestep, cmap=cmap, s=1)
    if elev==90:
        fract = 0.2
        fract2 = 0.3
        tol = 0.1
        bias = np.array([-1, -1])*0.1
        annotation_locs = []
        for ii, s in enumerate(equilibrium_vect[:]):
            sx, sy, sz = s
            s_dir = np.array([sx, sy])/np.sqrt(sx**2 + sy**2)
            dsx0 = fract*s_dir[0] + bias[0]
            dsy0 = fract*s_dir[1] + bias[1]
            sx = sx + dsx0
            sy = sy + dsy0
            dsx = 0
            dsy = 0
            if ii==0:
                annotation_locs.append([sx, sy, 0])
            else:
                count=0
                for jj in range(len(annotation_locs)):
                    ss = annotation_locs[jj]
                    if np.abs(sx-ss[0]) <= tol and np.abs(sy-ss[1])<= tol:
                        count+=1
                        ss[2]+=1
                        dsx = ss[2]*fract2*s_dir[0]
                        dsy = ss[2]*fract2*s_dir[1]
                        sx = sx + dsx
                        sy = sy + dsy
                    if count==0:
                        annotation_locs.append([sx, sy, 0])
            for n in range(ns):
                scatter = ax.scatter(fluctuations[n,:,0], fluctuations[n,:,1], fluctuations[n,:,2], 'o', c=timestep, cmap=cmap, s=1)
            ax.text(sx, sy, ns*lattice_const, str(ii+1))

    if colorbar==True:
        fig.colorbar(scatter, label='Fraction of Period')

    ax.view_init(elev, azim)
    if showplot:
        plt.show()
    
def plot_state(state, lattice_const=1, length=1, aspect_factor=4, elev=7.5, azim=-90, fig=None, ax=None, colors=None): 
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

    z_positions = np.arange(0, ns*lattice_const, lattice_const)
    for ii, s in enumerate(state):
        fact=1.1
        xs = np.array([0, s[0]*fact])
        ys = np.array([0, s[1]*fact])
        zs = np.array([z_positions[ii], z_positions[ii]+s[2]])
        if colors is not None:
            col = colors[ii]
        else:
            col = 'r'
        arrow = Arrow3D(xs, ys, zs, 
                lw=2, mutation_scale=5, arrowstyle="-|>", color=col)
        ax.add_artist(arrow)

    ax.set_box_aspect((1/aspect_factor,1/aspect_factor,1))
    ax.axis('off')
    ax.plot([0,0],[0,0],[0,lattice_const*ns], '--', color='black')

    lat_range=1
    xmin = -(lat_range)*lattice_const
    xmax = (lat_range)*lattice_const
    ymin = -(lat_range)*lattice_const
    ymax = (lat_range)*lattice_const
    zmin = np.min(z_positions.flatten())-lattice_const
    zmax = np.max(z_positions.flatten())+lattice_const
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    if elev!=90:
        for i in range(ns):
            circle = Circle((0, 0), length, facecolor=(0, 0, 1, 0.25))
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=i*lattice_const, zdir="z")
    if elev==90:
        fract = 0.1
        fract2 = 0.2
        tol = 0.1
        bias = np.array([-1, -1])*0.1
        annotation_locs = []
        for ii, s in enumerate(state[:]):
            sx, sy, sz = s
            s_dir = np.array([sx, sy])/np.sqrt(sx**2 + sy**2)
            sx = sx + fract*s_dir[0] + bias[0]
            sy = sy + fract*s_dir[1] + bias[1]
            if ii==0:
                annotation_locs.append([sx, sy, 0])
            else:
                count=0
                for jj in range(len(annotation_locs)):
                    ss = annotation_locs[jj]
                    if np.abs(sx-ss[0]) <= tol and np.abs(sy-ss[1])<= tol:
                        count+=1
                        ss[2]+=1
                        sx = sx + ss[2]*fract2*s_dir[0]
                        sy = sy + ss[2]*fract2*s_dir[1]
                    if count==0:
                        annotation_locs.append([sx, sy, 0])
            ax.text(sx, sy, ns*lattice_const, str(ii+1))

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
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)