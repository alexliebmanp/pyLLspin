'''
This file defines classes for defining a spin model for ground state and spin simulations. In addition to the object oriented approach, we'll write some import and export functions that will save models to files and be able to import them from files (to avoid pickeling things all the time)
'''

import numpy as np
import matplotlib.pyplot as plt
from pyLLspin.analytical_lleq import *
from pyLLspin.dispersion import *
from pyLLspin.numerical import *
from pyLLspin.groundstates import *
from pyLLspin.visualize import *
from pyLLspin.observables import *
from orenstein_analysis.measurement import helper
from tqdm import tqdm
import matplotlib.animation as animation
import importlib
import pickle
import glob
import re
import os

from scipy import fft as fft
from scipy import optimize as opt
from scipy import signal as sig
from scipy import integrate
from scipy import linalg as lin

match_dec = '[0-9]+.?[0-9]*'
match_dec = r'\d+\.\d+|\d+'


def create_spin_model(H_single, H_sum, num_spins, coupling_constants, coupling_constants_num, num_neighbors):
    '''
    from inputs, generates all necessary functions for working with a spin model.
    '''
    # setup various numerical functions that need be computed only once
    M, rot = get_analytical_ll_matrix_transv(H_single, num_spins, num_neighbors)
    driving_term = get_analytical_driving_term(H_single, num_spins, num_neighbors)
    rot_num = get_numerical_coord_transf(rot, num_spins)
    M_num = get_numerical_ll_matrix(M, coupling_constants, num_spins)
    H_num = get_numerical_H(H_sum, coupling_constants, num_spins, num_neighbors)
    H_math = get_mathematica_H(H_sum, num_spins, num_neighbors)
    driving_term_num = get_numerical_driving_term(driving_term, coupling_constants, num_spins)
    driving_term_numba = numba.jit(driving_term_num, nopython=False)
    H_num_numba = numba.jit(H_num, nopython=False)
    spin_model = {'num_spins':num_spins,
           'num_neighbors':num_neighbors,
           'coupling_constants':coupling_constants,
           'coupling_constants_num':coupling_constants_num,
           'H_single':H_single,
           'H_sum':H_sum,
           'M':M,
           'M_num':M_num,
           'H_num':H_num,
           'H_math':H_math,
           'rot':rot,
           'rot_num':rot_num,
           'driving_term':driving_term,
           'driving_term_num':driving_term_num,
           'driving_term_numba':driving_term_numba,
           'H_num_numba':H_num_numba}
    
    return spin_model

def load_spin_model(name, H_single=None, H_sum=None, num_spins=None, coupling_constants=None, num_neighbors=None, import_path='/Users/oxide/Documents/research/orenstein/code/spin_simulation/helical_spins/pickels/'):
    '''
    load spin model
    '''

    # quick load
    path = import_path
    ham=name
    with open(path+f'M_{ham}.pickle','rb') as f:
        M = pickle.load(f)
    with open(path+f'M_num_{ham}.pickle','rb') as f:
        M_num = pickle.load(f)
    with open(path+f'H_num_{ham}.pickle','rb') as f:
        H_num = pickle.load(f)
    with open(path+f'rot_{ham}.pickle','rb') as f:
        rot = pickle.load(f)
    with open(path+f'rot_num_{ham}.pickle','rb') as f:
        rot_num = pickle.load(f)
    with open(path+f'driving_term_{ham}.pickle','rb') as f:
        driving_term = pickle.load(f)
    with open(path+f'driving_term_num_{ham}.pickle','rb') as f:
        driving_term_num = pickle.load(f)
    with open(path+f'driving_term_numba_{ham}.pickle','rb') as f:
        driving_term_numba = pickle.load(f)
    with open(path+f'H_math_{ham}.pickle','rb') as f:
        H_math = pickle.load(f)
    if H_single is None:
        with open(path+f'H_single_{ham}.pickle','rb') as f:
            H_single = pickle.load(f)
    if H_sum is None:
        with open(path+f'H_sum_{ham}.pickle','rb') as f:
            H_sum = pickle.load(f)
    if num_spins is None:
        with open(path+f'num_spins_{ham}.pickle','rb') as f:
            num_spins = pickle.load(f)
    if num_neighbors is None:
        with open(path+f'num_neighbors_{ham}.pickle','rb') as f:
            num_neighbors = pickle.load(f)
    if coupling_constants is None:
        with open(path+f'coupling_constants_{ham}.pickle','rb') as f:
            coupling_constants = pickle.load(f)
    H_num_numba = numba.jit(H_num, nopython=False)
    coupling_constants_num = np.ones(len(coupling_constants))
    spin_model = {'num_spins':num_spins,
            'num_neighbors':num_neighbors,
            'coupling_constants':coupling_constants,
            'coupling_constants_num':coupling_constants_num,
            'H_single':H_single,
            'H_sum':H_sum,
            'M':M,
            'M_num':M_num,
            'H_num':H_num,
            'H_math':H_math,
            'rot':rot,
            'rot_num':rot_num,
            'driving_term':driving_term,
            'driving_term_num':driving_term_num,
            'driving_term_numba':driving_term_numba,
            'H_num_numba':H_num_numba}
    
    return spin_model

class SpinModel:
    '''
    spin model. Container for analytic and and numerical Hamiltonian coupling constants as a dictionary.

    methods:
        - compute_dispersion
        - compute_groundstates 
    '''

    def __init__(self, spin_model, name):
        '''
        instantiate spin model via a dictionary with entries:
        '''

        self.name = name
        self.spin_model = spin_model
        self.num_spins = spin_model['num_spins']
        self.num_neighbors = spin_model['num_neighbors']
        self.H_single = spin_model['H_single']
        self.H_sum = spin_model['H_sum']
        self.coupling_constants_analytic = spin_model['coupling_constants']
        self.coupling_constants = {}
        for ii, cc in enumerate(spin_model['coupling_constants']):
            self.coupling_constants[cc] = spin_model['coupling_constants_num'][ii]
        self.M = spin_model['M']
        self.M_num = spin_model['M_num']
        self.H_num = spin_model['H_num']
        self.H_math = spin_model['H_math']
        self.H_num_numba = spin_model['H_num_numba']
        self.rot = spin_model['rot']
        self.rot_num = spin_model['rot_num']
        self.driving_term = spin_model['driving_term']
        self.driving_term_num = spin_model['driving_term_num']
        self.driving_term_numba = spin_model['driving_term_numba']

        self.experiments = {}

    def compute_ground_states(self, experiment, fields, field_direction, x0, method=None, tqdm_disable=True):
    
        field_direction = field_direction/np.sqrt(np.sum(field_direction**2))
        groundstates = []
        energies = []
        for ii in tqdm(range(len(fields)), disable=tqdm_disable):
            hx, hy, hz = field_direction*fields[ii]
            self.coupling_constants[sp.symbols('h_x')] = hx
            self.coupling_constants[sp.symbols('h_y')] = hy
            self.coupling_constants[sp.symbols('h_z')] = hz
            coupling_constants = list(self.coupling_constants.values())
            state = find_ground_state_python(self.H_num, coupling_constants, self.num_spins, x0, method=method)
            energies.append(self.get_energy(state, np.array([hx, hy, hz])))
            x0 = state
            groundstates.append(state)

        self.experiments[experiment] = {'fields':fields, 'field_direction':field_direction, 'energies_imported':energies , 'groundstates':groundstates}
        self.compute_ground_state_properties(experiment)

    def compute_ground_states_llg(self, experiment, fields, field_direction, x0, alpha=0.01, dt=0.2, Ns=5000, tqdm_disable=True):
    
        field_direction = field_direction/np.sqrt(np.sum(field_direction**2))
        groundstates = []
        energies = []
        for ii in tqdm(range(len(fields)), disable=tqdm_disable):
            hx, hy, hz = field_direction*fields[ii]
            self.coupling_constants[sp.symbols('h_x')] = hx
            self.coupling_constants[sp.symbols('h_y')] = hy
            self.coupling_constants[sp.symbols('h_z')] = hz
            coupling_constants = list(self.coupling_constants.values())
            times, states = run_llg_sim_numba(self.driving_term_numba, coupling_constants, x0, alpha, dt, Ns)
            state = states[-1]
            energies.append(self.get_energy(state, np.array([hx, hy, hz])))
            groundstates.append(state)
            x0 = state

        self.experiments[experiment] = {'fields':fields, 'field_direction':field_direction, 'energies_imported':energies , 'groundstates':groundstates}
        self.compute_ground_state_properties(experiment)

    def load_groundstates_mathematica(self, experiment, directory):

        # in plane
        groundstates = []
        files = glob.glob(directory+'*.dat')
        fields = []
        states = []
        for f in files:
            if f==directory+'energies.dat':
                pass
            else:
                with open(f, 'r') as file:
                    field_str = file.readline()
                numbers = re.findall(match_dec, field_str)
                fields.append(np.array([float(n) for n in numbers]))
                state = np.loadtxt(f, skiprows=3)
                states.append(state)
        energies = np.loadtxt(directory+'energies.dat')
        fields = np.array(fields)
        fields_abs = np.sqrt(np.sum(fields**2, axis=1))
        states = np.array(states)
        sort_idx = np.argsort(fields_abs)
        fields = fields[sort_idx]
        states = states[sort_idx]
        fields_abs = fields_abs[sort_idx]
        field_dir = fields[-1]/fields_abs[-1]

        self.experiments[experiment] = {}
        self.experiments[experiment]['field_direction'] = field_dir
        self.experiments[experiment]['fields'] = fields_abs
        self.experiments[experiment]['groundstates'] = states
        self.experiments[experiment]['energies_imported'] = energies

        self.compute_ground_state_properties(experiment)

    def get_energy(self, state, field):
        hx, hy, hz = field
        self.coupling_constants[sp.symbols('h_x')] = hx
        self.coupling_constants[sp.symbols('h_y')] = hy
        self.coupling_constants[sp.symbols('h_z')] = hz
        coupling_constants_n = list(self.coupling_constants.values())
        return self.H_num(*coupling_constants_n, *state.flatten())/self.num_spins

    def compute_ground_state_properties(self,experiment):

        states = self.experiments[experiment]['groundstates']
        fields = self.experiments[experiment]['fields']
        field_dir = self.experiments[experiment]['field_direction']
        num_spins = self.num_spins

        # compute ground state properties:
        magnetization = np.array([np.sum(states[i], axis=0) for i in range(len(states))])
        mag_fielddir = np.array([np.dot(magnetization[i], field_dir) for i in range(len(states))]) 
        birefringence_amp_angle = np.array([get_birefringence(states[i]) for i in range(len(states))])
        birefringence = birefringence_amp_angle[:,0]
        angle = birefringence_amp_angle[:,1]
        energies = []
        q1amps = []
        q2amps = []
        for ii, hi in enumerate(fields):
            hxi, hyi, hzi = hi*field_dir
            self.coupling_constants[sp.symbols('h_x')] = hxi
            self.coupling_constants[sp.symbols('h_y')] = hyi
            self.coupling_constants[sp.symbols('h_z')] = hzi
            coupling_constants = list(self.coupling_constants.values())
            energies.append(self.H_num(*coupling_constants, *states[ii].flatten())/num_spins)
            q1_amp, q2_amp = get_tot_scattering_cross_section_xy([1/3,1], states[ii])
            q1amps.append(q1_amp)
            q2amps.append(q2_amp)

        self.experiments[experiment]['magnetization'] = magnetization
        self.experiments[experiment]['mag_fielddir'] = mag_fielddir
        self.experiments[experiment]['energies'] = energies
        self.experiments[experiment]['birefringence'] = birefringence
        self.experiments[experiment]['birefringence angle'] = angle
        self.experiments[experiment]['q1'] = q1amps
        self.experiments[experiment]['q2'] = q2amps
    
    def plot_groundstate_properties(self, experiment):

        hs = self.experiments[experiment]['fields']
        energies = self.experiments[experiment]['energies']
        energies_imported = self.experiments[experiment]['energies_imported']
        mag_fielddir = self.experiments[experiment]['mag_fielddir']
        birefringence = self.experiments[experiment]['birefringence']
        q1 = self.experiments[experiment]['q1']
        q2 = self.experiments[experiment]['q2']

        # plot ground state properties:
        fig , ax = plt.subplots(1,3, figsize=(12,4.5))
        ax[0].plot(hs, energies, 'o')
        ax[0].plot(hs, energies_imported, 'o', color='red')
        ax[1].plot(hs, mag_fielddir, 'o', color='red', label=r'M $\parallel$ field')
        ax[1].plot(hs, birefringence, 'o', color='blue', label='birefringence')
        ax[2].plot(hs, q1, 'o', color='red', label='q = 1/3')
        ax[2].plot(hs, q2, 'o', color='blue', label='q = 1')
        ax[0].set(xlabel=r'$H_x$', ylabel='Energy')
        ax[1].set(xlabel=r'$H_x$', ylabel=r'amplitude')
        ax[2].set(xlabel=r'$H_x$', ylabel='Scattering Amplitude')
        ax[2].legend()
        ax[1].legend()
        fig.suptitle(experiment)
        fig.tight_layout()
        plt.show()

    def plot_groundstate(self, experiment, field, elev=90, azim=90, tol=0.01):

        hs = self.experiments[experiment]['fields']
        groundstates = self.experiments[experiment]['groundstates']

        idx = np.argwhere(np.abs(hs-field)<=tol)[0][0]
        hi = hs[idx]

        fig, ax = plt.subplots(1,2, figsize=(10,5))
        qs = np.linspace(0,2,1000)
        sigmas = get_tot_scattering_cross_section_xy(qs, groundstates[idx])
        ax[1].plot(qs, sigmas)
        ax[1].set(xlabel=r'$q_z$', ylabel='$\sigma_{tot}$')
        fig.suptitle(f'Experiment:{experiment}; Field: {hi}')
        plot_state(groundstates[idx], elev=elev, azim=azim, fig=fig, ax=ax[0])
        fig.tight_layout()
        plt.show()

    def compute_dispersions(self, experiment, k_vect=[0], tqdm_disable=True):

        hs = self.experiments[experiment]['fields']
        field_dir = self.experiments[experiment]['field_direction']
        groundstates = self.experiments[experiment]['groundstates']
        num_spins = self.num_spins

        dispersions = 1j*np.zeros((2*num_spins, len(hs), len(k_vect)))
        eigenvectors = 1j*np.zeros((2*num_spins, len(hs), len(k_vect), 2*num_spins))
        mzs = np.zeros((2*num_spins, len(hs), len(k_vect)))
        neels = np.zeros((2*num_spins, len(hs), len(k_vect)))
        diags = np.zeros((2*num_spins, len(hs), len(k_vect)))
        off_diags = np.zeros((2*num_spins, len(hs), len(k_vect)))
        for ii in tqdm(range(len(hs)), disable=tqdm_disable):
            hxi, hyi, hzi = hs[ii]*field_dir
            self.coupling_constants[sp.symbols('h_x')] = hxi
            self.coupling_constants[sp.symbols('h_y')] = hyi
            self.coupling_constants[sp.symbols('h_z')] = hzi
            coupling_constants = list(self.coupling_constants.values())
            dispersion, eigenvects, g0 = compute_lswt(self.H_single, self.H_math, self.coupling_constants_analytic, coupling_constants, k_vect, num_spins, self.num_neighbors, self.M_num, self.M, groundstate=groundstates[ii])
            dispersions[:,ii,:] = dispersion
            eigenvectors[:,ii,:,:] = eigenvects
            # compute mode characters
            for jj in range(2*num_spins):
                for kk in range(len(k_vect)):
                    mode = eigenvectors[jj, ii, kk]
                    mode_cart = get_mode_cartesian(mode, self.rot_num(*groundstates[ii].flatten()))
                    mzs[jj, ii, 0] = get_magnetization_mode(mode_cart)[2]
                    neels[jj, ii, 0] = get_neel_mode(mode_cart)[2]
                    diag, off_diag = get_birefringence_mode(mode_cart, groundstates[ii])
                    diag2, off_diag2 = get_birefringence_mode_secondorder(mode_cart)
                    diags[jj, ii, kk] = diag
                    off_diags[jj, ii, kk] = off_diag
            
        self.experiments[experiment]['k_vect'] = k_vect
        self.experiments[experiment]['dispersions'] = dispersions
        self.experiments[experiment]['eigenvectors'] = eigenvectors
        self.experiments[experiment]['mode_mzs'] = mzs
        self.experiments[experiment]['mode_neels'] = neels
        self.experiments[experiment]['mode_diags'] = diags
        self.experiments[experiment]['mode_off_diags'] = off_diags

    def plot_dispersion(self, experiment, k_indx=0, ms=10, xlim=(0,None), ylim=(0,None)):
        
        hs = self.experiments[experiment]['fields']
        dispersions = self.experiments[experiment]['dispersions']
        num_spins = self.num_spins

        fig, ax = plt.subplots(1, figsize=(5,4))
        ax.set(xlabel='Field', ylabel='Frequency')
        kfreqs = dispersions[:,:,k_indx]
        for ii, mode in enumerate(kfreqs[num_spins:]):
            #ax3.scatter(hs_inplane**3, np.real(mode), s=ms, c=modecolors[num_spins+ii,:], vmin=0, vmax=15, cmap=cmap, label=ii)
            ax.scatter(hs, np.real(mode), s=ms)
        ax.set(xlim=xlim, ylim=ylim)
        fig.tight_layout()

        # view field dependence of dispersion:
        fig2, ax2 = plt.subplots(1, figsize=(5,4))
        ax2.set(xlabel='Field', ylabel='Imaginary Frequency')
        kfreqs = dispersions[:,:,k_indx]
        for ii, mode in enumerate(kfreqs[num_spins:]):
            #ax3.scatter(hs_inplane**3, np.real(mode), s=ms, c=modecolors[num_spins+ii,:], vmin=0, vmax=15, cmap=cmap, label=ii)
            ax2.scatter(hs, np.imag(mode), s=ms)
        ax2.set(xlim=xlim, ylim=ylim)
        fig2.tight_layout()

        plt.show()

    def plot_mode_characters(self, experiment, modes='all', k_indx=0, ms=10, xlim=(0,None), ylim=(0,None)):

        hs = self.experiments[experiment]['fields']
        num_spins = self.num_spins
        mzs = self.experiments[experiment]['mode_mzs']
        neels = self.experiments[experiment]['mode_neels']
        diags = self.experiments[experiment]['mode_diags']
        off_diags = self.experiments[experiment]['mode_off_diags']

        if modes=='all':
            modes = range(num_spins, 2*num_spins)

        # plot mode characters
        fig, ax = plt.subplots(2,2, figsize=(10,8))
        ax[0,0].set(xlabel='Field', ylabel=r'M$_z$')
        ax[0,1].set(xlabel='Field', ylabel=r'L$_z$')
        ax[1,0].set(xlabel='Field', ylabel=r'$\delta r$')
        ax[1,1].set(xlabel='Field', ylabel=r'$\Delta$')
        for ii in modes:
                #ax[0,0].scatter(hs, mzs, s=ms, cmap=cmap, c=modecolors[num_spins+ii,:])
                #ax[0,1].scatter(hs, neels, s=ms, cmap=cmap, c=modecolors[num_spins+ii,:])
                #ax[1,0].scatter(hs, drs, s=ms, cmap=cmap, c=modecolors[num_spins+ii,:])
                #ax[1,1].scatter(hs, deltas, s=ms, cmap=cmap, c=modecolors[num_spins+ii,:])
                ax[0,0].scatter(hs, mzs[ii,:,k_indx], s=ms)
                ax[0,1].scatter(hs, neels[ii,:,k_indx], s=ms)
                ax[1,0].scatter(hs, diags[ii,:,k_indx], s=ms)
                ax[1,1].scatter(hs, off_diags[ii,:,k_indx], s=ms)
        ax[0,0].set(xlim=xlim, ylim=ylim)
        ax[0,1].set(xlim=xlim, ylim=ylim)
        ax[1,0].set(xlim=xlim, ylim=ylim)
        ax[1,1].set(xlim=xlim, ylim=ylim)
        fig.tight_layout()
        
        plt.show()

    def plot_mode(self, experiment, mode, field, k_idx=0, elev=20, azim=-45):

        hs = self.experiments[experiment]['fields']
        groundstates = self.experiments[experiment]['groundstates']
        dispersions = self.experiments[experiment]['dispersions']
        eigenvectors = self.experiments[experiment]['eigenvectors']
        k_vect = self.experiments[experiment]['k_vect']

        k_idx = 0
        dh = hs[1]
        field_idx = np.argwhere(np.abs(hs-field)<=dh)[0][0]

        k_num = k_vect[k_idx]
        freqnum = dispersions[mode, field_idx, k_idx]
        mode_vect = get_mode_cartesian(eigenvectors[mode, field_idx, k_idx], self.rot_num(*groundstates[field_idx].flatten()))
        equilibrium_vect = groundstates[field_idx]

        fig, ax = plt.subplots(1)
        plot_mode(k_num, freqnum, mode_vect, equilibrium_vect, period_fraction=0.5, elev=elev, azim=azim, fig=fig, ax=ax)
        fig.suptitle(f'Field: {hs[field_idx]}   Frequency: {round(np.real(freqnum),5)}')
        fig.tight_layout()
        plot_mode(k_num, freqnum, mode_vect, equilibrium_vect, period_fraction=0.5, elev=90, azim=azim, colorbar=False)
        plt.show()

    def export_experiments(self, directory):
        '''
        exports experiments to a directory in following structure

        directory/
            name/
                model_info.dat
                experiment1/
                    ...
                    ...
                    ...
                experiment2/
                    ...
                    ...
                    ...

        where under each experiment directory there is a file for each entry the experiment dictionary, labeled by the dictionary key
        '''

        experiments = list(self.experiments.keys())
        name = self.name
        root_path = f'{directory}/{name}'
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        with open(root_path+'/model_info.dat', 'w') as f:
            f.write(f'name: {name}\n')
            f.write(f'num_spins: {self.num_spins}\n')
            f.write(f'num_neighbors: {self.num_neighbors}\n')
            f.write('[Coupling Constants]\n')
            for const in self.coupling_constants_analytic:
                f.write(f'{const}: {self.coupling_constants[const]}\n')

        for ii in experiments:
            experiment_path = f'{directory}/{name}/{ii}'
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)
            exp = self.experiments[ii]
            keys = list(exp.keys())
            for key in keys:
                val = np.array(exp[key])
                with open(experiment_path+f'/{key}.dat', 'w') as f:
                    f.write(f'[Shape]\n{val.shape}\n')
                    f.write(f'[Data]\n')
                    for v in val.flatten():
                        f.write(f'{v}\n')

    def import_experiments(self, directory):

        with open(f'{directory}/model_info.dat', 'r') as f:
            self.name = f.readline().rsplit('name: ', 1)[-1][:-1]
            self.num_spins = int(f.readline().rsplit('num_spins: ', 1)[-1])
            self.num_neighbors = int(f.readline().rsplit('num_neighbors: ', 1)[-1])
            f.readline()
            for const in list(self.coupling_constants.keys()):
                line = f.readline()
                val = float(line.rsplit(f'{const}: ', 1)[-1])
                self.coupling_constants[const] = val

        experiments = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        self.experiments = {}
        for exp in experiments:
            exp_dir = f'{directory}/{exp}'
            self.experiments[exp] = {}
            files = glob.glob(f'{exp_dir}/*')
            for file in files:
                key = file[len(f'{exp_dir}/'):-len('.dat')]
                with open(file, 'r') as f:
                    f.readline()
                    shape_string = f.readline()
                    shape = tuple(map(int, re.findall(r'\d+\.\d+|\d+', shape_string)))
                try:
                    data = np.loadtxt(file, skiprows=3)
                except:
                    data = np.loadtxt(file, skiprows=3, dtype=np.complex_)
                data = data.reshape(shape)
                self.experiments[exp][key] = data