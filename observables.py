'''
Functions in this file allow you to compute observables of spin states, such as net magnetization, birefringence, etc.

all states are assumed to be represented as state = [[Sx1, Sy1, Sz1],...,[Sxn, Syn, Szn]]

'''
import numpy as np

def get_magnetization(state):
    return np.sum(state, axis=0)

def get_neel(state):
    num_spins = int(len(state))
    idx1 = np.arange(0,num_spins,2)
    idx2 = np.arange(1,num_spins+1,2)
    m1 = np.sum(state[idx1], axis=0)
    m2 = np.sum(state[idx2], axis=0)
    neel = m1-m2
    return neel

def get_birefringence(state):
    '''
    evaluates the birefringence of a state according to the total matrix

    R = sum_spins([[Sx^2 - Sy^2, 2*Sx*Sy], [2*Sx*Sy, -(Sx^2 - Sy^2)]])

    by finding the eigenvalues and eigenvectors.
    '''

    bir = np.sum([np.array([[state[i][0]**2 - state[i][1]**2, 2*state[i][0]*state[i][1]], [2*state[i][0]*state[i][1], -(state[i][0]**2 - state[i][1]**2)]]) for i in range(len(state))], axis=0)

    '''
    vals, vects = lin.eig(bir)
    val = np.abs(vals[0])
    ang = np.arctan2(vects[0], vects[1])*(180/np.pi)
    '''

    val = np.sqrt(bir[0,0]**2 + bir[0,1]**2)/len(state)
    ang = (1/2)*np.arctan2(bir[0,1],bir[0,0])

    return val, ang

def get_magnetization_mode(mode):
    return np.abs(np.sum(mode, axis=0))

def get_neel_mode(mode):
    num_spins = int(len(mode))
    neel = np.abs(np.sum([(-1)**i*mode[i] for i in range(num_spins)], axis=0))
    return neel

def get_birefringence_mode(mode, groundstate):
    '''
    evaluates the first order birefringence amplitude of a mode. See Alex notes for details and for second order contribution.
    '''
    num_spins = int(len(mode))
    diagonal = 2*np.abs(np.sum([groundstate[i,0]*mode[i,0] - groundstate[i,1]*mode[i,1] for i in range(num_spins)]))
    off_diagonal = 2*np.abs(np.sum([groundstate[i,0]*mode[i,1] + groundstate[i,1]*mode[i,0] for i in range(num_spins)]))

    return diagonal, off_diagonal

def get_birefringence_mode_secondorder(mode):
    '''
    evaluates the second order order birefringence amplitude of a mode. See Alex notes for details and for second order contribution.
    '''
    num_spins = int(len(mode))
    diagonal = (1/2)*np.abs(np.sum([mode[i,0]**2 - mode[i,1]**2 for i in range(num_spins)]))
    off_diagonal = np.abs(np.sum([mode[i,1]*mode[i,0] for i in range(num_spins)]))

    return diagonal, off_diagonal

def fft_magnetic_structure(state, uc_length=2, numcells=10):
    '''
    return the fourier components of the state using

    M(q) = 1/\sqrt(N)\sum_n(-iqRn)M(n)
    '''

    numspins = len(state)
    mx = np.tile(state[:,0], numcells)
    my = np.tile(state[:,1], numcells)
    mz = np.tile(state[:,2], numcells)

    mqx = fft.fft(mx)
    mqy = fft.fft(my)
    mqz = fft.fft(mz)
    qvect = fft.fftfreq(numcells*numspins,1/uc_length)

    sortidx = np.argsort(qvect)
    qvect = np.concatenate([qvect[sortidx], [-qvect[sortidx][0]]])
    mqx = np.concatenate([mqx[sortidx], [mqx[sortidx][0]]])
    mqy = np.concatenate([mqy[sortidx], [mqy[sortidx][0]]])
    mqz = np.concatenate([mqz[sortidx], [mqz[sortidx][0]]])

    return qvect, mqx, mqy, mqz

def get_Sq(q, state, uc_length=2, numcells=10):
    '''
    explicilty compute S(q) = 1/\sqrt(N)\sum_n(-iQRn)S(n) for a given q in unit unit cell units, ie Q=(2pi/a)*q

    '''
    numspins = len(state)
    state_expanded = np.tile(state, (numcells,1))
    Q = q*(2*np.pi/uc_length)
    return (1/np.sqrt(numspins*numcells))*np.sum(np.array([np.exp(-1j*i*Q)*state_expanded[i] for i in range(numspins*numcells)]), axis=0)

def get_scattering_cross_section(qs, state, i=0, j=0, uc_length=2, numcells=10):
    '''
    return the scattering cross section sigma_ij = S(q)^\dagger_i * S(q)_j for a spin state over the specified range of qs.
    '''

    cross_section = np.zeros(len(qs))
    for ii, q in enumerate(qs):
        cross_section[ii] = np.conjugate(get_Sq(q, state, uc_length, numcells)[i])*get_Sq(q, state, uc_length, numcells)[j]
    return cross_section

def is_mode_uniform(mode, round_sig=5):
    '''
    returns True if the all vectors in the mode are the same
    '''

    unique = len(np.unique(np.round(mode,round_sig),axis=0))
    if unique==1:
        return True
    else:
        return False

def get_tot_scattering_cross_section_xy(qs, state, uc_length=2, numcells=10):
    '''
    return sigma_xx + sigma_yy
    '''
    return get_scattering_cross_section(qs, state, 0, 0, uc_length, numcells) + get_scattering_cross_section(qs, state, 1, 1, uc_length, numcells)