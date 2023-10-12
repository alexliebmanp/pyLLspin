'''
Functions in this file allow you to compute observables of spin states, such as net magnetization, birefringence, etc.

all states are assumed to be represented as state = [[Sx1, Sy1, Sz1],...,[Sxn, Syn, Szn]]

'''
import numpy as np

def get_magnetization(state):
    return np.sum(state, axis=0)

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