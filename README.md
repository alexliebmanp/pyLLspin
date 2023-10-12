# pyLLspin

pyLLspin is a package to analytically handle linear spin-wave theory from the Landau-Lifshitz equation in 1 dimension.

The code works by first working through the formalism of linear-spin wave theory for an arbitrary Hamiltonian to obtain an effective magnetic field, plugging in a harmonic ansatz to the LL equaiton, and obtainin a matrix M to diagonlize, where the eigenvalues represent the spin-wave ferquencies as a function of wavevect, and the eigenstates the spin-wave modes.

From here, everything else, starting from diagonlization of the M matrix, can be handled numerically. In addition, tools are included for finding ground states (which are crucial for proper evaluation of M), and visualization of results.

An example calculation based on the J1-J2 model is found in exampleJ1J2.py, which we breifly describe below:


We first define a Hamiltonian ```H_single``` as a sympy expression which represents the energy of the nth spin (ie, including coupling to all neighbors). For obtaining ground state codes, it is also necessary to define ```H_sum ```, which is the same Hamlitonian but such that it represents the nth term in $H = \sum_n H_n$. With these two analytical expressions we can then easily obtain 
```
# parameters and coupling constants
num_spins = 6
num_neighbors = 2 # specify number of neighbors for interaction
J1, J2, K, hx, hy, hz, a = sp.symbols('J_1, J_2, K, h_x, h_y, h_z, a_0')
h = sp.Matrix([hx, hy, hz]) # define coupling constant and material params
coupling_constants = [J1, J2, K, hx, hy, hz, a]

# energy of the nth spin
H_single = -2*J1*S(n).dot(S(n+1) + S(n-1)) + -2*J2*S(n).dot(S(n+2) + S(n-2)) - 2*S(n).dot(h) + K*Sz(n)**2
H_sum = -2*J1*S(n).dot(S(n+1)) + -2*J2*S(n).dot(S(n+2)) - 2*S(n).dot(h) + K*Sz(n)**2

# obtain analytical and numerical expressions for the LL matrix
M = get_analytical_ll_matrix_transv(H_single, num_spins, num_neighbors)
M_num = get_numerical_ll_matrix(M, coupling_constants, num_spins)
H_num = get_numerical_H(H_sum, coupling_constants, num_spins, num_neighbors)
H_math = get_mathematica_H(H_sum, num_spins, num_neighbors)
```
