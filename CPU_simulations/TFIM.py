import numpy as np
import matplotlib.pyplot as plt
import time as time
import scipy.sparse as sparse
import scipy.sparse.linalg as sLA
import itertools
import string

# build sparse Pauli matrices
Sp = sparse.csr_matrix((np.array([[0, 1], [0, 0]])))
Sm = np.transpose(Sp)
Sx = Sp + Sm
Sy = -1j*(Sp - Sm)
Sz = sparse.csr_matrix(np.array([[1, 0], [0, -1]]))
S0 = sparse.eye(2)
Sa = np.array([Sx, Sy, Sz])

def theoretical_groundstate_energy(N, B, J=1):
    """ Return the exact theoretical groundstate energy for given params"""
    J = B/J
    ms = np.arange(-0.5*(N - 1), 0.5*(N - 1) + 1, 1)
    ks = 2*np.pi*ms/N
    lambs = np.sqrt(1 + J**2 + 2*J*np.cos(ks))
    E_0 = -np.sum(lambs)
    return E_0

# Calculate operator o on jth spin
def operator_on_jth(o, j, N):
    if j == 0:
        o_counter = o
    else:
        o_counter = sparse.eye(2)
    for i in range(1, N):
        if i == j:
            o_counter = sparse.kron(o_counter, o)
        else:
            o_counter = sparse.kron(o_counter, sparse.eye(2))
    return o_counter

# build the single site operator strings for x, y, z gate
def buildXYZSingleSiteStrings(N):
    Sa_lists = [[], [], []]
    for op_i in range(3):
        for i in range(N):
            j_op = operator_on_jth(Sa[op_i], i, N)
            Sa_lists[op_i].append(j_op)
    return Sa_lists

# build the Hamiltonian with zz interaction and field in x direction
def buildHamiltonian(N, B, J, Sa_lists=None):
    if Sa_lists is None:
        Sa_lists = buildXYZSingleSiteStrings(N)
    H = sparse.csr_matrix(np.zeros(shape=[2**N, 2**N]))
    for i in range(N):
        H += -J*(Sa_lists[2][i].dot(Sa_lists[2][(i + 1)%N]))
        H += - B*Sa_lists[0][i]
    return H

# diagonalize the TFIM Hamiltonian and return the lowest eigenstates/values for a given set of parameters
def exact_diag(N, B, J, lower=1):
    xyz_strings = buildXYZSingleSiteStrings(N)
    H = buildHamiltonian(N, B, J, Sa_lists=xyz_strings)
    eigvals, psis = sLA.eigsh(H.toarray(), k=2**N-1, which="SA")
    sort_indices = np.argsort(eigvals)
    eigvals = eigvals[sort_indices]
    psis = psis[:, sort_indices].T
    return eigvals[:lower], psis[:lower, :]

def exact_diag_H(H, lower=1):
    N = int(np.log2(H.shape[0]))
    eigvals, psis = sLA.eigsh(H.toarray(), k=2**N-1, which="SA")
    sort_indices = np.argsort(eigvals)
    eigvals = eigvals[sort_indices]
    psis = psis[:, sort_indices].T
    return eigvals[:lower], psis[:lower, :]



