import numpy as np
import matplotlib.pyplot as plt
import time as time
import scipy.sparse as sparse
import scipy.sparse.linalg as sLA
import itertools
import string
import numba

# build sparse Pauli matrices
Sp = sparse.csr_matrix((np.array([[0, 1], [0, 0]])))
Sm = np.transpose(Sp)
Sx = Sp + Sm
Sy = -1j*(Sp - Sm)
Sz = sparse.csr_matrix(np.array([[1, 0], [0, -1]]))
S0 = sparse.eye(2)
Sa = np.array([Sx, Sy, Sz])

def calc_probs(state, povm):
    """ calculate the POVM distribution of a given state """
    if povm is None:
        raise ValueError("Specify a POVM to be used.")
    else:
        if callable(povm):
            n = int(np.log2(state.shape[0])) # number of spins
            M, T, outcomes = povm(n)
            Tinv = np.linalg.inv(T)
        elif len(povm) == 2:
            M, Tinv, outcomes = povm
        
    p = np.trace(M @ state, axis1=1, axis2=2).astype(np.float)
    return p, M, Tinv, outcomes

@numba.jit(nopython=True)
def to_a_tuple(a_joint, n):
    """ turn an integer in range [0, 4^n - 1] into an n-dimensional coordinate tuple """
    vols = [4**i for i in range(n)]
    coords = [0]*n
    temp_a = int(a_joint)
    for i in range(n - 1, -1, -1):
        coords[i] = temp_a // vols[i]
        temp_a -= int(coords[i]*vols[i])
    return coords

@numba.jit(nopython=True)
def to_a_joint(a_tuple, n):
    """ turn a n-dim coordinate tuple into an integer in range [0, 4^n - 1] """
    vols = 4**np.arange(n)
    return np.dot(np.array(a_tuple), vols)

@numba.jit(nopython=True)
def tetra_povm(n):
    """ calculate the tetra POVM elements and the overlap matrix for a given number of spins """
    # define elementary POVM elements
    m0 = np.array([[0.5 + 0*1j, 0 + 0*1j], [0 + 0*1j, 0 + 0*1j]])
    m1 = 1/6*np.array([[1 + 0*1j, np.sqrt(2) + 0*1j], [np.sqrt(2) + 0*1j, 2 + 0*1j]])
    m2 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) - np.sqrt(6)*1j], [-np.sqrt(2) + np.sqrt(6)*1j, 4 + 0*1j]])
    m3 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) + np.sqrt(6)*1j], [-np.sqrt(2) - np.sqrt(6)*1j, 4 + 0*1j]])
    m_single = [m0, m1, m2, m3]
    
    M = np.zeros((4**n, 2**n, 2**n), dtype=np.complex64) # POVM elements
    T = np.zeros((4**n, 4**n), dtype=np.complex64) # overlap matrix
    outcomes = [] # tuple representation of elements
    for a_joint in range(4**n):
        a_i = to_a_tuple(a_joint, n)
        outcomes.append(a_i)
        Ma = m_single[a_i[0]]
        for j in a_i[1:]: # multiply together all elements of the tuple
            Ma = np.kron(m_single[j], Ma)
        M[a_joint, :, :] = Ma.copy()
        T[a_joint, a_joint] = np.trace(np.dot(Ma, Ma)) # calculate diag entry of T
        for k in range(a_joint): # calculate off-diag entries of T 
            T[a_joint, k] = np.trace(np.dot(Ma.astype(np.complex64), M[k]))
            T[k, a_joint] = T[a_joint, k]#np.trace(np.dot(M[k], Ma))
    return M, T, outcomes

def angle_tetra_povm(n, theta=0, phi=0):
    """ calculate the tetra POVM elements and the overlap matrix for a given number of spins and given rotation angles """
    # define elementary POVM elements
    s0 = np.array([[1, 0], [0, 1]])
    s1 = [[0, 1], [1, 0]]
    s2 = [[0, -1j], [1j, 0]]
    s3 = [[1, 0], [0, -1]]
    S = np.array([s1, s2, s3])
    r0 = [3*np.sin(theta)*np.sin(phi), -3*np.sin(theta)*np.cos(phi), 3*np.cos(theta)]
    r1 = [2*np.sqrt(2)*np.cos(phi) - np.sin(phi)*np.sin(theta), 2*np.sqrt(2)*np.sin(phi) + np.sin(theta)*np.cos(phi), -np.cos(theta)]
    r2 = [-np.sqrt(2)*np.cos(phi) - np.sqrt(6)*np.sin(phi)*np.cos(theta) - np.sin(phi)*np.sin(theta), -np.sqrt(2)*np.sin(phi) + np.sqrt(6)*np.cos(phi)*np.cos(theta) + np.cos(phi)*np.sin(theta), np.sqrt(6)*np.sin(theta) - np.cos(theta)]
    r3 = [-np.sqrt(2)*np.cos(phi) + np.sqrt(6)*np.sin(phi)*np.cos(theta) - np.sin(phi)*np.sin(theta), -np.sqrt(2)*np.sin(phi) - np.sqrt(6)*np.cos(phi)*np.cos(theta) + np.cos(phi)*np.sin(theta), -np.sqrt(6)*np.sin(theta) - np.cos(theta)]
    rs = 1/3*np.array([r0, r1, r2, r3])
    m_single = 0.25*(s0[None,:,:] + np.einsum('ab,bcd->acd', rs,S))

    M = np.zeros((4**n, 2**n, 2**n), dtype=np.complex64) # POVM elements
    T = np.zeros((4**n, 4**n), dtype=np.complex64) # overlap matrix
    outcomes = [] # tuple representation of elements
    for a_joint in range(4**n):
        a_i = to_a_tuple(a_joint, n)
        outcomes.append(a_i)
        Ma = m_single[a_i[0]]
        for j in a_i[1:]: # multiply together all elements of the tuple
            Ma = np.kron(m_single[j], Ma)
        M[a_joint, :, :] = Ma.copy()
        #print(Ma.shape)
        T[a_joint, a_joint] = np.trace(Ma @ Ma) # calculate diag entry of T
        for k in range(a_joint): # calculate off-diag entries of T 
            T[a_joint, k] = np.trace(np.dot(Ma.astype(np.complex64), M[k]))
            T[k, a_joint] = T[a_joint, k]#np.trace(np.dot(M[k], Ma))
    return M, T, outcomes

def povm_marginals(as_marg, p_joint, num_outcomes=4):
    """ calculates the marginal POVM distribution given the spins to be marginalized over """
    n_spin = int(np.log2(p_joint.shape[0])//2) # number of spins TODO depends on num_outcomes
    p_joint_ndim = p_joint.reshape(*([num_outcomes]*n_spin)) # reshape joint distribution
    return p_joint_ndim.sum(axis=tuple(as_marg)) # marginalize out given spins

def povm_pair_marginals(p_joint, num_outcomes=4):
    """ calculates povm pair marginals for all spin pairs """
    n_spin = int(np.log2(p_joint.shape[0])//2) # the //2 depends on num_outcomes = 4
    spin_tuples = []
    p_margs = np.zeros(((n_spin**2 - n_spin)//2, 4, 4))
    tuple_count = 0
    for i in range(n_spin):
        for j in range(i):
            p_margs[tuple_count, :, :] = povm_marginals(np.setdiff1d(range(n_spin), [i, j]), p_joint)
            spin_tuples.append((i, j))
            tuple_count += 1
    return spin_tuples, p_margs

def povm_conditionals(as_cond, probs, margs=None):
    """ calculates the conditional probability for all outcomes given a subset of spins """
    n_spin = int(np.log2(probs.shape[0])//2) # number of spins TODO depends on num_outcomes
    marg_inds = np.setdiff1d(range(n_spin), as_cond)
    if margs is None:
        margs = povm_marginals(marg_inds, probs)
    norms = np.zeros(probs.shape[0])
    for a in range(probs.shape[0]):
        ind = tuple(np.array(to_a_tuple(a, n_spin), dtype=int)[as_cond].tolist())
        norms[a] = margs[ind]
    return probs/norms

def calc_op_coeffs(op, povm):
    """ calculate the operator coefficients for a specified POVM """
    if sparse.isspmatrix(op):
        op = op.toarray()
    if povm is None:
        raise ValueError("Specify a POVM to be used.")
    else:
        if callable(povm):
            n = int(np.log2(state.shape[0])) # number of spins
            M, T, _ = povm(n)
            Tinv = np.linalg.inv(T)
        elif len(povm) == 2:
            M, Tinv = povm
    tr_op_M = np.trace(np.dot(M, op), axis1=1, axis2=2).astype(np.complex)
    Qop = Tinv @ tr_op_M
    return Qop

def build_basis_strings(N):
    """ returns the string set of all possible basis states for N qubits """
    all_states = []
    for i in range(2**N):
        bstring = format(i, "b")
        bstring = "0"*(N - len(bstring)) + bstring
        all_states.append(bstring)
    return np.array(all_states)

def build_op(op_string):
    """ builds and returns the matrix representation of the given operator string, e.g. "1xy" = 1 x simga_x x sigma_y """
    ind = 0
    operator = S0
    for c in op_string[::-1]:
        if ind == 0:
            if c == "x":
                operator = Sx
            elif c == "y":
                operator = Sy
            elif c == "z":
                operator = Sz
            else:
                operator = S0
        else:
            if c == "x":
                operator = sparse.kron(operator, Sx)
            elif c == "y":
                operator = sparse.kron(operator, Sy)
            elif c == "z":
                operator = sparse.kron(operator, Sz)
            else:
                operator = sparse.kron(operator, S0)
        ind += 1
    return operator

def build_op_strings(N=2, up_to_order=2):
    """ build all spin operator strings up to a specified order """
    up_to_order = min(up_to_order, N)
    chars = ["x", "y", "z", "1"]
    for item in itertools.product(chars, repeat=N):
        perm = "".join(item)
        num_ones = perm.count("1")
        if num_ones < N - up_to_order or num_ones == N:
            continue
        yield perm

def build_op_strings_corr_d(N, op="x", diff=1, regex_filter=None, periodic=True):
    for i in range(N):
        pos_1 = i
        if periodic:
            pos_2 = (i + diff) % N
        else:
            pos_2 = i + diff
        string = "".join(["1" if (j != pos_1 and j != pos_2) else op for j in range(N)])
        if not regex_filter is None and len(re.findall(regex_filter, perm)) != 0:
            print(f"regex filter {regex_filter} on {string} yields: {re.findall(regex_filter, string)}" )
            continue
        yield string

def build_op_strings_two_point(N=2, first="x", second="x", regex_filter=None):
    """ build all spin operator strings up to a specified order """
    for x in range(N):
        for y in range(N):
            if x!=y:
                string = "1"*N
                string = string[:x] + first + string[x+1:]
                string = string[:y] + second + string[y+1:]
                if not regex_filter is None and len(re.findall(regex_filter, string)) != 0:
                    print(f"regex filter {regex_filter} on {string} yields: {re.findall(regex_filter, string)}" )
                    continue
                yield string

def theoretical_groundstate_energy(N, B, J=1): 
    """ Return the exact theoretical groundstate energy for given parameters
    Input:
    N       int     Number of spins
    B       float   Transvere field strength
    J       int     Interaction strength
    """
    ms = np.arange(-0.5*(N - 1), 0.5*(N - 1) + 1, 1)
    ks = 2*np.pi*ms/N
    lambs = np.sqrt(1 + (J/B)**2 + 2*(J/B)*np.cos(ks))
    E_0 = -np.sum(lambs)
    return E_0

def operator_on_jth(o, j, N):
    """ Calulcate N-spin operator for single site operation on spin j
    Input:
    o           sparse ndarray  Single site operator
    j           int             index of where to apply it
    N           int             Number of spins
    """

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

def buildXYZSingleSiteStrings(N):
    """ Construct single site local Pauli operators
    Input:
    N           int     Number of spins
    """
    Sa_lists = [[], [], []]
    for op_i in range(3):
        for i in range(N):
            j_op = operator_on_jth(Sa[op_i], i, N)
            Sa_lists[op_i].append(j_op)
    return Sa_lists

def buildHamiltonian(N, B, J, Sa_lists=None):
    """ Constructs TFIM Hamiltonian from given parameters
    Input:
    N           int         Number of spins
    B           float       Transverse field strength
    J           float       Interaction strength
    Sa_lists    list        List of lists of local Pauli operators in x,y,z direction
                            If equals 'None', the lists will be constructed
    """

    if Sa_lists is None:
        Sa_lists = buildXYZSingleSiteStrings(N)
    H = sparse.csr_matrix(np.zeros(shape=[2**N, 2**N]))
    for i in range(N):
        H += -J*(Sa_lists[2][i].dot(Sa_lists[2][(i + 1)%N]))
        H += - B*Sa_lists[0][i]
    return H

def exact_diag_H(H, lower=1):
    """ Returns the ground state energy and state of the given
    Input:
    H       sparse ndarray  Hamiltonian
    lower   int             up to which low-lying state corresponding energies
                            and coefficients are to be returned
    """

    N = int(np.log2(H.shape[0]))
    eigvals, psis = sLA.eigsh(H.toarray(), k=2**N-1, which="SA")
    sort_indices = np.argsort(eigvals)
    eigvals = eigvals[sort_indices]
    psis = psis[:, sort_indices].T
    return eigvals[:lower], psis[:lower, :]


def exact_diag(N, B, J, lower=1):
    """ Constructs TFIM Hamiltonian from given parameters and returns ground state energy and state
    Input:
    N       int         Number of spins
    B       float       Transverse field strength
    J       float       Interaction strength
    lower   int         up to which low-lying state corresponding energies
                        and coefficients are to be returned
    """

    xyz_strings = buildXYZSingleSiteStrings(N)
    H = buildHamiltonian(N, B, J, xyz_strings)
    eigvals, psis = sLA.eigsh(H.toarray(), k=2**N-1, which="SA")
    sort_indices = np.argsort(eigvals)
    eigvals = eigvals[sort_indices]
    psis = psis[:, sort_indices].T
    return eigvals[:lower], psis[:lower, :]

def up_state(N):
    """ Returns the product state with all spins pointing "up"
    Input: N    int     number of spins
    Output: psi ndarray wave function
            rho ndarray density matrix
    """
    psi = np.array(2**N*[0.0]); psi[0] = 1
    rho = np.outer(psi, psi)
    return psi, rho

def ghz_state(N):
    """ Returns the GHZ state
    Input: N    int     number of spins
    Output: psi ndarray wave function
            rho ndarray density matrix
    """
    psi = np.array(2**N*[0.0]); psi[0] = 1/np.sqrt(2); psi[-1] = 1/np.sqrt(2)
    rho = np.outer(psi, psi)
    return psi, rho

def max_mixed_state(N):
    """ Returns the maximally mixed state
    Input: N    int     number of spins
    Output: psi ndarray wave function
            rho ndarray density matrix
    """
    rho = np.eye(2**N); rho /= 2**N
    return None, rho

def tfim_groundstate_pbc(N, B):
    """ Returns the TFIM ground state for periodic boundary conditions
    Note: don't use for systems N>15 as this code is not optimized
    Input:
    N   int     number of spins
    B   float   transverse field strength (interaction J assumed to be =1)
    Output: psi ndarray wave function
            rho ndarray density matrix
    """
    # build TFIM Hamiltonian of given params
    modelstring = "TFIM_N{}_B{}".format(N, B)
    if Path(f"hamiltonians/{modelstring}.npz").is_file():
        H_sparse = load_npz(f"hamiltonians/{modelstring}.npz")
    else:
        H_sparse = buildHamiltonian(N, B, 1)
        save_npz(f"hamiltonians/{modelstring}.npz", H_sparse)

    # calculate solution with ED 
    e0, psi0 = exact_diag_H(H_sparse)
    e0 = e0[0]
    psi0 = psi0[0, :]
    rho0 = np.outer(psi0, psi0)

    return psi0, rho0

def calc_fidelity(probs, target_state, povm=None):
""" calculates fidelities of a given list of probabilities """
    if povm is None:
        #qfids = np.sqrt(np.abs(np.sqrt(probs) @ target_state))
        qfids = np.abs(np.sqrt(probs) @ target_state)
    else:
        povm_distr, M, Tinv, outcomes = povm
        Qrho_rbm = probs @ Tinv
        rho_rbm = np.real(np.tensordot(Qrho_rbm, M, axes=1))
        sqrt_rho = sqrtm(target_state)
        qfids = np.array([np.abs(np.trace(sqrtm(sqrt_rho @ rho_rbm[i, :, :] @ sqrt_rho))) for i in range(rho_rbm.shape[0])])                                                                           
    return qfids


