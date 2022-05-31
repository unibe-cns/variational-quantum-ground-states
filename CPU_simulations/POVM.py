import numpy as np
import matplotlib.pyplot as plt
import time as time
import re
import scipy.sparse as sparse
import scipy.sparse.linalg as sLA
import itertools
import string
import numba
from numba.typed import List


# build sparse Pauli matrices
Sp = sparse.csr_matrix((np.array([[0, 1], [0, 0]])))
Sm = np.transpose(Sp)
Sx = Sp + Sm
Sy = -1j*(Sp - Sm)
Sz = sparse.csr_matrix(np.array([[1, 0], [0, -1]]))
S0 = sparse.eye(2)


# mutual info between single RV's of a flattened discrete distribution with given number of outcomes per RV
def mutual_info(pjoint, i, j, num_outcomes=4):
    N = int(np.log2(pjoint.shape[0])/np.log2(num_outcomes))
    margi = np.setdiff1d(range(N), [i])
    margj = np.setdiff1d(range(N), [j])
    margij = np.setdiff1d(range(N), [i, j])
    pmargi = povm_marginals(margi, pjoint, num_outcomes=num_outcomes)
    pmargj = povm_marginals(margj, pjoint, num_outcomes=num_outcomes)
    if i == j:
        # here the conditional probability is the identity matrix
        pmargij = np.eye(pmargi.shape[0])*pmargi
    else:
        pmargij = povm_marginals(margij, pjoint, num_outcomes=num_outcomes)
    # sum up over the outcome space
    mi = np.sum(pmargij * np.nan_to_num(np.log(pmargij/np.outer(pmargi, pmargj))))
    return mi

# calculates the marginal POVM distribution given the spins to be marginalized over
def povm_marginals(as_marg, p_joint, num_outcomes=4):
    n_spin = int(np.log2(p_joint.shape[0])//np.log2(num_outcomes)) # number of spins TODO depends on num_outcomes
    p_joint_ndim = p_joint.reshape(*([num_outcomes]*n_spin)) # reshape joint distribution
    return p_joint_ndim.sum(axis=tuple(as_marg)) # marginalize out given spins

# calculates povm pair marginals for all spin pairs
def povm_pair_marginals(p_joint, num_outcomes=4):
    n_spin = int(np.log2(p_joint.shape[0])//np.log2(num_outcomes)) # the //2 depends on num_outcomes = 4
    spin_tuples = []
    p_margs = np.zeros(((n_spin**2 - n_spin)//2, 4, 4))
    tuple_count = 0
    for i in range(n_spin):
        for j in range(i):
            p_margs[tuple_count, :, :] = povm_marginals(np.setdiff1d(range(n_spin), [i, j]), p_joint)
            spin_tuples.append((i, j))
            tuple_count += 1
    return spin_tuples, p_margs

# calculates the conditional probability for all outcomes given a subset of spins
def povm_conditionals(as_cond, probs, margs=None, num_outcomes=4):
    n_spin = int(np.log2(probs.shape[0])//np.log2(num_outcomes)) # number of spins TODO depends on num_outcomes
    marg_inds = np.setdiff1d(range(n_spin), as_cond)
    if margs is None:
        margs = povm_marginals(marg_inds, probs)
    norms = np.zeros(probs.shape[0])
    for a in range(probs.shape[0]):
        ind = tuple(np.array(POVM.to_a_tuple(a, n_spin), dtype=int)[as_cond].tolist())
        norms[a] = margs[ind]
    return probs/norms

# generate a column string of given list entries 
def list_to_strstack(ls):
    st = ""
    for t in ls:
        st += (str(t) + "\n")
    return st

# turn an integer in range [0, 4^n - 1] into an n-dimensional coordinate tuple
@numba.jit(nopython=True)
def to_a_tuple(a_joint, n):
    vols = [4**i for i in range(n)]
    coords = [0]*n
    temp_a = int(a_joint)
    for i in range(n - 1, -1, -1):
        coords[i] = temp_a // vols[i]
        temp_a -= int(coords[i]*vols[i])
    return coords

# turn a n-dim coordinate tuple into an integer in range [0, 4^n - 1]
@numba.jit(nopython=True)
def to_a_joint(a_tuple, n):
    vols = 4**np.arange(n)
    return np.dot(np.array(a_tuple), vols)

#@numba.jit(nopython=True)
def kronprod(op_list):
    kron_prod = op_list[0]
    for i in range(1, len(op_list)):
        kron_prod = np.kron(kron_prod, op_list[i])
    return kron_prod

def tetra_povm(n):
    # define elementary POVM elements
    m0 = np.array([[0.5 + 0*1j, 0 + 0*1j], [0 + 0*1j, 0 + 0*1j]])
    m1 = 1/6*np.array([[1 + 0*1j, np.sqrt(2) + 0*1j], [np.sqrt(2) + 0*1j, 2 + 0*1j]])
    m2 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) - np.sqrt(6)*1j], [-np.sqrt(2) + np.sqrt(6)*1j, 4 + 0*1j]])
    m3 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) + np.sqrt(6)*1j], [-np.sqrt(2) - np.sqrt(6)*1j, 4 + 0*1j]])
    m_single = [m0, m1, m2, m3]

    M = np.zeros((4**n, 2**n, 2**n), dtype=np.complex64) # POVM elements
    T = np.zeros((4**n, 4**n), dtype=np.float64) # overlap matrix
    T_single = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            T_single[i, j] = np.trace(np.dot(m_single[i], m_single[j]))

    Tinv_single = np.ascontiguousarray(np.linalg.inv(T_single))
    #tol = np.finfo(np.float).eps*10
    #Tinv_single.real[np.real(Tinv_single) <= tol] = 0.0

    typed_tsingle_list = List([T_single]*n)
    typed_tinvsingle_list = List([Tinv_single]*n)
    T = kronprod(typed_tsingle_list)
    Tinv = kronprod(typed_tinvsingle_list)

    outcomes = [] # tuple representation of elements
    for a_joint in range(4**n):
        a_i = to_a_tuple(a_joint, n)
        Ms = [m_single[a_i[i]] for i in range(len(a_i))]
        Ma = kronprod(Ms)
        outcomes.append(a_i)
        M[a_joint, :, :] = Ma.copy()
    return M, Tinv, outcomes

@numba.jit(nopython=True)
def tetra_povm_numba(n):
    # define elementary POVM elements
    m0 = np.array([[0.5 + 0*1j, 0 + 0*1j], [0 + 0*1j, 0 + 0*1j]])
    m1 = 1/6*np.array([[1 + 0*1j, np.sqrt(2) + 0*1j], [np.sqrt(2) + 0*1j, 2 + 0*1j]])
    m2 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) - np.sqrt(6)*1j], [-np.sqrt(2) + np.sqrt(6)*1j, 4 + 0*1j]])
    m3 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) + np.sqrt(6)*1j], [-np.sqrt(2) - np.sqrt(6)*1j, 4 + 0*1j]])
    m_single = [m0, m1, m2, m3]

    M = np.zeros((4**n, 2**n, 2**n), dtype=np.complex64) # POVM elements
    T = np.zeros((4**n, 4**n), dtype=np.float64) # overlap matrix
    T_single = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            T_single[i, j] = np.trace(np.dot(m_single[i], m_single[j]))

    Tinv_single = np.ascontiguousarray(np.linalg.inv(T_single))

    typed_tsingle_list = List([T_single]*n) 
    typed_tinvsingle_list = List([Tinv_single]*n) 
    T = kronprod(typed_tsingle_list)
    Tinv = kronprod(typed_tinvsingle_list)

    #assert np.isclose((np.linalg.inv(T) - Tinv).all(), 0)

    outcomes = [] # tuple representation of elements
    for a_joint in range(4**n):
        a_i = to_a_tuple(a_joint, n)
        Ms = [m_single[a_i[i]] for i in range(len(a_i))]
        Ma = kronprod(Ms)
        outcomes.append(a_i)
        #Ma = m_single[a_i[0]]
        #for j in a_i[1:]: # multiply together all elements of the tuple
        #    Ma = np.kron(m_single[j], Ma)
        M[a_joint, :, :] = Ma.copy()
        #T[a_joint, a_joint] = np.trace(np.dot(Ma, Ma)) # calculate diag entry of T
        #for k in range(a_joint): # calculate off-diag entries of T 
        #    T[a_joint, k] = np.trace(np.dot(Ma.astype(np.complex64), M[k]))
        #    T[k, a_joint] = T[a_joint, k]
    return M, Tinv, outcomes

@numba.jit(nopython=True)
def tetra_povm_new(n):
    # define elementary POVM elements
    m0 = np.array([[0.5 + 0*1j, 0 + 0*1j], [0 + 0*1j, 0 + 0*1j]])
    m1 = 1/6*np.array([[1 + 0*1j, np.sqrt(2) + 0*1j], [np.sqrt(2) + 0*1j, 2 + 0*1j]])
    m2 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) - np.sqrt(6)*1j], [-np.sqrt(2) + np.sqrt(6)*1j, 4 + 0*1j]])
    m3 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) + np.sqrt(6)*1j], [-np.sqrt(2) - np.sqrt(6)*1j, 4 + 0*1j]])
    m_single = [m0, m1, m2, m3]

    M = np.zeros((4**n, 2**n, 2**n), dtype=np.complex64) # POVM elements
    T = np.zeros((4**n, 4**n), dtype=np.complex64) # overlap matrix
    T_single = np.zeros((4, 4), dtype=np.complex64)
    for i in range(4):
        for j in range(4):
            T_single[i, j] = np.trace(np.dot(m_single[i], m_single[j]))

    Tinv_single = np.ascontiguousarray(np.linalg.inv(T_single))

    typed_tsingle_list = List([T_single]*n) 
    typed_tinvsingle_list = List([Tinv_single]*n) 
    T = kronprod(typed_tsingle_list)
    Tinv = kronprod(typed_tinvsingle_list)

    #assert np.isclose((np.linalg.inv(T) - Tinv).all(), 0)

    outcomes = [] # tuple representation of elements
    for a_joint in range(4**n):
        a_i = to_a_tuple(a_joint, n)
        Ms = [m_single[a_i[i]] for i in range(len(a_i))]
        Ma = kronprod(Ms)
        outcomes.append(a_i)
        #Ma = m_single[a_i[0]]
        #for j in a_i[1:]: # multiply together all elements of the tuple
        #    Ma = np.kron(m_single[j], Ma)
        M[a_joint, :, :] = Ma.copy()
        #T[a_joint, a_joint] = np.trace(np.dot(Ma, Ma)) # calculate diag entry of T
        #for k in range(a_joint): # calculate off-diag entries of T 
        #    T[a_joint, k] = np.trace(np.dot(Ma.astype(np.complex64), M[k]))
        #    T[k, a_joint] = T[a_joint, k]
    return M, T_single, Tinv_single



# calculate the tetra POVM elements and the overlap matrix for a given number of spins
# TODO implement with sparse matrices
@numba.jit(nopython=True)
def tetra_povm2(n):
    # define elementary POVM elements
    m0 = np.array([[0.5 + 0*1j, 0 + 0*1j], [0 + 0*1j, 0 + 0*1j]])
    m1 = 1/6*np.array([[1 + 0*1j, np.sqrt(2) + 0*1j], [np.sqrt(2) + 0*1j, 2 + 0*1j]])
    m2 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) - np.sqrt(6)*1j], [-np.sqrt(2) + np.sqrt(6)*1j, 4 + 0*1j]])
    m3 = 1/12*np.array([[2 + 0*1j, -np.sqrt(2) + np.sqrt(6)*1j], [-np.sqrt(2) - np.sqrt(6)*1j, 4 + 0*1j]])
    m_single = [m0, m1, m2, m3]
    
    M = np.zeros((4**n, 2**n, 2**n), dtype=np.complex64) # POVM elements
    T = np.zeros((4**n, 4**n), dtype=np.complex64) # overlap matrix
    T2 = np.zeros((4**n, 4**n), dtype=np.complex64) # overlap matrix
    T_single = np.zeros((4, 4), dtype=np.complex64)
    for i in range(4):
        for j in range(4):
            T_single[i, j] = np.trace(np.dot(m_single[i], m_single[j]))

    T_temp_prev = T_single
    for ni in range(2, n):
        T_temp = np.zeros((4**ni, 4**ni), dtype=np.complex64)
        T_temp = np.kron(T_single, T_temp)
        T_temp_prev = T_temp
    T2 = T_temp_prev

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
            #MaMaprime = np.dot(Ma.astype(np.complex64), M[k])
            #print(MaMaprime)
            #assert (MaMaprime == MaMaprime.T).all()
            T[a_joint, k] = np.trace(np.dot(Ma.astype(np.complex64), M[k]))
            T[k, a_joint] = T[a_joint, k]#np.trace(np.dot(M[k], Ma))
    print(T.shape)
    print(T2.shape)
    assert (T == T2).all()
    return M, T_single, outcomes

#@numba.jit(nopython=True)
def pauli4_povm(n):
    # define elementary POVM elements
    psiz = np.array([1, 0], dtype=np.complex64) 
    m0 = 1/3*np.outer(psiz, psiz)
    psix = np.array([1, 1], dtype=np.complex64)
    m1 = 1/3*np.outer(psix, psix)/np.sqrt(2)
    psiy = np.array([1, 1j], dtype=np.complex64)
    m2 = 1/3*np.outer(psiy, psiy)/np.sqrt(2)
    m3 = np.eye(2) - m0 - m1 - m2
    m_single = [m0, m1, m2, m3]
    print(m_single)

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
            MaMaprime = np.dot(Ma.astype(np.complex64), M[k])
            T[a_joint, k] = np.trace(MaMaprime)
            T[k, a_joint] = T[a_joint, k]#np.trace(np.dot(M[k], Ma))
    #fig, ax = plt.subplots(2, sharex=True, sharey=True)
    #ax[0].imshow(np.real(T))
    #ax[1].imshow(np.imag(T))
    #ax[0].set_ylabel("a1")
    #ax[0].set_xlabel("a0")
    #plt.show()
    return M, T, outcomes



#calculate the tetra POVM elements and the overlap matrix for a given number of spins and given rotation angles
#@numba.jit(nopython=True)
def angle_tetra_povm(n, theta=0, phi=0):
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
    #print(m_single.shape)
    #m_single = np.array([0.25*(s0 + np.sum([rs[j][i]*S[i] for i in range(3)], 0)) for j in range(4)])

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

# returns the string set of all possible basis states for N qubits
def build_basis_strings(N):
    all_states = []
    for i in range(2**N):
        bstring = format(i, "b")
        bstring = "0"*(N - len(bstring)) + bstring
        all_states.append(bstring)
    return np.array(all_states)

# builds and returns the matrix representation of the given operator string, e.g. "1xy" = 1 x simga_x x sigma_y
def build_op(op_string):
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

# build all spin operator strings up to a specified order
def build_op_strings(N=2, up_to_order=2, regex_filter=None):
    up_to_order = min(up_to_order, N)
    chars = ["x", "y", "z", "1"]
    for item in itertools.product(chars, repeat=N):
        perm = "".join(item)
        num_ones = perm.count("1")
        if num_ones < N - up_to_order or num_ones == N:
            continue
        if not regex_filter is None and len(re.findall(regex_filter, perm)) != 0:
            #print(f"regex filter {regex_filter} on {perm} yields: {re.findall(regex_filter, perm)}" )
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
            #print(f"regex filter {regex_filter} on {string} yields: {re.findall(regex_filter, string)}" )
            continue
        yield string

# build all spin operator strings up to a specified order
def build_op_strings_two_point(N=2, first="x", second="x", regex_filter=None):
    #chars = ["x", "y", "z", "1"]
    for x in range(N):
        for y in range(N):
            if x!=y:
                string = "1"*N
                string = string[:x] + first + string[x+1:]
                string = string[:y] + second + string[y+1:]
                if not regex_filter is None and len(re.findall(regex_filter, string)) != 0:
                    #print(f"regex filter {regex_filter} on {string} yields: {re.findall(regex_filter, string)}" )
                    continue
                yield string

# calculate the POVM distribution of a given state
def calc_probs(state, povm):
    if povm is None:
        raise ValueError("Specify a POVM to be used.")
    else:
       if not callable(povm) and len(povm) >= 2:
            M, Tinv, outcomes = povm
       elif callable(povm):
            n = int(np.log2(state.shape[0])) # number of spins
            M, Tinv, outcomes = povm(n)
            #Tinv = np.linalg.inv(T)
         
    Mstate = np.einsum("ijk,kl->ijl", M, state)
    p = np.trace(Mstate, axis1=1, axis2=2).astype(np.float)
    return p, M, Tinv, outcomes

# calculate the operator coefficients for a specified POVM
def calc_op_coeffs(op, povm):
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

# calculate operator expectations
def calc_op_expectations(probs, op_coeffs):
    return np.dot(op_coeffs, probs)
##
def build_liouvillian(H_sparse, c_ops, gamma, M, Tinv):
    N = np.einsum("kl,lno->kno", Tinv, M)
    NM = np.einsum("ijk,lkn->iljn", N, M)
    MN = np.einsum("ijk,lkn->iljn", M, N)
    H = H_sparse.todense()
    arg = np.einsum("kl,mnlp->mnkp", H, NM - MN)
    A = -1j*np.trace(arg, axis1=2, axis2=3)
    B = np.zeros_like(A)
    for cop in c_ops:
        if sparse.issparse(cop):
            cop = cop.todense()
        firstfact = np.einsum("kl,mlo->mko", cop, N)
        secondfact = np.einsum("kl,mlo->mko", cop.conj(), M)
        firstarg = np.einsum("klm,nmp->knlp", firstfact, secondfact)
        firstterm = np.trace(2*firstarg, axis1=2, axis2=3)
        cop2 = cop.conj().dot(cop)
        secondarg = np.einsum("kl,mnlp->mnkp", cop2, NM + MN)
        secondterm = np.trace(secondarg, axis1=2, axis2=3)
        B += 0.5*gamma*(firstterm - secondterm)
    return A + B

def build_POVM_liouvillian(N_spin, H_sparse, c_ops, M, N):
    A = sparse.lil_matrix(np.zeros(shape=[4**N_spin, 4**N_spin], dtype=np.complex))
    B = sparse.lil_matrix(np.zeros(shape=[4**N_spin, 4**N_spin], dtype=np.complex))
    firstterm = sparse.lil_matrix(np.zeros(shape=H_sparse.shape, dtype=np.complex))
    secondterm = sparse.lil_matrix(np.zeros(shape=H_sparse.shape, dtype=np.complex))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            M_sparse = sparse.csr_matrix(M[i,:,:])
            N_sparse = sparse.csr_matrix(N[j,:,:])
            NM = sparse.lil_matrix(N[j,:,:].dot(M[i,:,:]))
            MN = sparse.lil_matrix(M[i,:,:].dot(N[j,:,:]))

            A[i, j] = -1j*(H_sparse.dot(NM - MN)).diagonal().sum()
            for cop in c_ops:
                cop_dag = cop.T.conj()
                firstterm[:,:] = 2*cop.dot(N_sparse).dot(cop_dag).dot(M_sparse)
                secondterm[:,:] = cop_dag.dot(cop).dot(NM + MN)
                B[i, j] += 0.5*(firstterm - secondterm).diagonal().sum()
    return A + B
