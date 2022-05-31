import numpy as np
import numba
import matplotlib.pyplot as plt

@numba.jit(nopython=True)
def weights_to_flat_RBM(w, n_vis):
    w_flat = w[:n_vis, n_vis:].flatten()
    return w_flat

@numba.jit(nopython=True)
def weights_to_square_RBM(w_flat, n_vis, n):
    w_square = np.zeros((n, n)).astype(np.float64)
    w_square[:n_vis,n_vis:] = w_flat.reshape(n_vis,n - n_vis)
    w_square[n_vis:,:n_vis] = w_square[:n_vis,n_vis:].T
    return w_square

@numba.jit(nopython=True)
def sigmod(x):
    return 1./(1. + np.exp(-x))

# calculates the joint distribution by summing all configurations
def calc_joint_distribution(w, b, N_visible, N_hidden, spin=False):
    vs = np.zeros((2**N_visible, N_visible))
    p_joint = np.zeros((2**N_visible, 2**N_hidden))
    for ai in np.arange(2**N_visible):
        vs[ai, :] = dec_to_bin(ai, N_visible)
        if spin: vs[ai, :] = bin_to_spin(vs[ai, :])
        for bi in np.arange(2**N_hidden):
            hs[bi, :] = dec_to_bin(bi, N_hidden)
            if spin: hs[bi, :] = bin_to_spin(hs[bi, :])
            p_joint[ai, bi] = np.exp(-energy(vs[ai, :], hs[bi, :], w, a, b))  
    return p_joint/p_joint.sum()

# calculate the visible distribution
@numba.jit
def calc_pvis_ana(w, b, vis_states, spin=False):
    N_vis = vis_states.shape[1]
    pvis = np.zeros(vis_states.shape[0])

    wres = w[:N_vis, N_vis:]
    wres = np.ascontiguousarray(wres)

    vb = b[:N_vis]
    hb = b[N_vis:]
    N_hid = hb.shape[0]
    C = np.exp(np.dot(vis_states, vb.T))
    B = 1 + np.exp(np.dot(vis_states, wres) + hb) if not spin else 2*np.cosh(np.dot(vis_states, wres) + hb)
    pvis = C
    for i in range(N_hid):
        pvis *= B[:,i]
    return pvis/pvis.sum()

@numba.jit(nopython=True)
def bin_to_spin(bin_string):
    return 2*bin_string - 1

@numba.jit(nopython=True)
def spin_to_bin(spin_string):
    return 0.5*(spin_string + 1)

def gen_states(N, spin=False):
    states = np.zeros((2**N, N), dtype=float)
    for i in range(2**N):
        states[i] = dec_to_bin(i, N)
        if spin:
            states[i] = bin_to_spin(states[i])
    return states

@numba.jit(nopython=True)
def calc_pvis(samples, N_vis, N_hid, spin=False):#, power_array):
    p = np.zeros(2**N_vis)
    for i in range(samples.shape[0]):
        #state_vals_vis = int(np.dot(samples[i, :], power_array[:]) // 2**N_hid)
        state_vals_vis = bin_to_dec(samples[i, :N_vis], N_vis)
        p[int(state_vals_vis)] += 1.

    p /= np.shape(samples)[0]
    return p

@numba.jit(nopython=True)
def calc_pvis_vinds(samples, N_vis):
    p = np.zeros(2**N_vis)
    for i in range(samples.shape[0]):
        p[int(samples[i])] += 1.

    p /= np.shape(samples)[0]
    return p


def SNN_sample(w, b, sim_dur, t_ref, sampling_interval, N_vis, N_hid, power_array):
    pop_visible, pop_hidden, sd, multimeter1, noise_ex, noise_in = createSNN(N_vis, N_hid, params_neuron, 
                                                                    params_poisson, params_spike_detector)
    synBiases = setBiases(b, pop_visible, pop_hidden, alpha_1kHz, u_bar_0_1kHz, noise_ex, noise_in, setDirect=False)
    synWeights = setWeights(w, pop_visible, pop_hidden, alpha_1kHz, noise_ex, noise_in)
    nest.Simulate(sim_dur)
    samples = sample_joint(sd, t_ref, sim_dur, int(sampling_interval), N_vis, N_hid)[0]
    nest.ResetKernel()
    #p_sampled, bins = np.histogram(v_idx_samples, bins=range(p_true.shape[0] + 1), density=True)
    p_sampled = calc_pvis(samples, N_vis, N_hid)#, power_array)
    #p_sampled_joint = makeDistribution(N_vis+N_hid, samples)
    #SNN_distribution = makeDistribution(N_vis, samples[:,:N_vis])
    return samples, p_sampled#, p_sampled_joint#SNN_distribution

@numba.jit(nopython=True)
def calc_weight_updates(p_true, p_sampled, samples, N_vis, N_hid):
    # calculate weight updates
    dw = np.zeros((N_vis + N_hid, N_vis + N_hid))
    db = np.zeros(N_vis + N_hid) 
    p_ratio = np.zeros(p_true.shape)
    for i in range(p_true.shape[0]):
        if p_sampled[i] != 0:
            p_ratio[i] = 1. - p_true[i]/p_sampled[i] 

    for i in range(samples.shape[0]):
        #vis_state_id = int(np.dot(samples[i,:], power_array[:]) // (2**n_hid))
        vis_state_id = bin_to_dec(samples[i, :N_vis], N_vis)
        db += - p_ratio[int(vis_state_id)]*samples[i, :]
        dw += - p_ratio[int(vis_state_id)]*np.outer(samples[i, :], samples[i, :])
    return dw/samples.shape[0], db/samples.shape[0]

def dec_to_bin(x, n):                                                                                                                                                                
    binary = np.binary_repr(x, width = n)
    ret = np.array([int(y) for y in list(binary)])
    return ret 

@numba.jit(nopython=True)
def bin_to_dec(x, n):
    decimal = 0
    for i in range(n):
        decimal += x[i] * 2 ** (n - 1 - i)
    return decimal

@numba.jit(nopython=True)
def sigmoid(z):
    return 1./(1. + np.exp(-z))

@numba.jit(nopython=True)
def p_vh_bin1(h, w, a, spin):
    z = np.dot(w, h) + a
    factor = 2. if spin else 1.
    return sigmoid(factor*z)

@numba.jit(nopython=True)
def p_hv_bin1(v, w, b, spin):
    z = np.dot(v, w) + b
    factor = 2. if spin else 1.
    return sigmoid(factor*z)

def p_vh_bin(h, w, a):
    z = a[None,:] + np.einsum("ij,kj->ki", w, h)
    return sigmoid(z)

def p_hv_bin(v, w, b):
    z = b[None,:] + np.einsum("ki,ij->kj", v, w)
    return sigmoid(z)

@numba.jit
def bin_sampling(p, spin):
    samples = []
    for i in range(p.shape[0]):
        sample = []
        for j in range(p.shape[1]):
            sample.append(np.random.binomial(1, p[i,j]))
        samples.append(sample)
    return np.array(samples)

@numba.jit(nopython=True)
def bin_sampling1(p, spin):
    sample = np.zeros(p.shape[0])
    for j in range(p.shape[0]):
        sample[j] = np.random.binomial(1, p[j])
    return sample if not spin else bin_to_spin(sample)

@numba.jit(nopython=True)
def gibbs_h(v, w, b, spin):
    phv = p_hv_bin1(v, w, b, spin)
    h = bin_sampling1(phv, spin=spin)
    return h#, phv

@numba.jit(nopython=True)
def gibbs_v(h, w, b, spin):
    pvh = p_vh_bin1(h, w, b, spin=spin)
    v = bin_sampling1(pvh, spin=spin)
    return v#, pvh

@numba.jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    randind = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
    return arr[randind]

@numba.jit(nopython=True)
def gibbs_sampling(w, b, vis_states, num_chains=4, num_samples=100, pvis=None, burn_in=0, spin=False, pdying=None):
    N_vis = vis_states.shape[1]
    #pvis = np.zeros(vis_states.shape[0])
    wres = w[:N_vis, N_vis:]
    wres = np.ascontiguousarray(wres)
    vb = b[:N_vis]
    hb = b[N_vis:]
    N_hid = hb.shape[0]
    N = N_vis + N_hid
    #vis_samples = np.zeros((int(num_chains*num_samples), N_vis))
    #hid_samples = np.zeros((int(num_chains*num_samples), N_hid))
    sample_vinds = np.zeros((num_chains, num_samples))
    samples = np.zeros((num_chains, num_samples, N_vis + N_hid))
    p_sample = np.zeros(2**N_vis)
    
    N_hid = hb.shape[0]
    for c in range(num_chains):
        #v = np.zeros(N_vis)
        #h = np.zeros(N_hid)
        if pvis is None:
            vind = np.random.randint(vis_states.shape[0])
        else:
            #vind = np.random.randint(vis_states.shape[0])
            vind = int(rand_choice_nb(np.arange(vis_states.shape[0]), pvis))
        v = vis_states[vind, :].astype(np.float64)
        for s in range(num_samples + burn_in):
            s = s - burn_in
            h = gibbs_h(v, wres, hb, spin=spin)
            v = gibbs_v(h, wres, vb, spin=spin)
            if s >= 0:
                samples[c, s, :] = np.concatenate((v.copy(), h.copy()))
                hind = bin_to_dec(h, N_hid) if not spin else bin_to_dec(spin_to_bin(h), N_hid)
                vind = bin_to_dec(v, N_vis) if not spin else bin_to_dec(spin_to_bin(v), N_vis)
                sample_vinds[c, s] = vind
                p_sample[int(vind)] += 1
        if not pdying is None:
            for ni in range(N_hid):
                if np.random.random() <= pdying:
                    samples[c, :, N_vis+ni] = np.zeros(num_samples)

    #samples = np.concatenate((vis_samples, hid_samples), axis=1)
    samples = samples.reshape(num_chains*num_samples, N_vis + N_hid)
    sample_vinds = sample_vinds.reshape(num_chains*num_samples)
    return p_sample/num_chains/num_samples, samples, sample_vinds

@numba.jit(nopython=True)
def gibbs_sampling2(w, b, vis_states, num_chains=4, num_samples=100, pvis=None, burn_in=0, spin=False):
    N_vis = vis_states.shape[1]
    #pvis = np.zeros(vis_states.shape[0])
    wres = w[:N_vis, N_vis:]
    wres = np.ascontiguousarray(wres)
    vb = b[:N_vis]
    hb = b[N_vis:]
    N_hid = hb.shape[0]
    #vis_samples = np.zeros((int(num_chains*num_samples), N_vis))
    #hid_samples = np.zeros((int(num_chains*num_samples), N_hid))
    sample_vinds = np.zeros((num_chains, num_samples))
    samples = np.zeros((num_chains, num_samples, N_vis + N_hid))
    p_sample = np.zeros((2**N_vis, 2**N_hid))
    
    N_hid = hb.shape[0]
    for c in range(num_chains):
        #v = np.zeros(N_vis)
        #h = np.zeros(N_hid)
        if pvis is None:
            vind = np.random.randint(vis_states.shape[0])
        else:
            #vind = np.random.randint(vis_states.shape[0])
            vind = int(rand_choice_nb(np.arange(vis_states.shape[0]), pvis))
        v = vis_states[vind, :].astype(np.float64)
        for s in range(num_samples + burn_in):
            s = s - burn_in
            h = gibbs_h(v, wres, hb, spin=spin)
            v = gibbs_v(h, wres, vb, spin=spin)
            if s >= 0:
                samples[c, s, :] = np.concatenate((v.copy(), h.copy()))
                hind = bin_to_dec(h, N_hid) if not spin else bin_to_dec(spin_to_bin(h), N_hid)
                vind = bin_to_dec(v, N_vis) if not spin else bin_to_dec(spin_to_bin(v), N_vis)
                sample_vinds[c, s] = vind
                p_sample[int(vind), int(hind)] += 1
    #samples = np.concatenate((vis_samples, hid_samples), axis=1)
    samples = samples.reshape(num_chains*num_samples, N_vis + N_hid)
    sample_vinds = sample_vinds.reshape(num_chains*num_samples)
    return p_sample/num_chains/num_samples, samples, sample_vinds



def mask_weights_rbm(w, n_vis, n_hid):
    # remove intra layer connections
    w[n_vis:, n_vis:] = np.zeros((n_hid, n_hid))
    w[:n_vis, :n_vis] = np.zeros((n_vis, n_vis))
    np.fill_diagonal(w, 0.)
    return w

def mask_weights_prrbm(w, n_vis, n_hid, restricted_layer="hid"):
    # remove intra layer connections for one layer
    if restricted_layer == "hid":
        w[n_vis:, n_vis:] = np.zeros((n_hid, n_hid))
    elif restricted_layer == "vis":
        w[:n_vis, :n_vis] = np.zeros((n_vis, n_vis))
    np.fill_diagonal(w, 0.)
    return w

def mask_weights_vprbm(w, n_vis, n_hid):
    # remove intra layer connections
    w[:n_vis,:n_vis] = np.zeros((n_vis, n_vis))
    w[n_vis:2*n_vis, n_vis:2*n_vis] = np.zeros((n_vis, n_vis))
    w[2*n_vis:, :n_vis] = np.zeros((n_hid, n_vis))
    w[:n_vis, 2*n_vis:] = np.zeros((n_vis, n_hid))
    w[2*n_vis:, 2*n_vis:] = np.zeros((n_hid, n_hid))
    
    # remove connections between neuron pairs of outocome layers
    for coord in range(2, n_vis, 2):
        w[n_vis:n_vis + coord, coord:n_vis] = np.zeros((coord, n_vis - coord))
        w[n_vis + coord:2*n_vis, :coord] = np.zeros((n_vis - coord, coord))

    np.fill_diagonal(w, 0.)
    return w    

def mask_weights_dbm(w, n_layers):
    ncum = np.cumsum(n_layers)
    for i in range(len(n_layers)):
        ncumi = ncum[i]
        nlayi = n_layers[i]
        ncumprev = 0 if i == 0 else ncum[i - 1]
        ncumnext = -1 if i == len(n_layers) - 1 else ncum[i + 1]
        w[ncumprev:ncumi,ncumprev:ncumi] = np.zeros((nlayi, nlayi))
        if i < len(n_layers) - 1:
            w[ncumnext:, :ncumi] = np.zeros((ncum[-1] - ncumnext, ncumi))
            w[:ncumi, ncumnext:] = np.zeros((ncumi, ncum[-1] - ncumnext))
    np.fill_diagonal(w, 0.)
    return w

