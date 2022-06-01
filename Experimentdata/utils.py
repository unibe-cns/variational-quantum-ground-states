import numpy as np
import numba
import matplotlib.pyplot as plt
import time
import os
import state_preparation
from scipy.linalg import sqrtm
from pathlib import Path
import json
import pylogging

import hxsampling.hxsampler as hxs
import hxsampling.utils as utils
import pylogging

log = pylogging.get("rob_utils")

def list_to_strstack(ls):
    """ generate a column string of given list entries """
    st = ""
    for t in ls:
        st += (str(t) + "\n")
    return st

def set_size(width="standard", fraction=1, subplots=(1, 1)):
    """ Returns uniform figure dimensions
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    Input:
    width      float or string         Document width in points, or string of predined document type
    fraction   float, optional         Fraction of the width which you wish the figure to occupy
    subplots    array-like, optional    The number of rows and columns of subplots.
    Returns:
    fig_dim     tuple      Dimensions of figure in inches
    """
    if width is None or width == 'standard':
        width_pt = 418.25372
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def random_noise_block(size, noisemultiplier=1, mixsigns=False):
    """ Returns random assignment of noise sources to targets
    Input:
    size            tuple       shape of the noise block
    noisemultiplier int         how many noise sources connect to one target
    mixsigns        boolean     whether to include both types of noise
    """
    noise_block = np.zeros(size)
    for i in range(size[1]):
        rands = np.random.choice(range(size[0]), int(noisemultiplier), replace=False)
        for multi in range(int(noisemultiplier)):
            if not mixsigns:
                noise_block[rands[multi], i] = 1
            else:
                sign = np.sign(np.random.random() - 0.5)
                noise_block[rands[multi], i] =  sign
    return noise_block

def save_data(settings, array_dict, outpath, nosettings=False):
    """ save settings dict and data arrays to a given path """
    # save settings json
    if not nosettings:
        with open("{}/run_{}_settings.json".format(outpath, settings.id),"w") as f:
            json.dump(vars(settings), f)
    
    # save data
    for (arname, array) in array_dict.items():
        with open("{}/run_{}_{}.npy".format(outpath, settings.id, arname), "wb") as f:
            np.save(f, array)

def LIF_sampling(hxsampler, dur, dt, repetitions, return_spikes=False):
    """ LIF sampling on the BrainScaleS-2 chip: perform 'repetition' experiments of real time length 'dur' with sampling interval 'dt', optionally return spike trains
    """
    hxsampler.update_parameters() # update parameters on the chip
    
    rep_spikes = []
    # record spike trains of the specified experiment
    log.debug("Calling run_network ")
    spikes = hxsampler.run_network(dur, input_spikes=None, return_inputs=False, readout_neuron=None, plot_raster=False, save_parameters=False, set_parameters=False)
    if return_spikes: # conditionally collect spike trains for all experiments
        rep_spikes.append(spikes)
    
    # storage for samples obtained from the spike trains given the sampling time dt
    samples = [hxs.get_states_from_spikes(hxsampler.number_of_neurons, spikes, hxsampler.measured_taurefs, dt=dt, duration=dur)]
    log.debug("Finished states from spikes")
    
    # repeat identical experiments 'repetitions' times
    for reps in range(repetitions - 1):
        log.debug("Calling run_network ")
        new_spikes = hxsampler.run_network(dur, input_spikes=None, return_inputs=False, readout_neuron=None, plot_raster=False, save_parameters=False, set_parameters=False)
        new_samples = hxs.get_states_from_spikes(hxsampler.number_of_neurons, new_spikes, hxsampler.measured_taurefs, dt=dt, duration=dur)
        log.debug("Finished states from spikes")
        if return_spikes: # conditionally collect spike trains
            rep_spikes.append(new_spikes) 
        samples.append(new_samples) # always collect samples

    rep_spikes = rep_spikes
    samples = np.array(samples)
    return samples, rep_spikes # return samples and spike trains (the latter may be emtpy)

@numba.jit(nopython=True)
def calc_dkl(n_vis, p_target, p_train):
    """ Returns the DKL between target and trained distribution """
    dkl = 0.
    for vis_state in range(2**n_vis):
        if p_train[vis_state] != 0 and p_target[vis_state] != 0:
            dkl += p_target[vis_state] * np.log(p_target[vis_state] / p_train[vis_state])
    return dkl 

def init_wb(n, n_vis, w_max=+1, w_min=-1, b_max=+1, b_min=-1, spin=False, distr="standard", translate=False):
    """ Returns weight and bias initialization """
    # randomly initialize weights
    if distr == "standard":
        J = w_max*(2*np.random.random(size = (n, n)) - 1)
        J = 0.5 * (J + J.T)
        #J[:n_vis,:n_vis] = 0.0
        #J[n_vis:,n_vis:] = 0.0
        h = b_max*(2*np.random.random(size=n) - 1)
    elif distr == "uniform":
        J = w_max*(2*np.random.random(size = (n, n)) - 1)
        #J[:n_vis,:n_vis] = 0.0
        #J[n_vis:,n_vis:] = 0.0
        J[n_vis:,:n_vis] = J[:n_vis,n_vis:].T
        h = b_max*(2*np.random.random(size=n) - 1)
    elif distr == "gauss":
        J = w_max*np.random.normal(size = (n, n)) 
        #J[:n_vis,:n_vis] = 0.0
        #J[n_vis:,n_vis:] = 0.0
        J[n_vis:,:n_vis] = J[:n_vis,n_vis:].T
        h = b_max*np.random.normal(size=n)
    np.fill_diagonal(J, 0.0)
    assert (J == J.T).all()

    if spin:
        return J.astype(np.float), h.astype(np.float)
    
    # transfer "spin glass interactions" to RBM weights
    if translate:
        W = 4*J
        b = 2*h - 2*J.sum(axis=1) - 0.5*J.sum() + h.sum()
    else:
        W = J
        b = h
    
    return W.astype(np.float), b.astype(np.float)

def params_spin_to_bin(J, h):
    """ Converts spin-like parameters to neuron-like parameters """
    W = 4*J
    b = 2*h - 2*J.sum(axis=1)
    return W, b

def mask_weights_rbm(w, n_vis, n_hid):
    """ Returns weight matrix for restricted Boltzmann machine """
    # remove intra layer connections
    w[n_vis:, n_vis:] = np.zeros((n_hid, n_hid))
    w[:n_vis, :n_vis] = np.zeros((n_vis, n_vis))
    np.fill_diagonal(w, 0.)
    return w

def mask_weights_prrbm(w, n_vis, n_hid, restricted_layer="hid"):
    """ Returns weight matrix for Boltzmann machine with restricted hidden or visible layer"""
    # remove intra layer connections for one layer
    if restricted_layer == "hid":
        w[n_vis:, n_vis:] = np.zeros((n_hid, n_hid))
    elif restricted_layer == "vis":
        w[:n_vis, :n_vis] = np.zeros((n_vis, n_vis))
    np.fill_diagonal(w, 0.)
    return w

def mask_weights_vprbm(w, n_vis, n_hid):
    """ Returns weight matrix for Boltzmann machine where visible units are not interconnected """
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
    """ Returns weight matrix of a deep Boltzmann machine with given layer structure """
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

def mask_weights_regrbm(dw,db, N_vis, N_hid):
    """ Returns a weight and bias vector for a double RBM with direct connections between corresponding visible units"""
    N = 2*N_hid + 2*N_vis
    db_nvis_mean = 0.5*(db[N_vis:2*N_vis] + db[:N_vis])
    db_nhid_mean = 0.5*(db[2*N_vis:2*N_vis+N_hid] + db[2*N_vis+N_hid:])
    db[N_vis:2*N_vis] = db_nvis_mean
    db[:N_vis] = db_nvis_mean
    db[2*N_vis:2*N_vis+N_hid] = db_nhid_mean
    db[2*N_vis+N_hid:] = db_nhid_mean
    dw_rbmT_mean = 0.5*(dw[2*N_vis:2*N_vis+N_hid,:N_vis] + dw[2*N_vis+N_hid:,N_vis:2*N_vis])
    dw[2*N_vis:2*N_vis+N_hid,:N_vis] = dw_rbmT_mean
    dw[2*N_vis+N_hid:,N_vis:2*N_vis] = dw_rbmT_mean
    dw[:N_vis,2*N_vis:2*N_vis+N_hid] = dw_rbmT_mean.T
    dw[N_vis:2*N_vis, 2*N_vis+N_hid:] = dw_rbmT_mean.T
    dw[:2*N_vis,:2*N_vis] = 0.0
    dw[:N_vis,2*N_vis+N_hid:] = 0.0
    dw[2*N_vis+N_hid:,:N_vis] = 0.0
    dw[N_vis:2*N_vis+N_hid,N_vis:2*N_vis+N_hid] = 0.0
    dw[2*N_vis:,2*N_vis:] = 0.0
    return dw, wb

@numba.jit(nopython=True)
def sigmoid(x):
    """ Returns the standard logistic function of the input """
    return 1./(1. + np.exp(-x))

def calc_joint_distribution(w, b, N_visible, N_hidden, spin=False):
    """ calculates the joint distribution by summing all configurations """
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

def calc_pjoint(w, b, N, spin=False):
    """ calculates the joint distribution by summing all configurations """
    p_joint = np.zeros(2**N)
    for zi in np.arange(2**N):
        z = dec_to_bin(zi, N)
        if spin: z = bin_to_spin(z)
        nege = 0.5*np.dot(z, np.dot(w, z)) + np.dot(b, z)
        p_joint[zi] = np.exp(nege)
    return p_joint/p_joint.sum()

@numba.jit
def calc_pvis_ana(w, b, vis_states, spin=False):
    """ calculate the visible distribution """
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
    """ return spin config from bin config """
    return 2*bin_string - 1

@numba.jit(nopython=True)
def spin_to_bin(spin_string):
    """ return bin config from spin config """
    return 0.5*(spin_string + 1)

def gen_states(N, spin=False):
    """ generate all configurations of length N"""
    states = np.zeros((2**N, N), dtype=float)
    for i in range(2**N):
        states[i] = dec_to_bin(i, N)
        if spin:
            states[i] = bin_to_spin(states[i])
    return states

@numba.jit(nopython=True)
def calc_pvis(samples, N_vis, N_hid, spin=False):
    """ Calculate the visible distriution from samples """
    p = np.zeros(2**N_vis)
    for i in range(samples.shape[0]):
        state_vals_vis = bin_to_dec(samples[i, :N_vis], N_vis)
        p[int(state_vals_vis)] += 1.

    p /= np.shape(samples)[0]
    return p

@numba.jit(nopython=True)
def weights_to_flat_RBM(w, n_vis):
    """ Returns flattened weights of a RBM"""
    w_flat = w[:n_vis, n_vis:].flatten()
    return w_flat

@numba.jit(nopython=True)
def weights_to_square_RBM(w_flat, n_vis, n):
    """ Returns a square weight matrix for a RBM """
    w_square = np.zeros((n, n)).astype(np.float64)
    w_square[:n_vis,n_vis:] = w_flat.reshape(n_vis,n - n_vis)
    w_square[n_vis:,:n_vis] = w_square[:n_vis,n_vis:].T
    return w_square

@numba.jit(nopython=True)
def params_to_wb_RBM(params, n_vis, n):
    """ Returns weights and biases of a RBM """
    w = weights_to_square_RBM(params[:-n], n_vis, n)
    b = params[-n:]
    return w, b

@numba.jit(nopython=True)
def wb_to_params_RBM(w, b, n_vis):
    """ Returns a 1D parameter vector describing a RBM """
    wflat = weights_to_flat_RBM(w, n_vis)
    params = np.concatenate((wflat, b))
    return params 

def dec_to_bin(x, n):
    """ turn integer (decimal system) into corresponding binary string """
    binary = np.binary_repr(x, width = n)
    ret = np.array([int(y) for y in list(binary)])
    return ret 

@numba.jit(nopython=True)
def bin_to_dec(x, n):
    """ turn a binary string into its corresponding decimal integer """
    decimal = 0
    for i in range(n):
        decimal += x[i] * 2 ** (n - 1 - i)
    return decimal

@numba.jit(nopython=True)
def p_vh_bin1(h, w, a, spin=False):
    z = np.dot(w, h) + a
    factor = 2. if spin else 1.
    return sigmoid(factor*z)

@numba.jit(nopython=True)
def p_hv_bin1(v, w, b, spin=False):
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
def bin_sampling(p, spin=False):
    samples = []
    for i in range(p.shape[0]):
        sample = []
        for j in range(p.shape[1]):
            sample.append(np.random.binomial(1, p[i,j]))
        samples.append(sample)
    return np.array(samples)

@numba.jit(nopython=True)
def bin_sampling1(p, spin=False):
    sample = np.zeros(p.shape[0])
    for j in range(p.shape[0]):
        sample[j] = np.random.binomial(1, p[j])
    return sample if not spin else bin_to_spin(sample)

@numba.jit(nopython=True)
def gibbs_h(v, w, b, spin=False):
    phv = p_hv_bin1(v, w, b, spin=spin)
    h = bin_sampling1(phv, spin=spin)
    return h#, phv

@numba.jit(nopython=True)
def gibbs_v(h, w, b, spin=False):
    pvh = p_vh_bin1(h, w, b, spin=spin)
    v = bin_sampling1(pvh, spin=spin)
    return v#, pvh

@numba.jit(nopython=True)
def rand_choice_nb(arr, prob):
    """ Returns sample from array with given probabilties
    Note: from https://github.com/numba/numba/issues/2539 
    arr A 1D numpy array of values to sample from.
    prob A 1D numpy array of probabilities for the given samples.
    """
    randind = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
    return arr[randind]

@numba.jit(nopython=True)
def gibbs_sampling(w, b, vis_states, num_chains=4, num_samples=100, pvis=None, burn_in=0, spin=False):
    N_vis = vis_states.shape[1]
    wres = w[:N_vis, N_vis:]
    wres = np.ascontiguousarray(wres)
    vb = b[:N_vis]
    hb = b[N_vis:]
    N_hid = hb.shape[0]
    samples = np.zeros((num_chains, num_samples, N_vis + N_hid))
    p_sample = np.zeros((2**N_vis, 2**N_hid))
    
    N_hid = hb.shape[0]
    for c in range(num_chains):
        if pvis is None:
            vind = np.random.randint(vis_states.shape[0])
        else:
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
                p_sample[int(vind), int(hind)] += 1
    samples = samples.reshape(num_chains*num_samples, N_vis + N_hid)
    return p_sample/num_chains/num_samples, samples
