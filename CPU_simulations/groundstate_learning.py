import numpy as np
import numba
import matplotlib
import matplotlib.pyplot as plt
import jit_RBM
import POVM
import time
import TFIM
import os

from pathlib import Path
import json
import os

# from my_plot import set_size
# matplotlib.style.use("default")
# matplotlib.style.use("tex")

@numba.jit(nopython=True)
def sigmoid(x):
    return 1./(1 + np.exp(-x))

@numba.jit(nopython=True)
def weights_to_flat(inds, w):
    w_flat = np.zeros(len(inds[0]))
    for x in range(len(inds[0])):
        w_flat[x] = w[inds[0][x], inds[1][x]]
    return w_flat

@numba.jit(nopython=True)
def weights_to_square(n, inds, w):
    w_square = np.zeros((n, n))
    for x in range(len(inds[0])):
        i = inds[0][x]
        j = inds[1][x]
        w_square[i, j] = w[x]
    return w_square + w_square.T - np.diag(w_square)

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
def wb_to_params_RBM(w, b, n_vis):
    wflat = weights_to_flat_RBM(w, n_vis)
    params = np.concatenate((wflat, b))
    return params

@numba.jit(nopython=True)
def params_to_wb_transinv_RBM(params, N_vis, N):
    N_hid = N - N_vis
    alpha = int(N_hid/N_vis)
    nwparams = N_hid
    b = np.zeros(N)
    w = np.zeros((N, N))
    # the visible biases have only one free parameter
    b[:N_vis] = params[nwparams]
    
    for a1 in range(alpha):
        # average over each of the N_vis sets of hidden biases
        b[int(N_vis + a1*N_vis):int(N_vis + (a1 + 1)*N_vis)] = params[nwparams + 1 + a1]
        count = 0
        for nv in range(N_vis):
            for a2 in range(alpha):
                # average over each of the M=N_vis*alpha sets of weights
                w[nv, int(N_vis + a2*N_vis):int(N_vis + (a2 + 1)*N_vis)] = params[count]
                w[int(N_vis + a2*N_vis):int(N_vis + (a2 + 1)*N_vis), nv] = params[count]
                count += 1
    return w, b

@numba.jit(nopython=True)
def params_to_wb_RBM(params, n_vis, n):
    w = weights_to_square_RBM(params[:-n], n_vis, n)
    b = params[-n:]
    return w, b

def calc_log_derivatives(w, b, n_vis):
    n = b.shape[0]
    n_hid = n - n_vis
    params = wb_to_params(w, b, n_vis)
    log_grad =  np.zeros((2**n_vis, n_vis*n_hid + n)) 
    for vid in range(2**n_vis):
        v = visstatelist[vid]
        
        # calculate hidden derivative factor (for {0,1} this is the cond prob p(h|v))
        if not spin:
            phv = sigmoid(np.dot(wblock.T, v) + b[n_vis:])
        else:
            phv = np.tanh(np.dot(wblock.T, v) + b[n_vis:])
        
        log_grad[vid, :n_vis*n_hid] = 0.5*np.outer(v, phv).flatten()
        log_grad[vid, n_vis*n_hid:n_vis*n_hid+n_vis] = 0.5*v
        log_grad[vid, n_vis*n_hid+n_vis:] = 0.5*phv

    return log_grad 

def grad_E_SR_hybrid(samples, E_loc, E, p_vis, w, b, visstatelist, n_vis, spin, natural, regname, epsilon_reg):

    n = np.shape(b)[0] # number of units/biases
    n_hid = n - n_vis # number of hidden units
    nwparams = n_vis*n_hid # number of weights
    
    # non-zero part of the weight matrix
    wblock = w[:n_vis, n_vis:]
    
    # prefactor of the correlations for gradient calculation
    prefactor = E_loc - E

    # gradient arrays
    e_grad = np.zeros((n_vis*n_hid + n_vis + n_hid))
    
    if natural:
        log_p_grad = np.zeros((2**n_vis, n_vis*n_hid + n_vis + n_hid))
        log_p_grad_mean = np.zeros(nwparams + n)
        # covariance matrix
        S = np.zeros((nwparams + n, nwparams + n))
     
        for si in range(samples.shape[0]):
            vis_state = int(jit_RBM.bin_to_dec(samples[si, :n_vis], n_vis))
            cur_sample = samples[si,:]
            pair_corr = np.outer(samples[si, :], samples[si, :])
            
            log_p_grad_mean[nwparams:] += cur_sample
            log_p_grad_mean[:nwparams] += pair_corr[:n_vis,n_vis:].flatten()

        log_p_grad_mean /= samples.shape[0]

    for vid in range(p_vis.shape[0]):
        v = visstatelist[vid]

        # calculate hidden derivative factor (for {0,1} this is the cond prob p(h|v))
        if not spin:
            phv = sigmoid(np.dot(wblock.T, v) + b[n_vis:])
        else:
            phv = np.tanh(np.dot(wblock.T, v) + b[n_vis:])
        
        # define weight and bias correlations
        b_corr = np.concatenate((v, phv))
        w_corr = np.outer(v, phv)

        # accumulate energy gradient terms
        e_grad[:nwparams] += prefactor[vid]*w_corr.flatten()*p_vis[vid]
        e_grad[nwparams:] += prefactor[vid]*b_corr*p_vis[vid]
       
        if natural:
            # calculate visible distribution gradients
            log_p_grad[vid, :nwparams] = (w_corr.flatten() - log_p_grad_mean[:nwparams]) - 2/2**n_vis
            log_p_grad[vid, nwparams:] = (b_corr - log_p_grad_mean[nwparams:]) - 2/2**n_vis
            
            # calculate covariance S
            S += 0.25*p_vis[vid]*np.dot(log_p_grad[vid,:].conj().T, log_p_grad[vid,:])

    kappa_S = None
    if natural:
        # calculate covariance matrix
        print(np.dot(p_vis, log_p_grad))
        assert (S == S.T).all()

        # calculate condition number of S
        u, svals, v = np.linalg.svd(S)
        kappa_S = svals[0]/svals[-1]

        # calculate pseudo inverse
        if regname == "pinv":
            Sinv = np.linalg.pinv(S, rcond=epsilon_reg)
        elif regname == "tikonov":
            S_reg = S + epsilon_reg*np.eye(S.shape[0])
            Sinv = np.linalg.inv(S_reg)
        elif regname == "carleo":
            S_reg = S + epsilon_reg*np.diag(S)*np.eye(np.shape(S)[0])
            Sinv = np.linalg.inv(S_reg)
        
        # calculate natural gradient
        e_grad = np.dot(Sinv, e_grad)

    # convert back to square weight and bias shape
    dw = weights_to_square_RBM(e_grad[:nwparams], n_vis, n)
    db = e_grad[nwparams:]
    return db, dw, kappa_S


def grad_E_ana_old(E_loc, E, p_vis, w, b, visstatelist, n_vis, spin, natural, regname, epsilon_reg, cutoff):

    n = np.shape(b)[0] # number of units/biases
    n_hid = n - n_vis # number of hidden units
    nwparams = n_vis*n_hid # number of weights
    
    # non-zero part of the weight matrix
    wblock = w[:n_vis, n_vis:]
    
    # prefactor of the correlations for gradient calculation
    prefactor = E_loc - E

    # gradient arrays
    e_grad = np.zeros((n_vis*n_hid + n_vis + n_hid))
    
    if natural:
        log_p_grad = np.zeros((2**n_vis, n_vis*n_hid + n_vis + n_hid))
        log_p_grad_mean = np.zeros(nwparams + n)
        # covariance matrix
        S = np.zeros((nwparams + n, nwparams + n))
     
        for vid in range(p_vis.shape[0]):
            v = visstatelist[vid]

            # calculate hidden derivative factor (for {0,1} this is the cond prob p(h|v))
            if not spin:
                phv = sigmoid(np.dot(wblock.T, v) + b[n_vis:])
            else:
                phv = np.tanh(np.dot(wblock.T, v) + b[n_vis:])
            
            # define weight and bias correlations
            b_corr = np.concatenate((v, phv))
            w_corr = np.outer(v, phv)
        
            log_p_grad_mean[nwparams:] += b_corr*p_vis[vid]
            log_p_grad_mean[:nwparams] += w_corr.flatten()*p_vis[vid]

    for vid in range(p_vis.shape[0]):
        v = visstatelist[vid]

        # calculate hidden derivative factor (for {0,1} this is the cond prob p(h|v))
        if not spin:
            phv = sigmoid(np.dot(wblock.T, v) + b[n_vis:])
        else:
            phv = np.tanh(np.dot(wblock.T, v) + b[n_vis:])
        
        # define weight and bias correlations
        b_corr = np.concatenate((v, phv))
        w_corr = np.outer(v, phv)

        # accumulate energy gradient terms
        e_grad[:nwparams] += prefactor[vid]*w_corr.flatten()*p_vis[vid]
        e_grad[nwparams:] += prefactor[vid]*b_corr*p_vis[vid]
       
        if natural:
            # calculate visible distribution gradients
            log_p_grad[vid, :nwparams] = (w_corr.flatten() - log_p_grad_mean[:nwparams])
            log_p_grad[vid, nwparams:] = (b_corr - log_p_grad_mean[nwparams:])
            
            # calculate covariance S
            #print("pvis ", p_vis)
            #print("logp_grad_mean = ", log_p_grad_mean)
            #print("wcorr ", w_corr.flatten())
            #print("bcorr ", b_corr)
            oconjo = np.outer(log_p_grad[vid,:], log_p_grad[vid,:])
            #print(oconjo)
            assert (oconjo.T == oconjo).all()
            S += 0.25*p_vis[vid]*oconjo

    kappa_S = None
    if natural:
        # calculate covariance matrix
        #print(np.dot(p_vis, log_p_grad))
        #print("covariance")
        #print(S)
        assert (S == S.T).all()

        # calculate condition number of S
        #u, svals, v = np.linalg.svd(S)
        #kappa_S = svals[0]/svals[-1]
        kappa_S = np.linalg.cond(S, p=-np.inf)
        # calculate pseudo inverse
        if regname == "pinv":
            Sinv = np.linalg.pinv(S, rcond=cutoff)
            e_grad = np.dot(Sinv, e_grad)
        elif regname == "carleo":
            S += epsilon_reg*np.diag(S)*np.eye(np.shape(S)[0])
            #Sinv = np.linalg.inv(S_reg)
            e_grad, residuals, rank, svals = np.linalg.lstsq(S, e_grad, rcond=cutoff)
        elif regname == "tikhonov":
            S += epsilon_reg*np.eye(S.shape[0])
            e_grad, residuals, rank, svals = np.linalg.lstsq(S, e_grad, rcond=cutoff)

    # convert back to square weight and bias shape
    #print("egrad = ",e_grad)
    dw = weights_to_square_RBM(e_grad[:nwparams], n_vis, n)
    db = e_grad[nwparams:]
    return db, dw, kappa_S

@numba.jit(nopython=True)
def grad_E_ana(E_loc, E, p_vis, w, b, visstatelist, n_vis, spin, natural, regname, epsilon_reg, cutoff):

    n = np.shape(b)[0] # number of units/biases
    n_hid = n - n_vis # number of hidden units
    nwparams = n_vis*n_hid # number of weights
    
    # non-zero part of the weight matrix
    wblock = w[:n_vis, n_vis:]
    wblock = np.ascontiguousarray(wblock)
    
    # prefactor of the correlations for gradient calculation
    prefactor = E_loc - E

    # gradient arrays
    e_grad = np.zeros((n_vis*n_hid + n_vis + n_hid))
    
    if natural:
        # helper arrays
        S_second_order = np.zeros((nwparams + n, nwparams + n))
        S_first_order = np.zeros(nwparams + n)
        # covariance matrix
        S = np.zeros((nwparams + n, nwparams + n))
     
    for vid in range(p_vis.shape[0]):
        v = visstatelist[vid]

        # calculate hidden derivative factor (for {0,1} this is the cond prob p(h|v))
        if not spin:
            phv = sigmoid(np.dot(wblock.T, v) + b[n_vis:])
        else:
            phv = np.tanh(np.dot(wblock.T, v) + b[n_vis:])
        
        # define weight and bias correlations
        b_corr = np.concatenate((v, phv))
        w_corr = np.outer(v, phv)
        param_corr = np.concatenate((w_corr.flatten(), b_corr))        

        # accumulate energy gradient terms
        e_grad[:] += prefactor[vid]*param_corr*p_vis[vid]

        if natural:
            # accumulate covariance terms
            S_first_order += p_vis[vid]*param_corr[:]
            S_second_order += p_vis[vid]*np.outer(param_corr, param_corr)
  
    kappa_S = None
    if natural:
        # calculate covariance matrix
        S = 0.25*(S_second_order - np.outer(S_first_order, S_first_order))
        #print(np.dot(p_vis, log_p_grad)) # == 0 ?
        assert (S == S.T).all()

        # calculate condition number of S
        kappa_S = np.linalg.cond(S, p=-np.inf)

        # calculate pseudo inverse
        """scale_inv_reg = False
        if scale_inv_reg:
            si_cutoff = 1.0e-10
            diag_S = np.diag(S)
            diag_S.setflags(write=1) 
            index = diag_S <= cutoff
            
            S[index, :].fill(0.0)
            S[:, index].fill(0.0)
            np.fill_diagonal(S, 1.0)
            diag_S[index] = 1.0
            S /= np.vdot(diag_S, diag_S)
            e_grad /= diag_S
        """
        if regname == "pinv":
            Sinv = np.linalg.pinv(S, rcond=cutoff)
            e_grad = np.dot(Sinv, e_grad)
        elif regname == "carleo":
            S += epsilon_reg*np.diag(S)*np.eye(np.shape(S)[0])
            e_grad, residuals, rank, svals = np.linalg.lstsq(S, e_grad, rcond=cutoff)
        elif regname == "tikhonov":
            S += epsilon_reg*np.eye(S.shape[0])
            e_grad, residuals, rank, svals = np.linalg.lstsq(S, e_grad, rcond=cutoff)

        """
        if scale_inv_reg:
            e_grad /= diag_S
        """
    # convert back to square weight and bias shape
    dw = weights_to_square_RBM(e_grad[:nwparams], n_vis, n)
    db = e_grad[nwparams:]
    return db, dw, kappa_S

def E_GD_2(E_loc, E, p_vis, w, b, visstatelist, n_vis, spin, natural, regname, epsilon_reg):

    n = np.shape(b)[0] # number of units/biases
    n_hid = n - n_vis # number of hidden units
    nwparams = n_vis*n_hid # number of weights
    
    # non-zero part of the weight matrix
    wblock = w[:n_vis, n_vis:]
    
    # gradient arrays
    grad = np.zeros(nwparams + n)
    
    # prefactor of the correlations for gradient calculation
    prefactor = E_loc - E

    # calculate gradient
    for vid in range(p_vis.shape[0]):
        v = visstatelist[vid]

        # calculate hidden derivative factor (for {0,1} this is the cond prob p(h|v))
        if not spin:
            phv = sigmoid(np.dot(wblock.T, v) + b[n_vis:])
        else:
            phv = np.tanh(np.dot(wblock.T, v) + b[n_vis:])
        
        # define weight and bias correlations
        b_corr = np.concatenate((v, phv))
        w_corr = np.outer(v, phv)
        # accumulate gradient terms
        grad[:nwparams] += prefactor[vid]*p_vis[vid]*(w_corr.flatten())
        grad[nwparams:] += prefactor[vid]*p_vis[vid]*(b_corr)

    # convert back to square weight and bias shape
    dw = weights_to_square_RBM(grad[:nwparams], n_vis, n)
    db = grad[nwparams:]
    return db, dw, 1

def E_SR_pvis_ana(E_loc, E, p_vis, w, b, visstatelist, n_vis, spin, regname, epsilon_reg):

    n = np.shape(b)[0] # number of units/biases
    n_hid = n - n_vis # number of hidden units
    nwparams = n_vis*n_hid # number of weights
    
    # non-zero part of the weight matrix
    wblock = w[:n_vis, n_vis:]
    
    # gradient arrays
    dw = np.zeros((n_vis, n_hid))
    db = np.zeros(n)
    # covariance matrix
    S = np.zeros((nwparams + n, nwparams + n))
    # first order expectations to used for calculation of S
    w_corrs_fo = np.zeros((n_vis, n_hid))
    b_corrs_fo = np.zeros(n)
 
    # prefactor of the correlations for gradient calculation
    prefactor = E_loc - E

    for vid in range(p_vis.shape[0]):
        v = visstatelist[vid]
        
        # calculate hidden derivative factor (for {0,1} this is the cond prob p(h|v))
        if not spin:
            phv = sigmoid(np.dot(wblock.T, v) + b[n_vis:])
        else:
            phv = np.tanh(np.dot(wblock.T, v) + b[n_vis:])
        #print("biases = {}, p(h|v) = {}".format(b[n_vis:], phv))
        
        # define weight and bias correlations
        b_corr = np.concatenate((v, phv))
        w_corr = np.outer(v, phv)

        # accumulate gradient terms
        dw[:,:] += prefactor[vid]*p_vis[vid]*w_corr
        db[:] += prefactor[vid]*p_vis[vid]*b_corr

        # accumulate first order expectations
        w_corrs_fo[:, :] += w_corr*p_vis[vid]
        b_corrs_fo[:] += b_corr*p_vis[vid]
        
        # flatten the current weight correlations for easy calc of covariance matrix
        w_corr_flat = w_corr.flatten()
        assert nwparams == np.shape(w_corr_flat)[0]

        for m in range(nwparams):
            # calculate covariance for weights 
            for l in range(m + 1):
                S[m, l] += p_vis[vid]*w_corr_flat[m]*w_corr_flat[l] 
        
            # calculate covariance between weights and biases
            for o in range(n):
                S[m, o + nwparams] += p_vis[vid]*w_corr_flat[m]*b_corr[o]

                # calculate covaraince between biases
                for p in range(o + 1):
                    S[p + nwparams, o + nwparams] += p_vis[vid]*b_corr[p]*b_corr[o]

    # flatten w_corrs_fo
    w_corrs_fo = w_corrs_fo.flatten() 

    # subtract first-order expectations
    for m in range(nwparams):

        # calculate covariance for weights 
        for l in range(m + 1):
            S[m, l] -= w_corrs_fo[m]*w_corrs_fo[l]
    
        # calculate covariance between weights and biases
        for o in range(n):
            S[m, o + nwparams] -= w_corrs_fo[m]*b_corrs_fo[o]

            # calculate covaraince between biases
            for p in range(o + 1):
                S[p + nwparams, o + nwparams] -= b_corrs_fo[p]*b_corrs_fo[o]
    
    # so far we have built half of the matrix
    assert (S[nwparams:,:nwparams] == 0).all()
    assert (S[:nwparams,nwparams:] != 0).any()

    # complement elements on the other side of the diagonal
    S = S + S.T - np.diag(S)*np.eye(S.shape[0])
    assert (S == S.T).all()

    # calculate condition number of S
    u, svals, v = np.linalg.svd(S)
    kappa_S = svals[0]/svals[-1]

    # calculate pseudo inverse
    if regname == "pinv":
        Sinv = np.linalg.pinv(S, rcond=epsilon_reg)
    elif regname == "tikonov":
        S_reg = S + epsilon_reg*np.eye(S.shape[0])
        Sinv = np.linalg.inv(S_reg)
    elif regname == "carleo":
        S_reg = S + epsilon_reg*np.diag(S)*np.eye(np.shape(S)[0])
        Sinv = np.linalg.inv(S_reg)
    
    # flattened param gradient
    dparams = np.concatenate((dw.flatten(), db))

    # calculate natural gradient
    nat_grad_params = np.dot(Sinv, dparams)

    # convert back to square weight and bias shape
    dw = weights_to_square_RBM(nat_grad_params[:nwparams], n_vis, n)
    db = nat_grad_params[nwparams:]
    return db, dw, kappa_S

@numba.jit(nopython=True)
def E_gradient_pvis_ana(E_loc, E, p_vis, w, b, visstatelist, n_vis, spin, natural, regname, epsilon_reg):
    n = w.shape[0]
    n_hid = n - n_vis
    wblock = w[:n_vis,n_vis:]
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    # this prefactor depends on the loss which here is the variational energy E
    prefactor = E_loc - E

    if natural:
        # define covariance matrix
        nwparams = n_vis*n_hid #n*(n - 1)//2# + n
        S = np.zeros((nwparams + n, nwparams + n))

    # iterate through all visible configurations
    for vid in range(p_vis.shape[0]):
        v = visstatelist[vid]
        
        # calculate conditionl probability p(h|v)
        if not spin:
            phv = sigmoid(np.dot(wblock.T, v) + b[n_vis:])
        else:
            phv = np.tanh(np.dot(wblock.T, v) + b[n_vis:])
        
        # define weight and bias correlations
        b_corr = np.concatenate((v, phv))
        w_corr = np.outer(b_corr, b_corr)
        
        # constrain wcorr to RBM topology
        w_corr[n_vis:, n_vis:] = np.zeros((n_hid, n_hid))
        w_corr[:n_vis, :n_vis] = np.zeros((n_vis, n_vis))
        np.fill_diagonal(w_corr, 0.)


        # accumulate gradient terms
        #dw[:n_vis,n_vis:] += prefactor[vid]*p_vis[vid]*w_corr[:n_vis,n_vis:]
        dw[:,:] += prefactor[vid]*p_vis[vid]*w_corr[:,:]
        db[:] += prefactor[vid]*p_vis[vid]*b_corr

        if natural:
            # flatten the current weight correlations for easy access for covariance matrix
            w_corr_flat = weights_to_flat_RBM(w_corr, n_vis)#weights_to_flat(flat_inds, w_corr)
            for m in range(nwparams):
                # calculate covariance for weights 
                for l in range(m + 1):
                    S[m, l] += p_vis[vid]*w_corr_flat[m]*w_corr_flat[l] 
            
                # calculate covariance between weights and biases
                for o in range(n):
                    S[m, o + nwparams] += p_vis[vid]*w_corr_flat[m]*b_corr[o]

                    # calculate covaraince between biases
                    for p in range(o + 1):
                        S[p + nwparams, o + nwparams] += p_vis[vid]*b_corr[p]*b_corr[o]

    # fill the other half of the weight matrix
    #dw[n_vis:,:n_vis] = dw[:n_vis,n_vis:].T

    if natural:
        assert (S[nwparams:,:nwparams] == 0).all()
        assert (S[:nwparams,nwparams:] != 0).any()

        # complement elements on the other side of the diagonal
        S = S + S.T - np.diag(S)*np.eye(S.shape[0])
        print(S)
        assert (S == S.T).all()

        u, svals, v = np.linalg.svd(S)
        kappa_S = svals[0]/svals[-1]
        kappa_reg = 1
        # calculate pseudo inverse
        if regname == "pinv":
            Sinv = np.linalg.pinv(S, rcond=epsilon_reg)
        elif regname == "tikonov":
            S_reg = S + epsilon_reg*np.eye(S.shape[0])
            u, svalsreg, v = np.linalg.svd(S_reg)
            #kappa_reg = svalsreg[0]/svalsreg[-1]
#            print("kappa S = ",kappa_S, " kappa S_tikh = ", kappa_reg)
            Sinv = np.linalg.inv(S_reg)
        elif regname == "carleo":
            S_reg = S + epsilon_reg*np.diag(S)
            u, svalsreg, v = np.linalg.svd(S_reg)
            kappa_reg = svalsreg[0]/svalsreg[-1]
#            print("kappa S = ", kappa_S," kappa S_carleo = ",kappa_reg)
            Sinv = np.linalg.inv(S_reg)
        elif regname == "test":
            Sinv = np.dot(np.linalg.inv(np.dot(S.T, S) + epsilon_reg*np.eye(S.shape[0])), S.T)

        # flattened param gradient
 
        dw_flat = weights_to_flat_RBM(dw, n_vis)#weights_to_flat(flat_inds, dw)
        dparams = np.concatenate((dw_flat, db))

        # calculate natural gradient
        nat_grad_params = np.dot(Sinv, dparams)
        #print("diff grad/nat grad = ",np.linalg.norm((nat_grad_params - dparams)**2)) 

        # convert back to weight and bias shape
        dw = weights_to_square_RBM(nat_grad_params[:nwparams], n_vis, n)#weights_to_square(n, flat_inds, nat_grad_params[:nwparams])
        db = nat_grad_params[nwparams:]
        return db, dw, kappa_reg/kappa_S

    return db, dw, None

@numba.jit(nopython=True)
def grad_E_sampled(samples, n, n_vis, p_train, E_loc, E, spin, natural, regname, epsilon_reg, cutoff):
    n_hid = n - n_vis
    nwparams = n_vis*n_hid
    nsamples = np.shape(samples)[0]
    
    e_grad = np.zeros(nwparams + n)
    
    S_first_order = np.zeros(nwparams + n)
    S_second_order = np.zeros((nwparams + n, nwparams + n))
    S = np.zeros((nwparams + n, nwparams + n))

    prefactor = E_loc - E
    
    for si in range(nsamples):
        
        cur_sample = samples[si,:]
        #vis_state = int(jit_RBM.bin_to_dec(samples[si, :n_vis], n_vis))
        vis_state = int(jit_RBM.bin_to_dec(cur_sample[:n_vis], n_vis)) if not spin else int(jit_RBM.bin_to_dec(jit_RBM.spin_to_bin(cur_sample[:n_vis]), n_vis))
        pair_corr = np.outer(samples[si, :n_vis], samples[si, n_vis:])
        params_corr = np.concatenate((pair_corr.flatten(), cur_sample))
        
        # accumulate energy gradient
        e_grad[:] += prefactor[vis_state]*params_corr

        if natural:
            # accumulate covariance terms
            S_first_order[:] += params_corr
            S_second_order[:,:] += np.outer(params_corr, params_corr)

    # normalize gradient
    e_grad /= nsamples

    kappa_S = None
    if natural:
        # normalize covariance terms
        #S_second_order /= nsamples
        #S_first_order /= nsamples
        # calculate covariance matrix
        #S = 0.25*(S_second_order - np.outer(S_first_order, S_first_order))
        S = 0.25*(S_second_order*nsamples - np.outer(S_first_order, S_first_order))/nsamples**2
        #print(np.dot(p_vis, log_p_grad)) # == 0 ?
        assert (S == S.T).all()

        # calculate condition number of S
        kappa_S = np.linalg.cond(S, p=-np.inf)
        # calculate pseudo inverse
        if regname == "pinv":
            Sinv = np.linalg.pinv(S, rcond=cutoff)
            e_grad = np.dot(Sinv, e_grad)
        elif regname == "carleo":
            S += epsilon_reg*np.diag(S)*np.eye(np.shape(S)[0])
            e_grad, residuals, rank, svals = np.linalg.lstsq(S, e_grad, rcond=cutoff)
        elif regname == "tikhonov":
            S += epsilon_reg*np.eye(S.shape[0])
            e_grad, residuals, rank, svals = np.linalg.lstsq(S, e_grad, rcond=cutoff)

    # convert back to square weight and bias shape
    dw = weights_to_square_RBM(e_grad[:nwparams], n_vis, n)
    db = e_grad[nwparams:]

    return db, dw, kappa_S


@numba.jit(nopython=True)
def E_gradient_pvis_sampled(samples, n, n_vis, p_train, E_loc, E):
    nsamples = np.shape(samples)[0]
    db = np.zeros(n)
    dw = np.zeros((n, n))

    prefactor = E_loc - E
 
    for si in range(nsamples):
        # accumulate correlations for gradient estimation and covariance estimation
        vis_state = int(jit_RBM.bin_to_dec(samples[si, :n_vis], n_vis))
        cur_sample = samples[si,:]
        pair_corr = np.outer(samples[si, :], samples[si, :])
        db[:] += prefactor[vis_state] * cur_sample
        dw[:,:] += prefactor[vis_state] * pair_corr
    return db/nsamples, dw/nsamples

def calc_energy_model(p_model, H):
    H_sqp = H.dot(np.sqrt(p_model))
    return np.dot(np.sqrt(p_model), H_sqp)

@numba.jit(nopython=True)
def calc_dkl2(n_vis, p_target, p_train):
    dkl = 0.
    for vis_state in range(2**n_vis):
        if p_train[vis_state] != 0 and p_target[vis_state] != 0:
            dkl += p_target[vis_state] * np.log(p_target[vis_state] / p_train[vis_state])
    return dkl

@numba.jit(nopython=True)
def calc_dkl(n_vis, p_target, p_train):
    dkl = 0.
    p_train += 1e-8
    p_train /= p_train.sum()
    for vis_state in range(2**n_vis):
        if p_train[vis_state] != 0 and p_target[vis_state] != 0:
            dkl += p_target[vis_state] * np.log(p_target[vis_state] / p_train[vis_state])
    return dkl


def init_wb(n, n_vis, w_max=+1, w_min=-1, b_max=+1, b_min=-1, spin=False, distr="standard", translate=True):
    # randomly initialize weights
    if distr == "standard":
        J = w_max*(2*np.random.random(size = (n, n)) - 1)
        J = 0.5 * (J + J.T)
        J[:n_vis,:n_vis] = 0.0
        J[n_vis:,n_vis:] = 0.0
        h = b_max*(2*np.random.random(size=n) - 1)
    elif distr == "uniform":
        J = w_max*(2*np.random.random(size = (n, n)) - 1)
        J[:n_vis,:n_vis] = 0.0
        J[n_vis:,n_vis:] = 0.0
        J[n_vis:,:n_vis] = J[:n_vis,n_vis:].T
        h = b_max*(2*np.random.random(size=n) - 1)
    elif distr == "gauss":
        J = w_max*np.random.normal(size = (n, n)) 
        J[:n_vis,:n_vis] = 0.0
        J[n_vis:,n_vis:] = 0.0
        J[n_vis:,:n_vis] = J[:n_vis,n_vis:].T
        h = b_max*np.random.normal(size=n)
    assert (J == J.T).all()

    #vis_states = jit_RBM.gen_states(n_vis, spin=True)
    #p_vis_spin_sampled, samples, sample_vinds = jit_RBM.gibbs_sampling(J, h, vis_states, num_chains=4, num_samples=10000, pvis=None, burn_in=0, spin=True)
    #p_vis_spin = jit_RBM.calc_pvis_ana(J, h, vis_states, spin=True)


    if spin:
        return J.astype(np.float), h.astype(np.float)
    
    # transfer "spin glass interactions" to RBM weights
    if translate:
        W = 4*J
        b = 2*h - 2*J.sum(axis=1) - 0.5*J.sum() + h.sum()
    else:
        W = J
        b = h
    #vis_states = jit_RBM.gen_states(n_vis, spin=False)
    #p_vis_bin_sampled, samples, sample_vinds = jit_RBM.gibbs_sampling(W, b, vis_states, num_chains=4, num_samples=10000, pvis=None, burn_in=0, spin=False)
    #p_vis_bin = jit_RBM.calc_pvis_ana(W, b, vis_states, spin=False)  
    #plt.bar(np.arange(p_vis_bin.shape[0]) - 0.25, p_vis_bin, width=0.25)
    #plt.bar(np.arange(p_vis_spin.shape[0]) + 0.25, p_vis_spin, width=0.25)
    #plt.show()
    #sys.exit()
    return W.astype(np.float), b.astype(np.float)

# search for the groundstate of Hamiltonian H by minimizing the energy given settings S
def learn_groundstate(H, S):

    # extract some settings
    N_vis = S.nvis
    N_hid = S.nhid
    winit = S.winit
    wmin = S.wbounds[0]
    wmax = S.wbounds[1]
    binit = S.binit
    bmin = S.bbounds[0]
    bmax = S.bbounds[1]
    spin = S.spin
    epochs = S.epochs
    p_target = S.ED_distribution
    e_target = S.ED_energy

    # check consistent N_vis
    if not p_target is None:
        assert N_vis == int(np.log2(p_target.shape[0]))
    
    N = N_vis + N_hid
    
    # generate visible configuations (not scalable)
    vis_states = jit_RBM.gen_states(N_vis, spin=spin)
    
    # create desired topology
    if S.topology.upper() == "RBM":
        nwparams = N_vis*N_hid
        nbparams = N
        nparams = nwparams + nbparams
        w, b = init_wb(N, N_vis, w_max=winit, w_min=-winit, b_max=binit, b_min=-binit, spin=spin, translate=S.translate_wb)
        w = jit_RBM.mask_weights_rbm(w, N_vis, N_hid) 
    else:
        raise NotImplemented("this topology is not implemented.")

    if len(S.symmetries) > 0 and S.topology == "RBM":
        if S.model == "TFIM":
            if len(S.symmetries) == 1 and "translation" in S.symmetries: 
                # translation invariance
                alpha = int(N_hid/N_vis)
                assert N_hid % N_vis == 0
                nwparams = N_hid
                nbparams = 1 + alpha
                nparams = nwparams + nbparams

                btemp = b.copy()
                wtemp = w.copy()
                
                # the visible biases have only one free parameter
                b[:N_vis] = btemp[:N_vis].mean()
                
                # the hidden biases have M/N free parameters, the weights have M free parameters
                for j in range(N_hid):
                    b[N_vis + j] = btemp[N_vis + j // N_vis]
                    w[:N_vis, N_vis + j] = wtemp[:N_vis, N_vis + j // N_vis]
                    w[N_vis + j, :N_vis] = wtemp[N_vis + j // N_vis, :N_vis]
                
                #print(np.unique(b).shape[0])
                #print(np.unique(w).shape[0]-1)
                print("Building in translation invariance symmetries")
    
    if S.optim == "GD" or S.optim == "ADAM":
        lr = S.lr
        eps_reg = S.eps_reg
        if S.optim == "ADAM":
            beta1 = float(S.optimparams[0])
            beta2 = float(S.optimparams[1])
            eps_adam = float(S.optimparams[2])
            mw = np.zeros(w.shape)
            mb = np.zeros(b.shape)
            vw = np.zeros(w.shape)
            vb = np.zeros(b.shape)

        delta_E = np.ones(epochs)*1e-12
        max_delta_E = 1e-12

        b_epoch = np.zeros((epochs, N))
        w_epoch = np.zeros((epochs, N, N))   
        p_epoch = np.zeros((epochs, 2**N_vis))
        dkl_epoch = np.zeros(epochs)
        subdkl_epoch = np.zeros((epochs, S.nsubsamples)) if S.subsampling else None
        energy_epoch = np.zeros(epochs)
        subenergy_epoch = np.zeros((epochs, S.nsubsamples)) if S.subsampling else None
        lr_epoch = np.zeros(epochs); lr_epoch[0] = lr
        condratio_epoch = np.zeros(epochs)
        
        starttime = time.time()
        #print("p_target: ",p_target)

        for e in range(epochs):
            # variable for when sampling is used
            samples = None

            # store weights and biases
            w_epoch[e, :, :] = w[:, :]
            b_epoch[e, :] = b[:]
            
            if S.lr_schedule == "exp":
                # decay step size and regularization parameter
                lr = max(S.lr*S.lr_decay**e, S.lr_min)
                #lr = max(start_lr*np.exp(-0.0001*e), 0.0001)
            elif S.lr_schedule == "adaptive1":
                max_delta_E = max(np.abs(delta_E[:e + 1]))
                lr = max(np.sum(S.lr_decay**np.arange(e + 1)[::-1]*lr_epoch[:e + 1]*np.abs(delta_E[:e + 1])/max_delta_E), S.lr_min)
            elif S.lr_schedule == "adaptive2":
                max_delta_E = max(delta_E[:e + 1])
                #print("max dE: ",max_delta_E)
                #print("Relu dE:", np.maximum(delta_E[:e+1], 0.0))
                lr = max(np.sum(S.lr_decay**np.arange(e + 1)[::-1]*lr_epoch[:e + 1]*np.maximum(delta_E[:e + 1], 0.0)/max_delta_E), S.lr_min)
                #print("lr: ",lr)
            lr_epoch[e] = lr

            if S.natgrad and S.eps_reg > 0 and S.eps_reg_min > 0 and S.eps_reg_decay > 0:
                eps_reg = max(S.eps_reg*S.eps_reg_decay**e, S.eps_reg_min)

            
            # generate samples if necessary
            if S.sample_pvis or S.sample_grad:
                p_vis, samples, sample_vinds = jit_RBM.gibbs_sampling(w[:,:], b[:], vis_states, S.nchains, S.nsamples, pvis=p_epoch[e - 1, :] if e > 0 else None, burn_in=S.burnin, spin=spin, pdying=S.p_dying)

                if S.dyingneurons:
                    corrupt_samples = np.zeros_like(samples)
                    for ni in range(N):
                        if np.random.random() <= S.p_dying:
                            corrupt_samples[:,ni] = np.zeros(S.nsamples*S.nchains)
                        else:
                            corrupt_samples[:,ni] = samples[:,ni]
                    samples = np.copy(corrupt_samples)
                    p_vis = jit_RBM.calc_pvis(samples, N_vis, N_hid, spin=S.spin)
            
            # subsampling
            if S.subsampling:
                subsamples = np.random.choice(sample_vinds, size=(S.nsubsamples, S.subsamplesize))
                subenergies = []
                subdkls = []
                for subsample in subsamples:
                    # calc pvis
                    subpvis = jit_RBM.calc_pvis_vinds(subsample, N_vis)
                    # calc DKL
                    if not p_target is None:
                        subdkl = calc_dkl(N_vis, p_target, subpvis)
                        subdkls.append(subdkl)
                     # calc energy
                    subenergy = calc_energy_model(subpvis, H)
                    subenergies.append(subenergy)
                subenergy_epoch[e, :] = np.array(subenergies)
                if len(subdkls) > 0:
                    subdkl_epoch[e, :] = np.array(subdkls)
                   
            # calculate visible distribution
            if not S.sample_pvis:
                p_vis = jit_RBM.calc_pvis_ana(w[:,:], b[:], vis_states, spin)

            if S.renormalize:
                p_vis = p_vis**2/np.sum(p_vis**2)
            
            # store visible distribution
            p_epoch[e, :] = p_vis[:]
            #print("p: ", p_vis)

            # calc DKL
            if not p_target is None:
                dkl_epoch[e] = calc_dkl(N_vis, p_target, p_vis)#p_target @ np.log(p_target/p_vis)
            # calc energy
            energy_epoch[e] = calc_energy_model(p_vis, H)
            delta_E[e] = energy_epoch[e] - (energy_epoch[e-1] if e > 0 else 0)
            #delta_E[e] = (np.maximum(energy_epoch[e-1] - energy_epoch[e], 0.0)) if e > 0 else np.abs(energy_epoch[e])
            # calc local energy
            sqrt_pvprime_pv = np.sqrt(np.outer(p_epoch[e, :], 1/(p_epoch[e, :] + S.eps_loc_energy)))
            E_loc = np.diag(H.dot(sqrt_pvprime_pv))

            #E_loc = np.zeros(2**N_vis)
            #H_psiket = H.dot(np.sqrt(p_epoch[e,:]))
            #for vi in range(2**N_vis):
            #    E_loc[vi] = H_psiket[vi]/np.sqrt(p_epoch[e,vi])

            #print(E_loc)
            #print(E_loc2)
            #print(E_loc.shape)
            #print(E_loc2.shape)
            #assert (E_loc == E_loc2).all()
            
            # calc gradients
            dw = np.zeros((N, N))
            db = np.zeros(N)
           
            if S.sample_grad:
                db_sampled, dw_sampled, condratio = grad_E_sampled(samples, N, N_vis, p_vis, E_loc, energy_epoch[e], spin, S.natgrad, S.regname, eps_reg, S.svd_cutoff)
                db = db_sampled; dw = dw_sampled
            else:
                db, dw, condratio = grad_E_ana(E_loc, energy_epoch[e], p_vis, w, b, vis_states, N_vis, spin, S.natgrad, regname=S.regname, epsilon_reg=eps_reg, cutoff=S.svd_cutoff)
                    
                if not condratio is None:
                    condratio_epoch[e] = condratio 
                
            if dw.any() != dw.T.any():
                raise ValueError("Weight update is not symmetric!")
            
            # enforce desired topology of update
            if S.topology.upper() == "RBM":
                dw = jit_RBM.mask_weights_rbm(dw, N_vis, N_hid)
            else:
                raise NotImplemented("this topology is not implemented.")

            if S.symmetries:
                if S.model == "TFIM" and S.topology == "RBM":
                    # translation invariance by taking mean over gradients
                    dbtemp = db.copy()
                    dwtemp = dw.copy()

                    # average gradient across visible biases
                    db[:N_vis] = dbtemp[:N_vis].mean()

                    for a1 in range(alpha):
                        # average over each of the N_vis sets of hidden biases
                        db[int(N_vis + a1*N_vis):int(N_vis + (a1 + 1)*N_vis)] = dbtemp[int(N_vis + a1*N_vis):int(N_vis + (a1 + 1)*N_vis)].mean()
                    for nv in range(N_vis):
                        for a2 in range(alpha):
                            # average over each of the M=N_vis*alpha sets of weights
                            dwavg = dwtemp[nv, int(N_vis + a2*N_vis):int(N_vis + (a2 + 1)*N_vis)].mean()
                            dw[nv, int(N_vis + a2*N_vis):int(N_vis + (a2 + 1)*N_vis)] = dwavg                  
                            dw[int(N_vis + a2*N_vis):int(N_vis + (a2 + 1)*N_vis), nv] = dwavg
                    #pass
                    #print(dw)
                    #print(db)
                    #print(np.unique(db).shape[0])
                    #print(np.unique(dw).shape[0]-1)
           
            if S.optim == "ADAM":
                # update weights using some optimization algorithm
                # ADAM
                mw = beta1*mw + (1. - beta1)*dw
                mb = beta1*mb + (1. - beta1)*db

                vw = beta2*vw + (1. - beta2)*dw**2
                vb = beta2*vb + (1. - beta2)*db**2

                mhw = mw/(1. - beta1**(e + 1))
                vhw = vw/(1. - beta2**(e + 1))
                #print("vw: ",np.max(vw),np.min(vw))
                #print("mw: ",np.max(mw),np.min(mw))
                #print("vb: ",np.max(vb),np.min(vb))
                #print("mb: ",np.max(mb),np.min(mb))
                #wnew = w - lr*mhw/(np.sqrt(vhw) + eps_adam)
                wnew = w - lr*np.divide(mhw, np.sqrt(vhw) + eps_adam)
                
                mhb = mb/(1. - beta1**(e + 1))
                vhb = vb/(1. - beta2**(e + 1))
                #bnew = b - lr*mhb/(np.sqrt(vhb) + eps_adam)
                bnew = b - lr*np.divide(mhb, np.sqrt(vhb) + eps_adam)
            
            elif S.optim == "GD":
                # gradient descent
                wnew = w - lr*dw
                bnew = b - lr*db

            # apply param constraints
            # IDEA: if param would be clipped, replace by random init
            wnew = np.clip(wnew, wmin, wmax)
            bnew = np.clip(bnew, bmin, bmax)
                
            # carry over new weights and biases to next iteration
            w[:,:] = wnew
            b[:] = bnew
                
            # logging some error measure 
            if S.logfreq > 0 and e % S.logfreq == 0:
                print("epoch{} / ee {:.4f} / kl-div {:.5f} / lr {:.3f} / reg {:.6f} / time {:.2f}".format(e, np.abs(energy_epoch[e] - e_target), dkl_epoch[e], lr, eps_reg, time.time() - starttime))
                starttime = time.time()
        
        if not p_target is None:
            fid = np.sqrt(p_epoch) @ np.sqrt(p_target)
        else:
            fid = np.zeros(epochs)
        data_dict = {"w": w_epoch, "b": b_epoch, "energy": energy_epoch, "dkl": dkl_epoch, "p": p_epoch, "fid": fid, "lr": lr_epoch, "condratio": condratio_epoch, "subdkl": subdkl_epoch, "subenergy": subenergy_epoch}
        return data_dict

    elif S.optim == "DE":

        import DE
        mutation = float(S.optimparams[0])
        crossover = float(S.optimparams[1])
        npop_per_dim = int(S.optimparams[2])
        
        if np.isinf([wmin, wmax, bmin, bmax]).any():
            wb_bounds = None
            init_ratios = np.array([winit]*nwparams+[binit]*nbparams)
        else:
            wb_bounds = [(wmin, wmax)]*nwparams+[(bmin, bmax)]*nbparams
            init_ratios = np.array([winit/(wmax - wmin)]*nwparams+[binit/(bmax - bmin)]*nbparams)

        def eval_func(params):
            if S.symmetries == ["translation"]:
                w, b = params_to_wb_transinv_RBM(params, N_vis, N)
            else:
                w, b = params_to_wb_RBM(params, N_vis, N)
            if S.sample_pvis:
                p_vis, samples, sample_vinds = jit_RBM.gibbs_sampling(w[:,:], b[:], vis_states, S.nchains, S.nsamples, pvis=None, burn_in=S.burnin, spin=spin)
            else:
                p_vis = jit_RBM.calc_pvis_ana(w[:,:], b[:], vis_states, spin)

            E = calc_energy_model(p_vis, H)
            return E, p_vis

        xbest_it, Fbest_it, mean_dist_it, fevals_it, pvis_it = DE.DE(eval_func, nparams, dim_F=1, init_ratio=init_ratios, bounds=wb_bounds, iterations=epochs, max_function_calls=np.inf,  npop_per_dim=npop_per_dim, crossover=crossover, mutation=mutation, best_pivot=True, logfreq=100)
        
        p_it = np.array(pvis_it)
        if not p_target is None:
            dkl_it = np.array([calc_dkl(N_vis, p_target, pvis_i) for pvis_i in p_it])
            fid_it = np.sqrt(p_it) @ np.sqrt(p_target)
        p_it = np.zeros(epochs)
        w_it = np.zeros((epochs, N, N))
        b_it = np.zeros((epochs, N))
        for it, params in enumerate(xbest_it):
            if S.symmetries == ["translation"]:
                w, b = params_to_wb_transinv_RBM(params, N_vis, N)
            else:
                w, b = params_to_wb_RBM(params, N_vis, N)
            w_it[it] = w; b_it[it] = b
        data_dict = {"energy": Fbest_it, "dkl": dkl_it, "p": p_it, "fid": fid_it, "fevals": fevals_it, "mean_dist": mean_dist_it, "w": w_it, "b": b_it}

        return data_dict

def landscape_analysis(H, S, w, b):
    
    vis_states = jit_RBM.gen_states(S.nvis, spin=S.spin)

    N = b.shape[0]
    max_step = S.perturb_size
    num_steps = S.perturb_steps
    
    wnorm = np.linalg.norm(w)
    bnorm = np.linalg.norm(b)
    wr1, br1 = init_wb(N, N_vis, w_max=S.winit, w_min=-S.winit, b_max=S.binit, b_min=-S.binit, spin=S.spin)
    wr2, br2 = init_wb(N, N_vis, w_max=S.winit, w_min=-S.winit, b_max=S.binit, b_min=-S.binit, spin=S.spin)

    wr1 = wnorm/np.linalg.norm(wr1)
    wr2 = wnorm/np.linalg.norm(wr2)
    br1 = bnorm/np.linalg.norm(br1)
    br2 = bnorm/np.linalg.norm(br2)
    
    # sample origin distribution
    if S.sample_pvis:
        pwb, samples, sample_vinds = jit_RBM.gibbs_sampling(w[:,:], b[:], vis_states, S.nchains, S.nsamples, pvis=None, burn_in=S.burnin, spin=S.spin)
    else:
        pwb = jit_RBM.calc_pvis_ana(w[:,:], b[:], vis_states, S.spin)


    landscape = np.zeros((num_steps, num_steps))
    for ix, dx in enumerate(np.linspace(-max_step, max_step, num_steps)):
        for iy, dy in enumerate(np.linspace(-max_step, max_step, num_steps)):
            wprobe = w + wr1*dx + wr2*dy
            bprobe = b + br1*dx + br2*dy
            # calc model visible distribution
            # generate samples if necessary
            if S.sample_pvis:
                p_vis, samples, sample_vinds = jit_RBM.gibbs_sampling(wprobe[:,:], bprobe[:], vis_states, S.nchains, S.nsamples, pvis=pwb, burn_in=S.burnin, spin=S.spin)
            else:
                p_vis = jit_RBM.calc_pvis_ana(wprobe[:,:], bprobe[:], vis_states, S.spin)

            landscape[ix, iy] = calc_energy_model(p_vis, H)

    return landscape

def save_data(settings, array_dict, outpath):
    # save settings json
    with open("{}/run_{}_settings.json".format(outpath, settings.id),"w") as f:
        json.dump(vars(settings), f)
    
    # save data
    for (arname, array) in array_dict.items():
        np.save("{}/run_{}_{}.npy".format(outpath, settings.id, arname), array)

def plotting(settings, array_dict, outpath):
    
    # retrieve data to be plotted
    e_target = settings.ED_energy 
    p_target = settings.ED_distribution
    
    # define metrics
    energy = array_dict["energy"]
    dkl = np.nan_to_num(array_dict["dkl"])
    fid = array_dict["fid"]
    
    # define energy errors
    energy_error = np.abs(energy- e_target)
    energy_error_per_site = energy_error/settings.nvis
    rel_energy_error = np.abs(energy- e_target)/np.abs(e_target)
    rel_energy_error_per_site = rel_energy_error/settings.nvis

    # define infidelity
    infidelity = 1. - fid 

    # subsampling
    if settings.subsampling:
        subdkls_e = array_dict["subdkl"]
        subenergies_e = array_dict["subenergy"]
        stddkl = subdkls_e.std(axis=1)
        meandkl = subdkls_e.mean(axis=1)
        stdenergy = subenergies_e.std(axis=1)
        meanenergyerror = np.abs(subenergies_e.mean(axis=1) - e_target)
        
    # DKL and absolute energy difference vs epochs
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2 if p_target is None else 3, 1, 1)
    ax1.plot(energy_error)
    if settings.subsampling:
        ax1.errorbar(range(len(meanenergyerror)), meanenergyerror, yerr=stdenergy, label="subsampled", fmt="xr")
    ax1.set_ylabel("$|E - E_0|$")
    ax1.set_yscale("log")
    ax1.set_yticks(10**(np.arange(-10, 2, 1, dtype=float)))
    ax1.set_ylim([min(energy_error), max(energy_error)])
    ax1.grid(which="both")
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    if p_target is not None:
        ax3 = fig.add_subplot(312)
        ax3.plot(dkl)
        if settings.subsampling:
            ax3.errorbar(range(len(meandkl)), meandkl, yerr=stddkl, label="subsampled", fmt="xr")
        ax3.set_yscale("log")
        ax3.set_yticks(10**(np.arange(-10, 10, 1, dtype=float)))
        ax3.set_ylim([min(dkl), max(dkl)])
        ax3.set_ylabel("$D_{KL}$")
        ax3.grid(which="both")
        ax4 = fig.add_subplot(313)
        ax4.plot(infidelity)
        ax4.set_yscale("log")
        ax4.set_ylabel("Infidelity $F$")
        ax4.set_yticks(10**(np.arange(-10, 10, 1, dtype=float)))
        ax4.set_ylim([np.min(infidelity), np.max(infidelity)])
        ax4.set_xlabel("epoch")
        ax4.grid(which="both")

    if settings.record:
        plt.savefig("{}/run_{}_dkl_ediff.pdf".format(outpath, settings.id), bbox_inches='tight')
    
    if settings.optim == "GD" or settings.optim == "ADAM":
        lr = array_dict["lr"]
    else:
        lr = np.zeros(settings.epochs)
    w = array_dict["w"]
    b = array_dict["b"]
    
    fig2 = plt.figure(figsize=(9,3)) #set_size(subplots=(3,1)))
    ax1 = fig2.add_subplot(311)

    ax1.plot(w.reshape(-1, w.shape[1]*w.shape[2]))
    ax1.set_ylabel("weights")
    ax1.grid()
    ax2 = fig2.add_subplot(312)
    ax2.plot(b)
    ax2.grid()
    ax2.set_ylabel("biases")
    ax3 = fig2.add_subplot(313)
    ax3.plot(lr)
    ax3.grid()
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("learning rate")

    if settings.record:
        plt.savefig("{}/run_{}_dwdb.pdf".format(outpath, settings.id), bbox_inches='tight')

    if settings.landscape:
        fig3 = plt.figure()
        ax1 = fig3.add_subplot(111)
        landscape = np.abs(data_dict["landscape"] - settings.ED_energy)
        implot = ax1.imshow(landscape, norm=matplotlib.colors.LogNorm())
        fiveskip = int(settings.perturb_steps/6)
        uspace = np.linspace(-settings.perturb_size, settings.perturb_size, settings.perturb_steps)
        rspace = np.arange(0, uspace.shape[0], fiveskip)
        ax1.set_xticks(rspace)
        ax1.set_xticklabels(np.around(uspace[::fiveskip], int(-np.log10(settings.perturb_size)) + 1))
        ax1.set_yticks(rspace)
        ax1.set_yticklabels(np.around(uspace[::fiveskip], int(-np.log10(settings.perturb_size)) + 1))
        ax1.set_xlabel("$u_1$")
        ax1.set_ylabel("$u_2$")
        cb = plt.colorbar(implot, ax=ax1)
        cb.set_label("$|E - E_0|$")
        if settings.record:
            plt.savefig("{}/run_{}_landscape.pdf".format(outpath, settings.id), bbox_inches='tight')
    
    if not settings.omitplots:
        plt.show()
    else:
        plt.close("all")

if __name__ == '__main__': 
    # build CLI
    import argparse
    parser = argparse.ArgumentParser()
    # unique run descriptor
    parser.add_argument('--id', type=str, default='test')
    # task settings (spin model + params)
    parser.add_argument('--model', type=str, default='TFIM')
    parser.add_argument('--modelparams', default=[2], nargs="+")
    parser.add_argument('--eps_loc_energy', type=float, default=1e-12)
    parser.add_argument('--renormalize', action='store_true')
    # network settings
    parser.add_argument('--topology', type=str, default="RBM")
    parser.add_argument('--nhid', type=int, default=4)
    parser.add_argument('--winit', type=float, default=0.1)
    parser.add_argument('--wbounds', type=float, nargs=2, default=[-np.inf, np.inf])
    parser.add_argument('--binit', type=float, default=0.1)
    parser.add_argument('--init_distr', type=str, default="standard")
    parser.add_argument('--bbounds', type=float, nargs=2, default=[-np.inf, np.inf])
    parser.add_argument('--spin', action='store_true')
    parser.add_argument('--translate_wb', action='store_true')
    parser.add_argument('--symmetries', default=[], nargs="*")
    # sampler settings
    parser.add_argument('--sample_pvis', action='store_true')
    parser.add_argument('--sample_grad', action='store_true')
    parser.add_argument('--nsamples', type=int, default=1000)
    parser.add_argument('--nchains', type=int, default=10)
    parser.add_argument('--burnin', type=int, default=0)
    parser.add_argument('--gamma_carleo', type=float, default=0.)
    # output settings
    parser.add_argument('--logfreq', type=int, default=1)
    parser.add_argument('--pathfromhome', type=str, default="NNQProject/results/simulation/EMIN")
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--omitplots', action='store_true')
    parser.add_argument('--save_weights', action='store_true')
    # optimizer settings
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--optim', type=str, default="GD")
    parser.add_argument('--lr', type=float, default=1.)
    parser.add_argument('--lr_schedule', type=str, default="exp")
    parser.add_argument('--lr_decay', type=float, default=1.)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--optimparams', default=[0.9, 0.999, 1e-8], nargs="*")
    parser.add_argument('--natgrad', action='store_true')
    parser.add_argument('--eps_reg', type=float, default=1.)
    parser.add_argument('--svd_cutoff', type=float, default=1e-15)
    parser.add_argument('--eps_reg_decay', type=float, default=1)
    parser.add_argument('--eps_reg_min', type=float, default=1e-6)
    parser.add_argument('--regname', type=str, default="pinv")
    # subsampling
    parser.add_argument('--subsamplesize', type=int, default=1000)
    parser.add_argument('--nsubsamples', type=int, default=10)
    parser.add_argument('--subsampling', action='store_true')
    # landscape analysis
    parser.add_argument('--landscape', action='store_true')
    parser.add_argument('--perturb_size', type=float, default=.01)
    parser.add_argument('--perturb_steps', type=int, default=10)
    # simulate dying neurons
    parser.add_argument('--dyingneurons', action='store_true')
    parser.add_argument('--p_dying', type=float, default=.01)

    
    args = parser.parse_args() 

    # report settings
    print("Run with following settings: {}".format(vars(args)))

    if args.model == "TFIM":
        # build TFIM Hamiltonian of given params
        N = int(args.modelparams[0])
        B = int(args.modelparams[1])
        modelstring = "TFIM_N{}_B{}".format(N, B)
        H_sparse = TFIM.buildHamiltonian(N, B, 1)
        if N <= 10:
            # calculate solution with ED 
            e_target, psi_target = TFIM.exact_diag_H(H_sparse)
            e_target = e_target[0]
            psi_target = psi_target[0]
            p_target = np.abs(psi_target)**2
            
            # add ED result 
            args.nvis = int(np.log2(p_target.shape[0]))
            args.ED_energy = e_target
            args.ED_distribution = p_target
        else:
            e_target = TFIM.theoretical_groundstate_energy(N, B, J=1)
            args.nvis = N
            args.ED_energy = e_target
            args.ED_distribution = None

    # run 
    data_dict = learn_groundstate(H_sparse, args)
    # landscape
    if args.landscape:
        landscape = landscape_analysis(H_sparse, args, w[-1,:,:], b[-1,:])
        data_dict["landscape"] = landscape
    
    outpath = None
    if args.record:
        # define output directory and create if necessary
        outpath = os.path.join('out', modelstring)
        print("Saving data to ", outpath)
        Path(outpath).mkdir(parents=True, exist_ok=True)
        # do plotting
        plotting(args, data_dict, outpath)
        # save data and settings
        if args.optim != "DE" and not args.save_weights:
            del data_dict["w"]
            del data_dict["b"]
        args.ED_distribution = None#args.ED_distribution.tolist()
        save_data(args, data_dict, outpath)
    else:
        plotting(args, data_dict, outpath)
