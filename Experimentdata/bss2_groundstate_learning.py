import time
import os
import sys
import json
from pathlib import Path
import pylogging

import matplotlib.pyplot as plt
import numpy as np
import numba
from scipy.linalg import sqrtm
import scipy.sparse.linalg as sLA
import scipy.sparse as sparse
from scipy.sparse import load_npz, save_npz

# sampling imports
import hxsampling.hxsampler as hxs
import hxsampling.utils as utils

from quantum_utils import *
from rob_utils import *

def to_level(level, default=pylogging.LogLevel.INFO):
    """parse level string"""
    if isinstance(level, pylogging.LogLevel):
        return level
    else:
        return pylogging.LogLevel.toLevel(level, default)

pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname=".logs/groundstate_learning",
        print_location=False,
        color=True,
        )

# This must be the exact same string as is defined in the respective
# files where the logging originates...
for name, level in [
                    ('HXSampler', 'ERROR'),
                    ('Backend', 'WARN'),
                    ('fisch', 'ERROR')
                    ]:
    pylogging.set_loglevel(pylogging.get(name), to_level(level))

log = pylogging.get("groundstatesearch")

@numba.jit(nopython=True)
def grad_E_sampled(samples, n, n_vis, p_train, E_loc, E, spin, natural, regname, epsilon_reg, cutoff):
    """ return gradients of the variational energy with respect to the weights and biases """
    n_hid = n - n_vis
    nwparams = n_vis*n_hid
    nsamples = np.shape(samples)[0]
    
    e_grad = np.zeros(nwparams + n)
    
    S_first_order = np.zeros(nwparams + n)
    S_second_order = np.zeros((nwparams + n, nwparams + n))
    S = np.zeros((nwparams + n, nwparams + n))

    prefactor = E_loc - E

    if spin:
        samples = bin_to_spin(samples)
    
    for si in range(nsamples):
        
        cur_sample = samples[si,:]
        vis_state = int(bin_to_dec(cur_sample[:n_vis], n_vis)) if not spin else int(bin_to_dec(spin_to_bin(cur_sample[:n_vis]), n_vis))
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
        # calculate covariance matrix
        S = 0.25*(S_second_order*nsamples - np.outer(S_first_order, S_first_order))/nsamples**2
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

def calc_energy_model(p_model, hamiltonian):
    """ return the energy of the given variational state """
    H_sqp = hamiltonian.dot(np.sqrt(p_model))
    return np.dot(np.sqrt(p_model), H_sqp)

def learn_groundstate(hamiltonian, settings, outpath):
    """ search for the groundstate of Hamiltonian H by minimizing the energy given settings S """

    # extract some settings
    N_vis = settings.nvis
    N_hid = settings.nhid
    topology = settings.topology
    winit = settings.winit
    binit = settings.binit
    spin = settings.spin
    epochs = settings.epochs
    p_target = settings.ED_distribution
    e_target = settings.ED_energy
    lr = settings.lr
    eps_reg = settings.eps_reg

    # check consistent N_vis
    if not p_target is None:
        assert N_vis == int(np.log2(p_target.shape[0]))
    
    N = N_vis + N_hid
    
    # generate visible configuations (not scalable)
    vis_states = gen_states(N_vis, spin=spin)

    # create desired topology
    if topology.upper() == "RBM":
        N = N_vis + N_hid 
        w, b = init_wb(N, w_max=winit[1], w_min=winit[0], b_max=binit[1], b_min=binit[0])
        w = mask_weights_rbm(w, N_vis, N_hid)
    elif topology.upper() == "VPRBM":
        N = 2*N_vis + N_hid 
        w, b = init_wb(N, w_max=winit[1], w_min=winit[0], b_max=binit[1], b_min=binit[0])
        w = mask_weights_vprbm(w, N_vis, N_hid)
    elif topology.upper() == "DBM":
        N = np.sum(N_layers) 
        w, b = init_wb(N, w_max=winit[1], w_min=winit[0], b_max=binit[1], b_min=binit[0])
        w = mask_weights_dbm(w, N_layers)
    elif topology.upper() == "PRBM":
        N = N_vis + N_hid
        w, b = init_wb(N, w_max=winit[1], w_min=winit[0], b_max=binit[1], b_min=binit[0])
        w =  mask_weights_prrbm(w, N_vis, N_hid, restricted_layer=restricted_layer)
    elif topology.upper() == "FCBM":
        N = N_vis + N_hid
        w, b = init_wb(N, w_max=winit[1], w_min=winit[0], b_max=binit[1], b_min=binit[0])

    if len(settings.symmetries) > 0 and settings.topology == "RBM":
        if settings.model == "TFIM":
            if len(settings.symmetries) == 1 and "translation" in settings.symmetries: 
                # translation invariance
                assert N_hid % N_vis == 0
                alpha = int(N_hid/N_vis)

                btemp = b.copy()
                wtemp = w.copy()
                
                # the visible biases have only one free parameter
                b[:N_vis] = btemp[:N_vis].mean()
                
                # the hidden biases have M/N free parameters, the weights have M free parameters
                for j in range(N_hid):
                    b[N_vis + j] = btemp[N_vis + j // N_vis]
                    w[:N_vis, N_vis + j] = wtemp[:N_vis, N_vis + j // N_vis]
                    w[N_vis + j, :N_vis] = wtemp[N_vis + j // N_vis, :N_vis]
                
                print("Building in translation invariance symmetries")

    if (settings.sample_pvis or settings.sample_grad) and not settings.gibbs:
        # calibration
        dir_path = os.path.dirname(os.path.realpath(__file__))
        calib_path = os.path.join(dir_path, f"cube_{settings.chipsetup}_reset80_thres120_taum0.5_memcap10_isyn350.calib.npz")

        # initialize HXSampler object
        hxsampler = hxs.HXSampler(N, np.copy(w), np.copy(b),
                                    noiseweight=settings.noise_weight,
                                    noiserate=settings.noise_rate,
                                    noisemultiplier=settings.noise_multiplier,
                                    noisetype=settings.noise_type,
                                    calib_path=calib_path)

        # measure activation functions with respect to the bias values
        hxsampler.measure_activation_functions(duration=settings.duration, stepsize=25)
        taurefs = hxsampler.measured_taurefs

        # initialize biases at the center of the sigmoid curve
        b_0 = np.array([hxsampler.activation['fit'][i][0] for i in range(N)])
        b_alpha = np.array([hxsampler.activation['fit'][i][1] for i in range(N)])

        
        bshift = 2
        if bshift != 0:
            log.debug("all biases are shifted!")
        b[:] = b_0[:] - bshift*b_alpha[:]
        hxsampler.logical_bias = b[:]
        log.info("taurefs={}, bstart={}".format(taurefs, b))
    
    if settings.optim == "ADAM":
        beta1 = float(settings.optimparams[0])
        beta2 = float(settings.optimparams[1])
        eps_adam = float(settings.optimparams[2])
        mw = np.zeros(w.shape)
        mb = np.zeros(b.shape)
        vw = np.zeros(w.shape)
        vb = np.zeros(b.shape)
 
    b_epoch = np.zeros((epochs, N))
    w_epoch = np.zeros((epochs, N, N))   
    p_epoch = np.zeros((epochs, 2**N_vis))
    dkl_epoch = np.zeros(epochs)
    energy_epoch = np.zeros(epochs)
    lr_epoch = np.zeros(epochs)
    if settings.natgrad:
        cond_epoch = np.zeros(epochs)
    
    starttime = time.time()
    restart = False

    for e in range(epochs):
        # variable for when sampling is used
        samples = None

        # store weights and biases
        w_epoch[e, :, :] = w[:, :]
        b_epoch[e, :] = b[:]

        # decay step size
        lr = max(settings.lr*settings.lr_decay**e, settings.lr_min)
        lr_epoch[e] = lr
        if settings.natgrad and settings.eps_reg > 0 and settings.eps_reg_min > 0 and settings.eps_reg_decay > 0:
            eps_reg = max(settings.eps_reg*settings.eps_reg_decay**e, settings.eps_reg_min)
        
        # calc visible distribution
        if settings.sample_pvis or settings.sample_grad:
            if settings.gibbs: # do Gibbs sampling
                p_sampled_joint, samples = gibbs_sampling(w[:,:], b[:], vis_states, settings.num_chains, settings.num_samples, pvis=p_epoch[e - 1, :] if e > 0 else None, burn_in=settings.burnin, spin=spin)
                p_vis = p_sampled_joint.sum(axis=1)
            else: # do LIF sampling
                issue_present = True
                vis_non_spiking = False
                max_sampling = 1
                while issue_present and max_sampling > 0:
                    log.debug("Before LIF_sampling")
                    samples, rep_spikes = LIF_sampling(hxsampler, settings.duration, settings.dt, settings.repetitions, return_spikes=True)
                    log.debug("After LIF_sampling")
                    samples = samples.reshape(-1, N)
                    max_sampling -= 1

                    # check for issues like non-spiking neurons, spike loss
                    dt_spikes = rep_spikes[0][1:, 0] - rep_spikes[0][:-1,0]
                    spiking_neurons = np.unique(rep_spikes[0][:, 1])
                    if max(spiking_neurons) >= np.shape(spiking_neurons)[0]:
                        # there are non-spiking neurons
                        non_spiking_neurons = np.setdiff1d(np.arange(N), spiking_neurons) 
                        vis_non_spiking = min(non_spiking_neurons - N_vis) < 0
                        log.info("Non-spiking neurons: {}{}".format(non_spiking_neurons, "(including visibles)" if vis_non_spiking else ""))
                    elif (dt_spikes > 100e-6).any():
                        # spikes have been lost
                        log.info("Spike loss detected")
                    else:
                        issue_present = False

                if vis_non_spiking:
                    restart = e < int(0.75*epochs)
                    break

        # calc visible distribution
        if settings.sample_pvis:
            p_vis = calc_pvis(samples, N_vis, N_hid, spin=spin)
        else:
            p_vis = calc_pvis_ana(w[:,:], b[:], vis_states, spin)

        # store visible distribution
        p_epoch[e, :] = p_vis[:]

        # calc DKL
        if not p_target is None:
            dkl_epoch[e] = calc_dkl(N_vis, p_target, p_vis)
        # calc energy
        energy_epoch[e] = calc_energy_model(p_vis, hamiltonian)
        # calc local energy
        sqrt_pvprime_pv = np.sqrt(np.outer(p_epoch[e, :] + settings.eps_loc_energy, 1/(p_epoch[e, :] + settings.eps_loc_energy)))
        E_loc = np.diag(hamiltonian.dot(sqrt_pvprime_pv))
        
        # calc gradients
        dw = np.zeros((N, N))
        db = np.zeros(N)
       
        if settings.sample_grad:
            db, dw, cond = grad_E_sampled(samples, N, N_vis, p_vis, E_loc, energy_epoch[e], spin, settings.natgrad, settings.regname, eps_reg, settings.svd_cutoff)
        
        if not cond is None:
            cond_epoch[e] = cond

        if topology.upper() == "RBM":
            dw = mask_weights_rbm(dw, N_vis, N_hid)
        elif topology.upper() == "VPRBM":
            dw = mask_weights_vprbm(dw, N_vis, N_hid)
        elif topology.upper() == "DBM":
            dw = mask_weights_dbm(dw, N_layers)
        elif topology.upper() == "PRRBM":
            dw =  mask_weights_prrbm(dw, N_vis, N_hid, restricted_layer=restricted_layer)

        if dw.any() != dw.T.any():
            raise ValueError("Weight update is not symmetric!")
        
        if settings.symmetries:
            if settings.model == "TFIM" and settings.topology == "RBM":
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
                pass

        if settings.optim == "ADAM":
            # update weights using some optimization algorithm
            # ADAM
            mw = beta1*mw + (1. - beta1)*dw
            mb = beta1*mb + (1. - beta1)*db

            vw = beta2*vw + (1. - beta2)*dw**2
            vb = beta2*vb + (1. - beta2)*db**2

            mhw = mw/(1. - beta1**(e + 1))
            vhw = vw/(1. - beta2**(e + 1))
            wnew = w - lr*np.divide(mhw, np.sqrt(vhw) + eps_adam)
            
            mhb = mb/(1. - beta1**(e + 1))
            vhb = vb/(1. - beta2**(e + 1))
            bnew = b - lr*np.divide(mhb, np.sqrt(vhb) + eps_adam)
        
        elif settings.optim == "GD":
            # gradient descent
            wnew = w - lr*dw
            bnew = b - lr*db

        # clip weights and biases
        wnew = np.clip(wnew, settings.wbounds[0], settings.wbounds[1])
        bnew = np.clip(bnew, settings.bbounds[0], settings.bbounds[1])

        if not settings.gibbs and (settings.sample_pvis or settings.sample_grad):
            hxsampler.logical_weight = np.round(wnew).astype(int)
            hxsampler.logical_bias = np.round(bnew).astype(int)

        # carry over new weights and biases to next iteration
        w[:,:] = wnew
        b[:] = bnew
        
        log.debug("Finished analysis")

        # logging some error measure 
        if settings.loginterval > 0 and e % settings.loginterval == 0:
            log.info("epoch{} / ee {:.4f} / kl-div {:.5f} / lr {:.3f} / reg {:.5f} / time {:.2f}".format(e, np.abs(energy_epoch[e] - e_target), dkl_epoch[e], lr, eps_reg, time.time() - starttime))
            starttime = time.time()
        if settings.record and settings.save_every_epoch:
            data_dict = {"w": w_epoch, "b": b_epoch, "energy": energy_epoch, "dkl": dkl_epoch, "p": p_epoch, "lr": lr_epoch}
            if not args.save_weights:
                del data_dict["w"]
                del data_dict["b"]
            # save data and settings
            save_data(args, data_dict, outpath, nosettings=True)

    hxsampler.connection_mgr.__exit__()
    return w_epoch, b_epoch, p_epoch, energy_epoch, dkl_epoch, lr_epoch, cond_epoch if settings.natgrad else None, restart

def plotting(settings, array_dict, outpath):
    """ Plot results from the learning process"""
    num_subplots_metrics = 3
    subplots_metrics_ind = 1
    
    # retrieve data to be plotted
    e_target = settings.ED_energy 
    p_target = settings.ED_distribution
    energy = array_dict["energy"]
    dkl = np.nan_to_num(array_dict["dkl"])
    p = array_dict["p"]
    lr = array_dict["lr"]
    energy_error = np.abs(energy- e_target)
    #condition = np.nan_to_num(array_dict["cond"])
    if p_target is None:
        infidelity = np.zeros_like(dkl)
        num_subplots_metrics = 1
    else:
        infidelity = 1. - np.sqrt(p) @ np.sqrt(p_target)

    # DKL and absolute energy difference vs epochs
    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(num_subplots_metrics, 1, subplots_metrics_ind)
    subplots_metrics_ind += 1
    ax1.plot(energy_error)
    ax1.set_ylabel("$|E - E_0|$")
    ax1.set_yscale("log")
    ax1.set_yticks(10**(np.arange(-10, 2, 1, dtype=float)))
    ax1.set_ylim([min(energy_error), max(energy_error)])
    ax1.grid(which="both")
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    if p_target is not None:
        ax3 = fig.add_subplot(num_subplots_metrics, 1, subplots_metrics_ind)
        subplots_metrics_ind += 1
        ax3.plot(dkl)
        ax3.set_yscale("log")
        ax3.set_yticks(10**(np.arange(-10, 10, 1, dtype=float)))
        ax3.set_ylim([min(dkl), max(dkl)])
        ax3.set_ylabel("$D_{KL}$")
        ax3.grid(which="both")

        ax4 = fig.add_subplot(num_subplots_metrics, 1, subplots_metrics_ind)
        subplots_metrics_ind += 1
        ax4.plot(infidelity)
        ax4.set_yscale("log")
        ax4.set_ylabel("$1 - F$")
        ax4.set_yticks(10**(np.arange(-10, 10, 1, dtype=float)))
        ax4.set_ylim([np.min(infidelity), np.max(infidelity)])
        ax4.set_xlabel("epoch")
        ax4.grid(which="both")

    if settings.record:
        plt.savefig("{}/run_{}_dkl_ediff.pdf".format(outpath, settings.id), bbox_inches='tight')
    
    #if settings.save_weights:
    w = array_dict["w"]
    b = array_dict["b"]

    fig2 = plt.figure()
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

    if not settings.omitplots:
        plt.show()
    
    return infidelity

if __name__ == '__main__': 
    # build CLI
    import argparse
    parser = argparse.ArgumentParser()
    # unique run descriptor
    parser.add_argument('--id', type=str, default='test')
    # task settings (spin model + params)
    parser.add_argument('--model', type=str, default='TFIM')
    parser.add_argument('--modelparams', default=[4, 1], nargs="+", help="N, h/J")
    parser.add_argument('--eps_loc_energy', type=float, default=1e-6, help="bias to avoid dividing by zero")
    # network settings
    parser.add_argument('--topology', type=str, default="RBM")
    parser.add_argument('--nhid', type=int, default=8)
    parser.add_argument('--winit', type=float, nargs=2, default=[-50, 50])
    parser.add_argument('--binit', type=float, nargs=2, default=[500, 600], help="not used")
    parser.add_argument('--bbounds', type=float, nargs=2, default=[300, 800])
    parser.add_argument('--wbounds', type=float, nargs=2, default=[-63, 63])
    parser.add_argument('--spin', action='store_true', help="not used")
    parser.add_argument('--symmetries', default=[], nargs="*")
    # sampler settings
    parser.add_argument('--sample_pvis', action='store_true')
    parser.add_argument('--sample_grad', action='store_true')
        # Gibbs
    parser.add_argument('--gibbs', action='store_true')
    parser.add_argument('--nsamples', type=int, default=1000)
    parser.add_argument('--nchains', type=int, default=10)
    parser.add_argument('--burnin', type=int, default=0)
        # hardware
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--repetitions', type=int, default=5, help="number of repeated sampling expermients per epoch")
    parser.add_argument('--dt', type=float, default=5e-6)
    parser.add_argument('--noise_weight', type=float, default=20)
    parser.add_argument('--noise_rate', type=float, default=74000)
    parser.add_argument('--noise_multiplier', type=int, default=4)
    parser.add_argument('--noise_type', type=str, default='On-chip')
    parser.add_argument('--chipsetup', type=int, default=74, help="load correct calib")
    # output settings
    parser.add_argument('--loginterval', type=int, default=1, help="number of epochs per logging line")
    parser.add_argument('--pathfromhome', type=str, default="data/EMIN")
    parser.add_argument('--record', action='store_true', help="need to be set to save data")
    parser.add_argument('--save_weights', action='store_true', help="save weights and biases arrays")
    parser.add_argument('--save_every_epoch', action='store_true')
    parser.add_argument('--omitplots', action='store_true')
    # optimizer settings
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--optim', type=str, default="ADAM")
    parser.add_argument('--lr', type=float, default=1.)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--optimparams', default=[0.9, 0.999, 1e-8], nargs="*")
    parser.add_argument('--natgrad', action='store_true')
    parser.add_argument('--eps_reg', type=float, default=1.)
    parser.add_argument('--svd_cutoff', type=float, default=1e-15)
    parser.add_argument('--eps_reg_decay', type=float, default=1)
    parser.add_argument('--eps_reg_min', type=float, default=1e-6)
    parser.add_argument('--regname', type=str, default="pinv")

    args = parser.parse_args() 

    # some reasonable presets for the hardware
    args.sample_pvis = True
    args.sample_grad = True
    args.save_weights = True
    args.save_every_epoch = True
    args.record = True
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    args.id = f"emintest_{timestamp}"

    # report settings
    log.info("Run with following settings: {}".format(vars(args)))

    if args.model == "TFIM":
        # build TFIM Hamiltonian of given params
        N = int(args.modelparams[0])
        B = float(args.modelparams[1])
        modelstring = "TFIM_N{}_B{}".format(N, B)
        if Path(f"hamiltonians/{modelstring}.npz").is_file():
            H_sparse = load_npz(f"hamiltonians/{modelstring}.npz")
        else:
            H_sparse = buildHamiltonian(N, B, 1)
            save_npz(f"hamiltonians/{modelstring}.npz", H_sparse)
        if N <= 10:
            # calculate solution with ED 
            e_target, psi_target = exact_diag_H(H_sparse)
            e_target = e_target[0]
            psi_target = psi_target[0]
            p_target = np.abs(psi_target)**2
            
            # add ED result 
            args.nvis = int(np.log2(p_target.shape[0]))
            args.ED_energy = e_target
            args.ED_distribution = p_target
        else:
            # calculate analytic groundstate energy
            e_target = theoretical_groundstate_energy(N, B, J=1)
            args.nvis = N
            args.ED_energy = e_target
            args.ED_distribution = None

    outpath = None
    if args.record:
        # define output directory and create if necessary
        outpath = os.path.join(Path.home(), args.pathfromhome, modelstring)
        log.info("Creating folder {}".format(outpath))
        Path(outpath).mkdir(parents=True, exist_ok=True)
    
    # run 
    starttime = time.time()
    restart = True
    restarts_left = 1
    while restart and restarts_left + 1 > 0:
        w, b, p, energy, dkl, lr, cond, restart = learn_groundstate(H_sparse, args, outpath)
        if restart:
            restarts_left -= 1
    data_dict = {"w": w, "b": b, "energy": energy, "dkl": dkl, "p": p, "lr": lr}

    log.info(f"Finished run after {(time.time() - starttime)/3600}h")
    
    if args.record:
        # do plotting
        infidelity = plotting(args, data_dict, outpath)
        data_dict["infid"] = infidelity
        if not args.save_weights:
            del data_dict["w"]
            del data_dict["b"]
        # save data and settings
        args.ED_distribution = None
        save_data(args, data_dict, outpath)
    else:
        plotting(args, data_dict, outpath)
