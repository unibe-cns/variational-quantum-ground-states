
import json
import numpy as np
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt


def plot_nhid_energy(ax, data, data_fig4):
    final_energies = np.array(data['final_energies'])
    final_energies = np.swapaxes(final_energies, 0, 1)
    for i, (n_hidden, energies) in enumerate(zip(data['number_of_hiddens'], final_energies)):
        print(energies.shape)
        ax.errorbar(data['number_of_spins'], np.median(energies, axis=1), np.percentile(energies, [15., 85.], axis=1), color=f'C{i+3}', label=r'$N_h='+f'{n_hidden}'+r'$')

   # add figure 4 data
    num_spins = []
    final_energy_diffs_mean = []
    final_energy_diffs_std = []
    for i, (n, fin_ener) in enumerate(zip(data_fig4['number_of_spins'], data_fig4['final_energy_diffs'])):
        fin_ener = np.array(fin_ener)
        num_spins.append(n)
        final_energy_diffs_mean.append(np.median(fin_ener))
        final_energy_diffs_std.append(np.percentile(fin_ener, [15., 85.]))
    label = r"$N_h="+f"{'40/50'}"+r"$"
    ax.errorbar(num_spins, np.array(final_energy_diffs_mean), list(zip(*final_energy_diffs_std)),
                color="black", label=label, linestyle="--")

    ax.set_yscale('log')
    ax.set_xlabel(r'N')
    ax.set_ylabel(r'$|E-E_0|/N$')
    ax.set_ylim(0.1e-4, None)
    ax.legend(loc='lower right', fontsize='x-small')


def plot_nhid_inf(ax, data, data_fig4):
    final_infidelities = np.array(data['final_infidelities'])
    final_infidelities = np.swapaxes(final_infidelities, 0, 1)
    for i, (n_hidden, infs) in enumerate(zip(data['number_of_hiddens'], final_infidelities)):
        ax.errorbar(data['number_of_spins'], np.median(infs, axis=1), np.percentile(infs, [15., 85.], axis=1), color=f'C{i+3}', label=r'$N_h='+f'{n_hidden}'+r'$')


   # add data from figure 4
    num_spins = []
    final_infid_diffs_mean = []
    final_infid_diffs_std = []
    for i, (n, fin_infid) in enumerate(zip(data_fig4['number_of_spins'], data_fig4['final_infidelities'])):
        fin_infid = np.array(fin_infid)
        num_spins.append(n)
        final_infid_diffs_mean.append(np.median(fin_infid))
        final_infid_diffs_std.append(np.percentile(fin_infid, [15., 85.]))
    ax.errorbar(num_spins, np.array(final_infid_diffs_mean), list(zip(*final_infid_diffs_std)),
                color="black", label=None, linestyle="--")

    ax.set_yscale('log')
    ax.set_xlabel(r'N')
    ax.set_ylabel(r'$1-F$')


def plot_discretization(ax, data, dkl):
    xticklabels = [64, 32, 16, 8, 4, 2, 1]
    xticks = np.arange(len(xticklabels))
    print(xticks)
    for wmax in [63]: # [15, 31, 63]:
        median = np.median(data[f'dkls_{wmax}'], axis=1)
        print(len(median))
        low = np.percentile(data[f'dkls_{wmax}'], 15, axis=1)
        up = np.percentile(data[f'dkls_{wmax}'], 85, axis=1)
        ax.errorbar(xticks, median, [low, up], color='C2', label=r'$W_m='f'{wmax}'+r'$')
    ax.axhline(np.median(dkl[-200:]), color="black", linestyle="--", label="final DKL")
    print("DKL = {}".format(np.median(dkl[-200:])))
    ax.set_xlabel(r'step size $\Delta W$')
    ax.set_ylabel(r'$D_\mathrm{KL}(p_{\Delta W}||p_\mathrm{full})$')
    ax.set_yscale('log')
    ax.set_ylim(0.3e-3, 5.)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


def plot_duration(ax, ax_pseudo, data, data2, dkl):
    durations = np.array(data['durations'])
    reconv_dkls_to_target = np.array(data['reconv_dkls_to_target'])
    reconv_dkls_to_final = np.array(data['reconv_dkls_to_final'])
    diff_weight_dkls = np.array(data['diff_weight_dkls'])
    diff_weight_dkls_975 = np.array(data2['diff_weight_dkls_975'])

    print(durations.shape, reconv_dkls_to_target.shape, reconv_dkls_to_final.shape, diff_weight_dkls.shape)

    stride = 3
    offset = 2
    print(durations[::stride])
    idx = [i for i in range(offset, durations.shape[0], stride)]
    print(idx)

    ax.errorbar(durations[idx], np.median(reconv_dkls_to_target[:, idx], axis=0), yerr=np.percentile(reconv_dkls_to_target[:, idx], q=[15., 85.], axis=0), c='C0', ls='-', label=r'$p_\mathrm{target}=\left<p^{(T)}\right>_n$')
    idx = [i-1 for i in idx]
    ax.errorbar(durations[idx], np.median(reconv_dkls_to_final[:, idx], axis=0), yerr=np.percentile(reconv_dkls_to_final[:, idx], q=[15., 85.], axis=0), c='C1', ls='-', label=r'$p_\mathrm{target}=p^{(T)}$')
    idx = [i-1 for i in idx]
    ax_pseudo.errorbar(durations[idx], np.median(diff_weight_dkls[:, idx], axis=0), yerr=np.percentile(diff_weight_dkls[:, idx], q=[15., 85.], axis=0), c='C0', ls='-', label=r'$p_\mathrm{flip}$=10%')
    ax_pseudo.errorbar(durations[idx], np.median(diff_weight_dkls_975[:, idx], axis=0), yerr=np.percentile(diff_weight_dkls_975[:, idx], q=[15., 85.], axis=0), c='C1', ls='-', label=r'$p_\mathrm{flip}$=2.5%')
    ax_pseudo.axhline(np.median(dkl[-200:]), color="black", linestyle="--", label="final DKL")
    ax.axhline(np.median(dkl[-200:]), color="black", linestyle="--", label="final DKL")

    ax.set_xlabel(r'duration $t$ [s]')
    ax.set_ylabel(r'$D_\mathrm{KL}(p^{(t)}||p_\mathrm{target})$')
    ax.set_xscale('log')
    ax.set_xticks([1e-2, 1e-1, 1e0, 1e1])
    ax.set_yscale('log')
    ax.set_ylim(0.3e-3, 5.)
    # ax.set_xlim(None, 40)
    ax.legend(fontsize='small')

    ax_pseudo.set_xlabel(r'duration $t$ [s]')
    ax_pseudo.set_ylabel(r'$D_\mathrm{KL}(\tilde{p}^{(t)}||p^{(T)})$')
    ax_pseudo.set_xscale('log')
    ax_pseudo.set_xticks([1e-2, 1e-1, 1e0, 1e1])
    ax_pseudo.set_yscale('log')
    ax_pseudo.set_ylim(0.3e-3, 5.)
    # ax_pseudo.set_xlim(None, 40)
    ax_pseudo.legend(fontsize='small')


def plot_wdistr(ax_wdistr, data_wdistr):
    ax_wdistr.hist(data_wdistr[-200:, 8:,:8].flatten(), bins=np.arange(-63, 64, 4), density=True)
    ax_wdistr.set_ylabel(r"frequency $\langle \rho(w) \rangle$")
    ax_wdistr.set_xlabel(r"weight value $w$")
    ax_wdistr.set_yticklabels([])
    ax_wdistr.set_yticks([])
    ax_wdistr.set_xlim(-70, 70)
    ax_wdistr.set_xticks([-63, 0, 63])


def main():
    data_fig4 = json.load(open('figure4.data'))

    data_wdistr = np.load("data/run_nhidsweep_nhid20_reps10_noiseweight15_isyn500_lr1_W74F3_w.npy")
    data_finaldkl = np.load("data/run_nhidsweep_nhid20_reps10_noiseweight15_isyn500_lr1_W74F3_dkl.npy")

    #plt.savefig("../new_figs/figure5_weight_distr.pdf")

    data = json.load(open('figure5.data'))
    # dict_keys(['number_of_spins', 'number_of_hiddens', 'final_energies',
    # 'final_infidelities', 'final_dkls'])
    data_stability = json.load(open('figure5_stability.data'))
    data_stability2 = json.load(open('figure5_training.data'))
    print(data_stability2.keys())
    print(data_stability['nrepetitions'])
    print(data_stability['zero_fraction_in_dw'])
    # dict_keys(['n', 'durations', 'mean_p', 'all_dkls_vs_single_run',
    # 'all_dkls_vs_all_runs'])
    data_discretization = json.load(open('figure5_discretization.data'))
    # dict_keys(['nbits', 'dkls_15', 'dkls_31', 'dkls_63'])

    print(data_discretization.keys())

    fig, ((ax_nhid_energy, ax_nhid_inf, ax_wdistr), (ax_discretization, ax_pseudoupdate, ax_duration)) = plt.subplots(2, 3, figsize=(8.8, 4.5))
    plot_wdistr(ax_wdistr, data_wdistr)
    plot_nhid_energy(ax_nhid_energy, data, data_fig4)
    plot_nhid_inf(ax_nhid_inf, data, data_fig4)
    plot_discretization(ax_discretization, data_discretization, data_finaldkl)
    plot_duration(ax_duration, ax_pseudoupdate, data_stability, data_stability2, data_finaldkl)

    plt.tight_layout()
    plt.savefig('figure5.pdf')

if __name__ == '__main__':
    main()

