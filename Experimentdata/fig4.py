
import json
import numpy as np
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
import matplotlib.collections as mcol


def myexp(x, tau, a, b):
    return a*np.exp(-x/tau) + b


plot_n = [3, 5, 7, 9]


def plot_n_infidelity(ax, data, rbm_data_fid, show_software=False):
    bss_label_set = False
    for i, (n, fin_infid) in enumerate(zip(data['number_of_spins'], data['final_infidelities'])):
        if show_software:
            # software RBM points
            cpu_label = "CPU" if i == 0 else None
            rbm_fin_fid = rbm_data_fid[str(n)]
            infid = 1 - np.array(rbm_fin_fid)
            percentiles = np.percentile(infid, [15., 85.])
            median = np.median(infid)
            ax.errorbar(n, median, [[p] for p in percentiles], color="lightgrey", marker='o', alpha=1, label=cpu_label)

        # BSS-2 points
        c = f'C{i}' if n in plot_n else 'k'
        bss_label = "BSS-2" if n not in plot_n and not bss_label_set else None
        if not n in plot_n:
            bss_label_set = True
        percentiles = np.percentile(np.array(fin_infid), [15., 85.])
        median = np.median(np.array(fin_infid))
        ax.errorbar(n, median, [[p] for p in percentiles], color=c, marker='x', label=bss_label)

    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$1-F$')

    if show_software:
        #specify order of items in legend
        handles, labels = ax.get_legend_handles_labels()
        order = [1, 0]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower right")

def plot_n_energy(ax, data, groundstate_data, rbm_data, show_software=False):
    bss_label_set = False
    for i, (n, fin_ed) in enumerate(zip(data['number_of_spins'], data['final_energy_diffs'])):
        if show_software:
            # software RBM points
            cpu_label = "CPU" if i == 0 else None
            rbm_fin_en = rbm_data[str(n)]
            ediff = np.abs(np.array(rbm_fin_en) - groundstate_data[str(n)])/n
            percentiles = np.percentile(ediff, [15., 85.])
            median = np.median(ediff)
            ax.errorbar(n, median, [[p] for p in percentiles], color="lightgrey", marker='o', alpha=1, label=cpu_label)

        # BSS2 points
        c = f'C{i}' if n in plot_n else 'k'
        bss_label = "BSS-2" if n not in plot_n and not bss_label_set else None
        if not n in plot_n:
            bss_label_set = True
        percentiles = np.percentile(np.array(fin_ed), [15., 85.])
        median = np.median(np.array(fin_ed))
        ax.errorbar(n, median, [[p] for p in percentiles], color=c, marker='x', label=bss_label)

    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$|E-E_0|/N$')
    
    if show_software:
        #specify order of items in legend
        handles, labels = ax.get_legend_handles_labels()
        order = [1, 0]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower right")

def plot_learning_energy(ax, data, groundstate_data):
    for i, (n, learn_ed) in enumerate(zip(data['number_of_spins'], data['learning_energy_diffs'])):
        if n not in plot_n:
            continue
        ax.plot(np.array(learn_ed), color=f'C{i}')

    ax.set_yscale('log')
    ax.set_xlabel(r'training iteration')
    ax.set_ylabel(r'$|E-E_0|/N$')

def plot_learning_infidelity(ax, data):
    plot_n = [3, 5, 7, 9]
    for i, (n, learn_inf) in enumerate(zip(data['number_of_spins'], data['learning_infidelities'])):
        if n not in plot_n:
            continue
        ax.plot(learn_inf, color=f'C{i}')

    ax.set_yscale('log')
    ax.set_xlabel(r'training iteration')
    ax.set_ylabel(r'$1-F$')


def main():
    data = json.load(open('figure4.data'))
    rbm_data = json.load(open("rbm_fig4_200k.json"))
    rbm_data_fid = json.load(open("rbm_fig4_200k_fid.json"))
    groundstate_data = json.load(open("n_groundstate_energy.json"))

    print(data.keys())

    fig, ((ax_fin_en, ax_fin_inf), (ax_learn_en, ax_learn_inf)) = plt.subplots(2, 2, figsize=(8.8, 5.5))
    plot_n_infidelity(ax_fin_inf, data, rbm_data_fid)
    plot_n_energy(ax_fin_en, data, groundstate_data, rbm_data)
    plot_learning_infidelity(ax_learn_inf, data)
    plot_learning_energy(ax_learn_en, data, groundstate_data)

    plt.tight_layout()
    plt.savefig('figure4.pdf')

if __name__ == '__main__':
    main()

