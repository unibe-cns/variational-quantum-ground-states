
import json
import numpy as np
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
import matplotlib.collections as mcol


def myexp(x, tau, a, b):
    return a*np.exp(-x/tau) + b


plot_n = [3, 5, 7, 9]


def plot_n_energy(ax, data, groundstate_data, rbm_data):
    bss_label_set = False
    for i, (n, fin_ed) in enumerate(zip(data['number_of_spins'], data['final_energy_diffs'])):
        # software RBM points
        cpu_label = "CPU" if i == 0 else None
        rbm_fin_en = rbm_data[str(n)]
        ediff = np.abs(np.array(rbm_fin_en) - groundstate_data[str(n)])/n
        percentiles = np.percentile(ediff, [15., 85.])
        median = np.median(ediff)
        ax.errorbar(n + 0.2, median, [[p] for p in percentiles], color="k", marker='o', alpha=1, label=cpu_label)

        # BSS2 points
        c = f'C{i}' if n in plot_n else 'k'
        bss_label = "BSS-2" if n not in plot_n and not bss_label_set else None
        if not n in plot_n:
            bss_label_set = True
        percentiles = np.percentile(np.array(fin_ed), [15., 85.])
        median = np.median(np.array(fin_ed))
        ax.errorbar(n, median, [[p] for p in percentiles], color="k", marker='x', label=bss_label)

    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$|E-E_0|/N$')
    #specify order of items in legend
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower right")


def main():
    data = json.load(open('figure4.data'))
    rbm_data = json.load(open("rbm_fig4_200k.json"))
    rbm_data_fid = json.load(open("rbm_fig4_200k_fid.json"))
    groundstate_data = json.load(open("n_groundstate_energy.json"))

    print(data.keys())

    fig = plt.figure(figsize=(4, 3))
    plot_n_energy(fig.gca(), data, groundstate_data, rbm_data)

    plt.tight_layout()
    plt.savefig('figure4_cpu_comparison.pdf')

if __name__ == '__main__':
    main()

