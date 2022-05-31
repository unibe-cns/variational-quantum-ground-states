
import json
import numpy as np
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.gridspec as gridspec

from matplotlib import pyplot as plt
import matplotlib.collections as mcol


def myexp(x, tau, a, b):
    return a*np.exp(-x/tau) + b


def plot_mag_corr(ax, ax2, data, colors):
    magline, = ax.plot(data['exact_fields'], data['exact_magnetizations'],
             linestyle='-', color='k', label=r'$\left<\sigma_x\right>$')
    corrline, = ax2.plot(data['exact_fields'], data['exact_corrlength'],
             linestyle='-', color='k', label=r'$\zeta_{xx}$')
    fields, means_h = np.array(data['fields']), np.array(data['means_h'])
    for i, (field, meanmag, corrlength, corrlength_std) in enumerate(zip(fields, means_h, data['corrlength_data'], data['corrlength_data_std'])):
        magmarker = ax.errorbar(field, meanmag[0].mean(), meanmag[0].std(), marker='o', color=colors[i], mfc='none')
        corrmarker = ax2.errorbar(field, corrlength, corrlength_std, marker='o', color=colors[i], mfc='none')
    ax.set_xscale('log')
    ax2.set_xscale('log')
    ax.set_xticklabels([])
    ax2.set_xlabel(r'$h\;/\;J$')

    labels=[r'$\left<\sigma_x\right>$', r'$\xi_{zz}$']
    ax.set_ylabel(labels[0])
    ax2.set_ylabel(labels[1])


def plot_corrlength(ax, data, colors):
    plot_fields = [0.9, 1.0, 1.25, 10.0]
    for i, (field, corr, corr_std, corr_target) in enumerate(zip(data['fields'], data['allcorrs'], data['allcorrs_std'], data['allcorrs_target'])):
        if field not in plot_fields:
            continue
        color = colors[i]
        data, = ax.plot(np.arange(len(corr)), corr, marker='o', mfc="none",
                    color=color, label=r'$h/J='+f'{field}'+r'$', linestyle='')
        target, = ax.plot(np.arange(len(corr)), corr_target, marker='x', color=color, linestyle="", mfc='none',
                          markersize=10)
        popt, pcov = curve_fit(myexp, np.arange(len(corr)), corr)
        d = np.linspace(0, len(corr), 100)
        p, = ax.plot(d, myexp(d, *popt), color=color, linestyle=':')
    target_proxy, = ax.plot([], [], color="black", marker="x", mfc="none", linestyle="")
    data_proxy, = ax.plot([], [], color="black", marker="o", mfc="none", linestyle="")
    fit_proxy, = ax.plot([], [], color="black", marker="", linestyle=":")
    leg = ax.legend(handles=[fit_proxy, data_proxy, target_proxy],
                    labels=['best fit', 'data', 'theory'],
                    fontsize="small")
    ax.set_xlabel('d')
    ax.set_ylabel(r'$C_{zz}(d)$')

def plot_mag_prob(ax, data, data_h01, colors):
    plot_fields = [0.1, 0.9, 5.0]
    for i, (field, pmag, pmag_err, pmag_target) in enumerate(zip(data['fields'], data['prob_magnetization'], data['prob_magnetization_err'], data['prob_magnetization_target'])):
        color = colors[i]
        if field in plot_fields and not field == 0.1:
            err = ax.errorbar(data['magnetization'], pmag, np.sqrt(pmag_err), marker='o', mfc='none', color=color, label='data', linestyle="-")
            line, = ax.plot(data['magnetization'], pmag_target, marker='x', color=color, mfc='none',
                    linestyle='-', label='theory', markersize=10)

    h01_data = [(field, pmag, pmag_err, pmag_target) for i, (field, pmag, pmag_err, pmag_target) in enumerate(zip(data_h01['fields'], data_h01['prob_magnetization'], data_h01['prob_magnetization_err'], data_h01['prob_magnetization_target']))]
    
    for i, (field, pmag, pmag_err, pmag_target) in enumerate(h01_data):
        if i == 0:
            marker = "v"
            linestyle = '-.'
        else:
            marker = "^"
            linestyle = '--'
        if i < 2:
            alpha = 0.5
        else:
            alpha = 1
        err = ax.errorbar(data['magnetization'], pmag, np.sqrt(pmag_err), marker=marker, linestyle=linestyle,
                          color=colors[0], label='data', mfc="none", alpha=alpha)
    
    # calculate mean of both symmetry broken ground states
    pmag_mean = 0.5 * (np.array(h01_data[0][1]) + np.array(h01_data[1][1]))
    err, = ax.plot(data['magnetization'], pmag_mean, marker='>', linestyle=':', color=colors[0], label='mixed')
    line, = ax.plot(data['magnetization'], h01_data[0][3], marker='x', color=colors[0], mfc='none',
                    linestyle='-', label='theory')

    ax.set_yscale('log')
    ax.set_xlabel(r'$m$')
    ax.set_ylabel(r'$p(m_z=m)$')
    err_proxy, = ax.plot([], [], marker="o", color="black", mfc="none", linestyle="-")
    h01_proxy_low, = ax.plot([], [], marker="v", linestyle="-.", color="black", mfc="none")
    h01_proxy_high, = ax.plot([], [], marker="^", linestyle="--", color="black", mfc="none")
    line_proxy, = ax.plot([], [], marker="x", linestyle="-", color="black")
    dotted_proxy, = ax.plot([], [], marker=">", linestyle=":", color="black")
    ax.legend(handles=[(err_proxy, h01_proxy_low, h01_proxy_high), line_proxy, dotted_proxy],
              labels=['data', 'theory', 'mixed'],
              handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None)},
              fontsize='small')

def clear_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_colorbar(ax, data):
    import matplotlib as mpl
    fields = data["fields"] 

    cmap = plt.cm.plasma  # define the colormap
    offset_start = 0.1
    offset_end = 0.2
    cmap_start = int(offset_start*cmap.N)
    cmap_end = int((1 - offset_end)*cmap.N)
    cmaplist = [cmap(i) for i in range(cmap_start, cmap_end, 1)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'discrete cmap', cmaplist, cmap_end - cmap_start)

    # define the bins and normalize
    bounds = np.linspace(0, len(fields), len(fields) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap_end - cmap_start)
    colors = [cmap(norm(i)) for i in range(len(fields))]

    # create a second axes for the colorbar
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
        spacing='proportional', ticks=[b + 0.5 for b in bounds], boundaries=bounds, format='%1i')
    cb.set_ticklabels(fields)
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel(r'$h\,/\,J$', size=12)
    c0p1 = tuple((np.array([68, 1, 84])/255.).tolist())
    c0p5 = tuple((np.array([68, 57, 131])/255.).tolist())
    c0p9 = tuple((np.array([48, 104, 142])/255.).tolist())
    c1p0 = tuple((np.array([32, 144, 141])/255.).tolist())
    c1p25 = tuple((np.array([53, 183, 121])/255.).tolist())
    c5 = tuple((np.array([142, 215, 68])/255.).tolist())
    c10 = tuple((np.array([254, 231, 36])/255.).tolist())
    return colors

def main():
    data = json.load(open('figure3.data'))
    data_low = np.load("figure3_run_emin_lowactivity_p.npy")
    data_high = np.load("figure3_run_emin_highactivity_p.npy")
    data_h01 = json.load(open('figure3_pmag_h0.1.data'))

    fig = plt.figure(figsize=(8.8, 5.5))
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, height_ratios=[1., 1.], width_ratios=[1., 0.05, 1.])
    ax_colorbar = fig.add_subplot(spec[0:2, 1])
    ax_xmag = fig.add_subplot(spec[0, 0])
    ax_corrd = fig.add_subplot(spec[0, 2])
    ax_corrh = fig.add_subplot(spec[1, 0])
    ax_zmag = fig.add_subplot(spec[1, 2])

    colors = plot_colorbar(ax_colorbar, data)
    plot_mag_corr(ax_xmag, ax_corrh, data, colors)
    plot_corrlength(ax_corrd, data, colors)
    plot_mag_prob(ax_zmag, data, data_h01, colors)

    plt.tight_layout()
    plt.savefig('figure3.pdf')

if __name__ == '__main__':
    main()

