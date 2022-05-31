import os
import json
import numpy as np
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
import matplotlib.collections as mcol
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.colorbar import Colorbar


def plot_matrix(ax, data):
    np.random.seed(12345678)
    weight = np.zeros((64, 80))
    # network
    weight[4:28, :4] = np.random.randint(-63, 63, (24, 4))
    weight[:4, 4:28] += weight[4:28, :4].T
    weight[28:weight.shape[0]-32] = 0
    # noise
    for i in range(28):
        rands_in = np.random.choice(range(16), int(5), replace=False)
        rands_ex = np.random.choice(range(16), int(5), replace=False)
        for multi in range(int(5)):
            weight[i, weight.shape[1]-32+rands_in[multi]] = 20
            weight[i, weight.shape[1]-16+rands_ex[multi]] = -20

    X, Y = np.meshgrid(range(weight.shape[1]), range(weight.shape[0]))
    # WTF matplotlib is doing something very wrong and stupid...
    cplot = ax.scatter(X+3, Y+2, c=weight, marker='s')

    rect = mpl.patches.Rectangle((0, 0), 29, 28, linewidth=1, edgecolor='w', facecolor='none')
    ax.add_patch(rect)
    rect = mpl.patches.Rectangle((48, 0), 33, 28, linewidth=1, edgecolor='w', facecolor='none')
    ax.add_patch(rect)
    ax.text(1, 32, 'network', fontsize=8, color='w')
    ax.text(56, 32, 'noise', fontsize=8, color='w')

    ax.set_xlabel('source neurons')
    ax.set_ylabel('target neurons')
    ax.set_xticks([0, 28, weight.shape[1]-32, weight.shape[1]-16, weight.shape[1]])
    ax.set_xticklabels([0, 28, 192, 224, 256], rotation=30)
    ax.set_xlim(-1, 81)
    ax.set_ylim(-1, 64)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cb = plt.colorbar(ax = ax, mappable = cplot, orientation = 'vertical')
    cb.set_label(r'weight $w$ [a.u.]')

def plot_activation(ax, data):
    nspikes = np.array(data['nspikes'])

    for i in range(5, 196):
        ax.plot(data['biases'], nspikes[:, i], c='silver', alpha=0.6)
    for i in range(4):
        ax.plot(data['biases'], nspikes[:, i], c='k')

    ax.set_xlabel(r'leak potential $V_l$ [a.u.]')
    ax.set_ylabel('spike rate [1/sec]')


def plot_timing(ax, data):
    data = {
        'Run': {
            'chip': 0.11,
            'readout': 0.034,
        },
        'Epoch': {
            'config': 0.065,
            'sampling1': 0.143,
            'sampling2': 0.145,
            'sampling3': 0.144,
            'eval': 0.06,
        },
        'Training': {
            'epoch1': 0.51,
            'epoch2': 0.508,
            'epoch3': 0.512,
        },
    }
    labelfontsize = 8
    ylabel = ['Run', 'Epoch', 'Training']
    ypos = np.arange(len(ylabel))


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time [s]')
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([-0.25, 0.75, 1.75])
    ax.set_yticklabels(ylabel)
    # bottom most

    ax.set_xticks([0.05, 0.1])
    ax.barh(-0.25, data['Run']['chip'], left=0, label='chip', color='C0', alpha=0.8, height=0.5)
    ax.barh(-0.25, data['Run']['readout'], left=data['Run']['chip'], label='readout', color='C1', alpha=0.8, height=0.5)
    ax.text(0.04, -0.32, 'chip', fontsize=labelfontsize)
    ax.text(0.12, -0.32, 'IO', fontsize=labelfontsize)


    # middle
    axins = ax.inset_axes([0., 0.33, 1., 0.67])
    axins.set_ylim(0.5, 2.5)
    axins.set_yticks([])
    axins.set_xticks([0.2, 0.4, 0.6])
    axins.spines['right'].set_visible(False)
    axins.spines['top'].set_visible(False)
    left = 0.
    axins.barh(0.75, data['Epoch']['config'], left=left, color='C2', alpha=0.8, height=0.5, edgecolor='C2')
    left += data['Epoch']['config']
    axins.barh(0.75, data['Epoch']['sampling1'], left=left, color='C3', alpha=0.8, height=0.5, edgecolor='gray')
    left += data['Epoch']['sampling1']
    axins.barh(0.75, data['Epoch']['sampling2'], left=left, color='C3', alpha=0.8, height=0.5, edgecolor='gray')
    left += data['Epoch']['sampling2']
    axins.barh(0.75, data['Epoch']['sampling3'], left=left, color='C3', alpha=0.8, height=0.5, edgecolor='gray')
    left += data['Epoch']['sampling3']
    axins.barh(0.75, data['Epoch']['eval'], left=left, color='C4', alpha=0.8, height=0.5, edgecolor='C4')
    axins.text(0.001, 0.61, 'con', fontsize=labelfontsize, rotation=30.)
    axins.text(0.21, 0.61, 'Run', fontsize=labelfontsize, rotation=30.)
    axins.text(0.44, 0.61, 'eval', fontsize=labelfontsize, rotation=30.)

    axinsins = axins.inset_axes([0., 0.5, 1., 0.5])
    axinsins.set_ylim(1.5, 2.5)
    axinsins.set_xticks([0.5, 1., 1.5])
    axinsins.spines['right'].set_visible(False)
    axinsins.spines['top'].set_visible(False)
    left = 0.
    axinsins.barh(1.75, data['Training']['epoch1'], left=left, color='C5', alpha=0.8, height=0.5, edgecolor='gray')
    left += data['Training']['epoch1']
    axinsins.barh(1.75, data['Training']['epoch2'], left=left, color='C5', alpha=0.8, height=0.5, edgecolor='gray')
    left += data['Training']['epoch2']
    axinsins.barh(1.75, data['Training']['epoch3'], left=left, color='C5', alpha=0.8, height=0.5, edgecolor='gray')
    axinsins.text(0.5, 1.55, 'epoch', fontsize=labelfontsize, rotation=30.)
    axinsins.set_yticks([])



def main():
    data_activation = json.load(open('figure2_activation.data'))
    # dict_keys(['biases', 'nspikes', 'b0s', 'alphas', 'taurefs'])
    data_weight = np.load('figure2_weights.npy')

    fig = plt.figure(figsize=(5, 2.2))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, height_ratios=[1.0], width_ratios=[1., 0.6])
    ax_matrix = fig.add_subplot(spec[0, 0])
    ax_timing = fig.add_subplot(spec[0, 1])

    plot_matrix(ax_matrix, data_weight)
    plot_timing(ax_timing, {})

    plt.tight_layout()
    plt.savefig('figure2.pdf')

if __name__ == '__main__':
    main()

