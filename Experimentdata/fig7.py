import sys
sys.path.append("../CPU_simulations")
import TFIM

import json
import numpy as np
import matplotlib.pyplot as plt


## settings
path = "data_figure7"
Nc = 8

fields = [
0.1,
0.5,
0.9,
1.0,
1.25,
5.0,
10.0
]

## corresponds to fields (one file per data point)
files = [
        "/run_emintest_210527-214506_{}.{}",
        "/run_emintest_210528-014021_{}.{}",
        "/run_emintest_210526-091457_{}.{}",
        "/run_seq2_bin_nhid40_reps10_noiseweight15_isyn500_lr1_winit40_W74F3_{}.{}",
        "/run_reps10_nhid30_{}.{}",
        "/run_emintest_210528-122128_{}.{}",
        "/run_fourpoints_nhid30_reps10_lr1_0.999_{}.{}"
         ]

## load data
ps = []
energies = []
infids = []
targets = []; target_probs = []
settings = []

for hi, h in enumerate(fields):
    print("h: ",h)
    if h == 1.0:
        folder = "/{}_N{}_B{}".format("TFIM", Nc, int(h))
    else:
        folder = "/{}_N{}_B{}".format("TFIM", Nc, h)
    # define paths
    fil = files[hi] 
    # load prob distr
    print("loading from "+path + folder + fil.format("p", "npy"))
    p = np.load(path + folder + fil.format("p", "npy"))
    if h == 0.1:
        p = p[:-150,:]
    ps.append(p)
    # load dkl
    #dkl = np.load(path + folder + fil.format("dkl", "npy"))
    #dkls.append(dkl)
    # load energy
    energy = np.load(path + folder + fil.format("energy", "npy"))
    energies.append(energy)
    # load settings
    try:
        with open(path + folder + fil.format("settings", "json")) as sets_file:
            sets = json.load(sets_file)
            settings.append(sets)
    except FileNotFoundError:
        settings.append("no setting file")

    e0, target_state = TFIM.exact_diag(Nc,h,1)
    target_state = target_state[0,:]
    target_prob = np.abs(target_state)**2
    targets.append(target_state)
    target_probs.append(target_prob)

    # load infid
    infid = 1 - np.sqrt(np.abs(np.sqrt(p) @ target_state))
    if h == 0.1:
        infid = infid[:-150]
    infids.append(infid)

##
ezeros = np.array([settings[si]["ED_energy"] for si in range(len(settings))])

## extract certain range of epochs over which to average
perc = 15
meanint = np.s_[-200:]
meaninte = np.s_[-350:-150]
ediff = np.array([np.abs(energies[i][meaninte] - ezeros[i])/np.abs(ezeros[i]) for i in range(len(energies))])
infids = np.array([infids[i][meanint] for i in range(len(infids))])

energy_medians = np.median(ediff[:,meaninte], axis=1)
energy_low = np.percentile(ediff[:,meaninte], q=perc, axis=1)
energy_up = np.percentile(ediff[:,meaninte], q=100 - perc, axis=1)

infid_means = infids[:,meanint].mean(axis=1)
infid_stds = infids[:,meanint].std(axis=1)
infid_medians = np.median(infids[:,meanint], axis=1)
infid_low = np.percentile(infids[:,meanint], q=perc, axis=1)
infid_up = np.percentile(infids[:,meanint], q=100 - perc, axis=1)

## energies plot
fig = plt.figure()
plt.errorbar(range(len(fields)), energy_medians, yerr=[energy_medians - energy_low, energy_up - energy_medians], label="data", marker="o", linestyle="")
plt.xticks(range(len(fields)))
fig.gca().set_xticklabels([str(f) for f in fields])
plt.xlabel("$h/J$")
plt.ylabel("$|E - E_0|/|E_0|$")
plt.grid()
plt.yscale("log")
plt.savefig("hfields_energies_N{}.pdf".format(Nc), bbox_inches="tight")
#plt.show()

## fidelities plot
fig = plt.figure()
plt.errorbar(range(len(fields)), infid_medians, yerr=[infid_medians - infid_low, infid_up - infid_medians], label="data", marker="o", linestyle="")
plt.xticks(range(len(fields)))
fig.gca().set_xticklabels([str(f) for f in fields])
plt.xlabel("$h/J$")
plt.ylabel("$1 - F$")
plt.grid()
plt.yscale("log")
plt.savefig("hfields_infidelities_N{}.pdf".format(Nc), bbox_inches="tight")
#plt.show()

