import sys
sys.path.append("../CPU_simulations")
import TFIM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# calculate exact ground state
e0, psi0 = TFIM.exact_diag(8, 0.1,1)
psi0 = psi0[0,:]

# load BSS2 data
#p01sb = np.load("run_emintest_210531-123116_p.npy") # bshift = 1
p01_sblow = np.load("run_emintest_210531-133933_p.npy") # bshift = 2
p01_sbhigh = np.load("run_emintest_210527-214506_p.npy") # bshift = 0

# calculate p(v) histograms
p01_mean_low = p01_sblow[280:300,:].mean(axis=0)
p01_mean_low_nonzero = np.nonzero(p01_mean_low)[0]
p01_mean_high = p01_sbhigh[280:300,:].mean(axis=0)
p01_mean_high_nonzero = np.nonzero(p01_mean_high)[0]

# plot
plt.figure()
plt.plot(range(p01_sbhigh.shape[1]),np.abs(psi0)**2, label="exact", linestyle="-", marker="", linewidth=1, color="black")
plt.plot(range(p01_mean_low.shape[0]), p01_mean_low, label="low activity", linestyle=":", marker=".", color="tab:red", alpha=0.5)
plt.plot(range(p01_mean_high.shape[0]), p01_mean_high, label="high activity", linestyle=":", marker=".", color="tab:blue", alpha=0.5)
#plt.ylim([1e-8, 1.3])
plt.legend()
plt.yscale("log")
plt.xlabel("v")
plt.ylabel("p(v)")
plt.savefig("symmetrybreaking_mean.pdf", bbox_inches="tight")
