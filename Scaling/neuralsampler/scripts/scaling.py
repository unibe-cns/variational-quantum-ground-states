import subprocess
import time
import json

out = {}

for nvis in [4, 8, 12, 16, 20, 50, 100]:
    for nhid in [20, 40, 80]:
        subprocess.check_call(['python', 'generate_sim.py', '--nvis', str(nvis), '--nhid', str(nhid)])
        t0 = time.time()
        subprocess.check_call(['../bin/neuralsampler', 'sim.json'])
        t1 = time.time()
        print('nvis: {nvis}, nhid: {nhid}, time: {time}'.format(nvis=nvis, nhid=nhid, time=t1-t0))
        out['{nvis}_{nhid}'.format(nvis=nvis, nhid=nhid)] = t1-t0

json.dump(out, open('timings.json', 'w'))


