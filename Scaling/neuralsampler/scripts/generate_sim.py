import json
import numpy as np

config = {
        'Config': {
            'randomSeed': 42424242,
            'randomSkip': 1000000,
            'nupdates': 1000000,
            'tauref': 1,
            'tausyn': 1,
            'delay': 1,
            'subsampling': 1,
            'NeuronType': 'log',
            'synapseType': 'exp',
            'networkUpdateScheme': 'InOrder',
            'output': {
                # 'outputScheme': 'SummarySpikes',  # MeanActivityEnergy, BinaryState, InternalStateOutput, Spikes, SummarySpikes, SummaryStates
                'outputScheme': 'BinaryState',  # MeanActivityEnergy, BinaryState, InternalStateOutput, Spikes, SummarySpikes, SummaryStates
                'outputIndexes': [],            # if exists and non-empty: output only these indices
                'outputTimes': [],              # if exists and non-empty: output only at these times
                'outputEnv': True
                }
            # 'neuronIntegrationType': MemoryLess,      # OU integration
            # 'neuronUpdate': {'theta': 0.1, 'mu': 0.0, 'sigma': 1.0}
            },
        'bias': None,           # vector of floats
        'weight': None,         # vector of (preind: int, postind: int, weight: float)
        'initialstate': None,   # vector of ints
        'temperature': {
            'type': 'Const',    # 'Const' or 'Linear'
            'times': [0, 11111111],
            'values': [1.0, 1.0]
            },
        'externalCurrent': {
            'type': 'Const',    # 'Const' or 'Linear'
            'times': [0, 11111111],
            'values': [0.0, 0.0]
            },
        'outfile': 'out'
        }


def generate_parameterfile(outputdict, parameterfile):
    data = json.load(open(parameterfile))
    bias = data['biases']
    weight = data['weights']

    outputdict['bias'] = bias
    outputdict['weight'] = []
    for i, wline in enumerate(weight):
        for j, w in enumerate(wline):
            if w!=0.:
                outputdict['weight'].append((i, j, float(w)))
    outputdict['initialstate'] = np.random.randint(0, 2, size=len(bias)).tolist()

    json.dump(outputdict, open('sim.json', 'w'))


def generate_rbm_weights_bias(outputdict, nvis, nhid):
    bias = np.random.normal(0., 1., size=(nvis+nhid))
    weight = np.zeros((nvis+nhid, nvis+nhid))
    weight[:nvis, nvis:] = np.random.normal(0., 1., size=(nvis, nhid))
    weight += weight.T

    outputdict['bias'] = bias.tolist()
    outputdict['weight'] = []
    for i, wline in enumerate(weight):
        for j, w in enumerate(wline):
            if w!=0.:
                outputdict['weight'].append((i, j, float(w)))
    outputdict['initialstate'] = np.random.randint(0, 2, size=(nvis+nhid)).tolist()

    json.dump(outputdict, open('sim.json', 'w'))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nvis', type=int, default=None)
parser.add_argument('--nhid', type=int, default=None)
parser.add_argument('--parameterfile', type=str, default=None)

args = parser.parse_args()

if args.parameterfile is not None:
    generate_parameterfile(config, args.parameterfile)
else:
    generate_rbm_weights_bias(config, args.nvis, args.nhid)
