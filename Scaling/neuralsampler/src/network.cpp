#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <random>
#include <algorithm>

#include "network.h"


double activation(const double pot)
{
    return 1./(1.+std::exp(-pot));
}

Network::Network(const std::vector<double> &_biases,
                 const std::vector<std::vector<double> > &_weights,
                 const std::vector<int64_t> &_initialstate):
    biases(_biases),
    weights(_weights)
{

    state.resize(biases.size());
   // nspikes.resize(biases.size());
    for (std::size_t i = 0; i < biases.size(); ++i)
    {
        state[i] = _initialstate[i];
   //     nspikes[i] = 0;
    }

    generate_connected_neuron_ids();  // TODO: make const
}

void Network::generate_connected_neuron_ids()
{
    std::size_t n_connections = 0;
    connected_neuron_ids.resize(biases.size());
    for (std::size_t i = 0; i < biases.size(); ++i)
    {
        for (std::size_t j = 0; j < biases.size(); ++j)
        {
            if (fabs(weights[i][j])>1E-14)
            {
                connected_neuron_ids[i].push_back(j);
                n_connections++;
            }
        }
    }
}

void Network::produce_header(std::ostream& stream)
{
    stream << "# using sparse connectivity: yes\n";
}

void Network::produce_summary(std::ostream& stream)
{
    stream.fill('0');
    stream << "Summary:\n";
    stream << "NeuronNr NumberOfSpikes\n-----------\n";
  //  for (std::size_t i = 0; i < biases.size(); ++i) {
  //      stream << std::setw(5) << i << " " << std::setw(12) << nspikes[i] << "\n";
  //  }
    stream << "Internalstate:\n";
    for (std::size_t i = 0; i < biases.size(); ++i) {
        stream << std::setw(5) << i << " " << std::setw(5) << state[i] << "\n";
    }
}

void Network::update_state()
{
    // update neurons in sequence determined above
    for (std::size_t neuronid = 0; neuronid < biases.size(); ++neuronid)
    {
        double pot = biases[neuronid];  
        for (auto conid = connected_neuron_ids[neuronid].begin();
             conid != connected_neuron_ids[neuronid].end();
             ++conid)
        {
            pot += state[*conid] * weights[neuronid][*conid];
        }
        auto newstate = update_single_state(pot);
        state[neuronid] = newstate;
//     nspikes[neuronid] += newstate;
    }
}

int64_t Network::update_single_state(const double pot)
{
    double r = random_double(mt_random);
    int bspike = activation(pot) > r;
    return bspike;
}
