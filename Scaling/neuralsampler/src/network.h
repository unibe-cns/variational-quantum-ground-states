#ifndef NETWORK_H
#define NETWORK_H

#include <ostream>
#include <vector>
#include <random>
#include <algorithm>

extern std::mt19937_64 mt_random;
extern std::uniform_real_distribution<double> random_double;
extern std::normal_distribution<double> random_normal;

#include "config.h"

class Network
{
public:
    Network(const std::vector<double> &_biases,
            const std::vector<std::vector<double> > &_weights,
            const std::vector<int64_t> &_initialstate);
    ~Network() {};
    
    
    const std::vector<double> biases;
    const std::vector<std::vector<double> > weights;
    std::vector<std::vector<int64_t> > connected_neuron_ids;

    std::vector<int64_t> state;
//    std::vector<int64_t> nspikes;

    int64_t update_single_state(const double pot);
//    double activation(const double pot);

    void generate_connected_neuron_ids();
    void produce_header(std::ostream& stream);
    void produce_output(std::ostream& stream);
    void produce_summary(std::ostream& stream);
    void update_state();
};



#endif // NETWORK_H

