#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "main.h"
#include "myrandom.h"

#include "network.h"


std::vector<double> get_bias_from_node(YAML::Node biasNode)
{
    std::vector<double> bias;
    for(YAML::const_iterator it=biasNode.begin(); it!=biasNode.end(); it++) {
        bias.push_back(it->as<double>());
    }
    return bias;
}


std::vector<std::vector<double>> get_weights_from_node(YAML::Node weightNode, std::size_t biassize)
{
    // read sparse representation of weight matrix in
    std::vector<std::vector<double>> weights(biassize);
    for(std::size_t i = 0; i < biassize; ++i) {
        std::vector<double> weight_line(biassize);
        for(std::size_t j = 0; j < biassize; ++j) {weight_line[j] = 0.;}
        weights[i] = weight_line;
    }
    // translate in dense matrix
    int64_t i, j;
    double w;
    for(YAML::const_iterator it=weightNode.begin(); it!=weightNode.end(); it++) {
        i = (*it)[0].as<int64_t>();
        j = (*it)[1].as<int64_t>();
        w = (*it)[2].as<double>();
        weights[i][j] = w;
    }
    return weights;
}


std::vector<int64_t> get_initialstate_from_node(YAML::Node initialStateNode)
{
    std::vector<int64_t> initialstate;
    for(YAML::const_iterator it=initialStateNode.begin(); it!=initialStateNode.end(); it++) {
        initialstate.push_back(it->as<int64_t>());
    }
    return initialstate;
}



int main(int argc, char const *argv[])
{
    // report inputfilename and load the yaml content
    if (argc<2) {
        std::cout << "This is a neuralsampler, implemented by Andreas Baumbach, modelled after Buesing et al, 2013" << std::endl;
        return -1;
    }
    YAML::Node baseNode = YAML::LoadFile(argv[1]);
    YAML::Node configNode = baseNode["Config"];
    YAML::Node biasNode = baseNode["bias"];
    if (!biasNode) {
        YAML::Node biasFileNode = baseNode["biasFile"];
        if (!biasFileNode) {
            biasNode = YAML::LoadFile(biasFileNode.as<std::string>());
        } else {
            std::cout << "Didn't find either bias or biasFile. Aborting!";
            throw;
        }
    }
    YAML::Node weightNode = baseNode["weight"];
    if (!weightNode) {
        YAML::Node weightFileNode = baseNode["weightFile"];
        if (!weightFileNode) {
            weightNode = YAML::LoadFile(weightFileNode.as<std::string>());
        } else {
            std::cout << "Didn't find either weight or weightFile. Aborting!";
            throw;
        }
    }
    YAML::Node initialStateNode = baseNode["initialstate"];
    if (!initialStateNode) {
        YAML::Node initialStateFileNode = baseNode["initialstateFile"];
        if (!initialStateNode) {
            initialStateNode = YAML::LoadFile(initialStateFileNode.as<std::string>());
        } else {
            std::cout << "Didn't find either initialstate or initialstateFile. Aborting!";
            throw;
        }
    }

    YAML::Node simulationFolderNode = baseNode["outfile"];
    bool b_output_file = baseNode["outfile"].IsDefined();

    // get network configuration
    std::vector<double> bias = get_bias_from_node(biasNode);
    std::vector<std::vector<double>> weights =
                        get_weights_from_node(weightNode, bias.size());
    std::vector<int64_t> initialstate =
                        get_initialstate_from_node(initialStateNode);

    // get general configuration
    Config config = Config(bias.size());
    config.updateConfig(configNode);

    std::streambuf *buf;
    std::ofstream of;
    if (b_output_file) {
        of.open(simulationFolderNode.as<std::string>() ) ;
        buf = of.rdbuf();;
    } else {
        buf = std::cout.rdbuf();
    }
    std::ostream output(buf);

    Network net(bias, weights, initialstate);

    // seed random number generator and discard for higher entropy
    mt_random.seed(config.randomSeed);
    mt_random.discard(config.randomSkip);

    // actual simulation
    for (int64_t i = 0; i < config.nupdates; ++i)
    {
        net.update_state();
    }
    output << "____End of simulation____\n\n\n";
    net.produce_summary(output);
    output.flush();
    of.close();

    return 0;
}






