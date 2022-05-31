#include <iostream>

#include "main.h"
#include "config.h"


Config::Config(int64_t _nneurons):
    output(ConfigOutput(_nneurons))
{
    // generate Config with sensible defaults
    randomSeed = 42424242;
    randomSkip = 1000000;
    nupdates = 100000;
    nneurons = _nneurons;
}


void Config::updateConfig(YAML::Node configNode) {
    if (configNode["randomSeed"]) {
        randomSeed = configNode["randomSeed"].as<int64_t>();
    }
    if (configNode["randomSkip"]) {
        randomSkip = configNode["randomSkip"].as<int64_t>();
    }
    if (configNode["nupdates"]) {
        nupdates = configNode["nupdates"].as<int64_t>();
    }
    if (configNode["output"]) {
        output.updateConfig(configNode["output"]);
    }
}