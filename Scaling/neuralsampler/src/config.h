#ifndef CONFIG_H
#define CONFIG_H

#include <ostream>
#include <vector>
#include <random>
#include <algorithm>

#include "configOutput.h"
#include "main.h"

class Config
{
public:
    int64_t randomSeed;
    int64_t randomSkip;
    int64_t nupdates;
    int64_t nneurons;
    ConfigOutput output;

    Config(int64_t nneurons);
    ~Config() {};

    void updateConfig(YAML::Node ConfigNode);
};


#endif // CONFIG_H

