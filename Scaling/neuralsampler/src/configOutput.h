#ifndef CONFIG_OUTPUT_H
#define CONFIG_OUTPUT_H

#include <ostream>
#include <vector>

#include "main.h"

class ConfigOutput
{
public:
    bool outputEnv;
    std::vector<int64_t> outputIndexes;
    std::vector<int64_t> outputTimes;

    ConfigOutput(int64_t nneurons);
    ~ConfigOutput() {};

    void updateConfig(YAML::Node ConfigOutputNode);
};


#endif // CONFIG_OUTPUT_H