#include <iostream>

#include "main.h"
#include "configOutput.h"


ConfigOutput::ConfigOutput(int64_t nneurons) {
    // generate ConfigOutput with sensible defaults
    outputEnv = true;
    outputIndexes = {};
    for (auto i = 0; i < nneurons; ++i)
    {
        outputIndexes.push_back(i);
    }
    outputTimes.push_back(1L<<63);
}


void ConfigOutput::updateConfig(YAML::Node configOutputNode) {
    if (configOutputNode["outputIndexes"] && (configOutputNode["outputIndexes"].size() > 0)) {
        outputIndexes.clear();
        for (auto it = configOutputNode["outputIndexes"].begin(); it != configOutputNode["outputIndexes"].end(); ++it)
        {
            outputIndexes.push_back(it->as<int64_t>());
        }
    }
    if (configOutputNode["outputTimes"] && (configOutputNode["outputTimes"].size() > 0)) {
        outputTimes.clear();
        for (auto it = configOutputNode["outputTimes"].begin(); it != configOutputNode["outputTimes"].end(); ++it)
        {
            outputTimes.push_back(it->as<int64_t>());
        }
    }
}
