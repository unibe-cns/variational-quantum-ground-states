#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "config.h"
#include "neuron.h"
#include "myrandom.h"

SCENARIO("Config constructor") {

    GIVEN("Default config") {
        Config config = Config(10);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }
}

SCENARIO("Update config") {

    GIVEN("set randomSeed") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("randomSeed: 42");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set randomSkip") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("randomSkip: 42");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 42 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set nupdates") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("nupdates: 42");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 42 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set tauref") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("tauref: 42");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 42 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set tausyn") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("tausyn: 42");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 42 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set delay") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("delay: 42");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 42 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    // GIVEN("set neuronType to fail") {
    //     Config config = Config();
    //     YAML::Node node = YAML::Load("neuronType: blub");
    //     CHECK_THROWS(config.updateConfig(node));
    // }

    GIVEN("set neuronType") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("neuronType: erf");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Erf );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set neuronType") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("neuronType: log");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    // GIVEN("set interactionType to fail") {
    //     Config config = Config();
    //     YAML::Node node = YAML::Load("synapseType: 42");
    //     REQUIRE_THROWS(config.updateConfig(node));
    // }

    GIVEN("set interactionType") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("synapseType: exp");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Exp );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set interactionType") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("synapseType: cuto");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Cuto );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set interactionType") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("synapseType: tail");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Tail );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set updateScheme") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("networkUpdateScheme: BatchRandom");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == BatchRandom );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set updateScheme") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("networkUpdateScheme: Random");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == Random );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set updateScheme") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("output: {outputScheme: MeanActivityEnergy}");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityEnergyOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set updateScheme") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("output: {outputScheme: BinaryState}");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == BinaryStateOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set updateScheme") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("output: {outputScheme: Spikes}");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == SpikesOutput );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set updateScheme") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("output: {outputScheme: SummarySpikes}");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == SummarySpikes );
        REQUIRE( config.output.outputEnv == true );
    }

    GIVEN("set updateScheme") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("output: {outputEnv: false}");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == false );
    }

    GIVEN("set neuronUpdate") {
        Config config = Config(10);
        YAML::Node node = YAML::Load("neuronUpdate: {theta: 2.0}");
        config.updateConfig(node);

        REQUIRE( config.randomSeed == 42424242 );
        REQUIRE( config.randomSkip == 1000000 );
        REQUIRE( config.nupdates == 100000 );
        REQUIRE( config.tauref == 1 );
        REQUIRE( config.tausyn == 1 );
        REQUIRE( config.delay == 1 );
        REQUIRE( config.neuronActivationType == Log );
        REQUIRE( config.neuronInteractionType == Rect );
        REQUIRE( config.updateScheme == InOrder );
        REQUIRE( config.output.outputScheme == MeanActivityOutput );
        REQUIRE( config.output.outputEnv == true );
        REQUIRE( config.neuronUpdate.theta == 2. );
    }

}
