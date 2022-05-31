#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "main.h"
#include "myrandom.h"

#include "network.h"
#include "config.h"

SCENARIO("Network constructor") {

    GIVEN("3 Neuron network") {
        std::vector<double> biases = {0., 1., -1.};
        std::vector< std::vector<double> > weights = {
            {0, 1, 0.5},
            {1, 0, 0.3},
            {2, 1, 0.}};
        std::vector<int64_t> initialstate = {0, 4, 8};
        Config config = Config(3);
        YAML::Node node = YAML::Load("tauref: 5\noutput: {outputIndexes: [0, 1, 2]}");
        config.updateConfig(node);

        Network n(biases, weights, initialstate, config);
        REQUIRE( n.states[0] == 1 );
        REQUIRE( n.states[1] == 1 );
        REQUIRE( n.states[2] == 0 );
        n.get_internalstate();
        REQUIRE( n.states[0] == 0 );
        REQUIRE( n.states[1] == 4 );
        REQUIRE( n.states[2] == 8 );

    }


}
