#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "neuron.h"
#include "myrandom.h"

SCENARIO("Neuron constructor") {

    GIVEN("Rectangular neuron - active") {
        Neuron n(100, 100, 1, 99, Log, Rect);

        REQUIRE( n.get_internalstate()==99 );
        REQUIRE( n.get_state()== 1);
        REQUIRE( n.get_interaction() == 1. );
        REQUIRE( n.activation(0.) == 0.5 );
        REQUIRE( n.activation(0.- std::log(100)) == Approx(0.0099009900990098976).epsilon(0.000001) );
        REQUIRE( n.activation(1.) == Approx(0.7310585786).epsilon(0.000001));
        REQUIRE( n.activation(-1.) == Approx(0.2689414214).epsilon(0.000001));

        WHEN("State is updated - impossible to spike") {
            n.update_state(-50.);
            REQUIRE( n.get_internalstate() == 100 );
            REQUIRE( n.get_state() == 0 );
            REQUIRE( n.get_interaction() == 1. );
            n.update_interaction();
            REQUIRE( n.get_interaction() == 0. );
        }
        WHEN("State is updated - inevitable to spike") {
            n.update_state(50.);
            REQUIRE( n.get_internalstate() == 0 );
            REQUIRE( n.get_state() == 1 );
            REQUIRE( n.get_interaction() == 1. );
            n.update_interaction();
            REQUIRE( n.get_interaction() == 1. );
        }
    }

    GIVEN("Exponential neuron - active") {
        Neuron n(100, 100, 1, 99, Log, Exp);

        REQUIRE( n.get_internalstate()==99 );
        REQUIRE( n.get_state()== 1);
        double f_int = std::exp(-0.99)/(1.-std::exp(-1.));
        REQUIRE( n.get_interaction() == f_int );
        REQUIRE( n.activation(0.) == 0.5 );
        REQUIRE( n.activation(1.) == Approx(0.7310585786).epsilon(0.000001));
        REQUIRE( n.activation(-1.) == Approx(0.2689414214).epsilon(0.000001));

        WHEN("State is updated - impossible to spike") {
            n.update_state(-50.);
            REQUIRE( n.get_internalstate() == 100 );
            REQUIRE( n.get_state() == 0 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = std::exp(-1.)/(1.-std::exp(-1.));
            REQUIRE( n.get_interaction() == f_int );
        }
        WHEN("State is updated - inevitable to spike") {
            n.update_state(50.);
            REQUIRE( n.get_internalstate() == 0 );
            REQUIRE( n.get_state() == 1 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = 1./(1.-std::exp(-1.));
            REQUIRE( n.get_interaction() == f_int );
        }

    }

    GIVEN("Tail neuron - active") {
        Neuron n(100, 100, 1, 99, Log, Tail);

        REQUIRE( n.get_internalstate()==99 );
        REQUIRE( n.get_state()== 1);
        double f_int = 1.;
        REQUIRE( n.get_interaction() == f_int );
        REQUIRE( n.activation(0.) == 0.5 );
        REQUIRE( n.activation(1.) == Approx(0.7310585786).epsilon(0.000001));
        REQUIRE( n.activation(-1.) == Approx(0.2689414214).epsilon(0.000001));

        WHEN("State is updated - impossible to spike") {
            n.update_state(-50.);
            REQUIRE( n.get_internalstate() == 100 );
            REQUIRE( n.get_state() == 0 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = std::exp(-1.)/(1.-std::exp(-1.));
            REQUIRE( n.get_interaction() == f_int );
        }
        WHEN("State is updated - inevitable to spike") {
            n.update_state(50.);
            REQUIRE( n.get_internalstate() == 0 );
            REQUIRE( n.get_state() == 1 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = 1.;
            REQUIRE( n.get_interaction() == f_int );
        }

    }

    GIVEN("Cuto neuron - active") {
        Neuron n(100, 100, 1, 99, Log, Cuto);

        REQUIRE( n.get_internalstate()==99 );
        REQUIRE( n.get_state()== 1);
        double f_int = std::exp(-0.99)/(1.-std::exp(-1.));
        REQUIRE( n.get_interaction() == f_int );
        REQUIRE( n.activation(0.) == 0.5 );
        REQUIRE( n.activation(1.) == Approx(0.7310585786).epsilon(0.000001));
        REQUIRE( n.activation(-1.) == Approx(0.2689414214).epsilon(0.000001));

        WHEN("State is updated - impossible to spike") {
            n.update_state(-50.);
            REQUIRE( n.get_internalstate() == 100 );
            REQUIRE( n.get_state() == 0 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = 0.;
            REQUIRE( n.get_interaction() == f_int );
        }
        WHEN("State is updated - inevitable to spike") {
            n.update_state(50.);
            REQUIRE( n.get_internalstate() == 0 );
            REQUIRE( n.get_state() == 1 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = 1./(1.-std::exp(-1.));
            REQUIRE( n.get_interaction() == f_int );
        }

    }

        GIVEN("Rectangular erf neuron - active") {
        Neuron n(100, 100, 1, 99, Erf, Rect);

        REQUIRE( n.get_internalstate()==99 );
        REQUIRE( n.get_state()== 1);
        REQUIRE( n.get_interaction() == 1. );
        REQUIRE( n.activation(0.) == 0.5 );
        REQUIRE( n.activation(1.) == Approx(0.7219864828).epsilon(0.000001));
        REQUIRE( n.activation(-1.) == Approx(0.2780135172).epsilon(0.000001));

        WHEN("State is updated - impossible to spike") {
            n.update_state(-50.);
            REQUIRE( n.get_internalstate() == 100 );
            REQUIRE( n.get_state() == 0 );
            REQUIRE( n.get_interaction() == 1. );
            n.update_interaction();
            REQUIRE( n.get_interaction() == 0. );
        }
        WHEN("State is updated - inevitable to spike") {
            n.update_state(50.);
            REQUIRE( n.get_internalstate() == 0 );
            REQUIRE( n.get_state() == 1 );
            REQUIRE( n.get_interaction() == 1. );
            n.update_interaction();
            REQUIRE( n.get_interaction() == 1. );
        }
    }

    GIVEN("Exponential erf neuron - active") {
        Neuron n(100, 100, 1, 99, Erf, Exp);

        REQUIRE( n.get_internalstate()==99 );
        REQUIRE( n.get_state()== 1);
        double f_int = std::exp(-0.99)/(1.-std::exp(-1.));
        REQUIRE( n.get_interaction() == f_int );
        REQUIRE( n.activation(0.) == 0.5 );
        REQUIRE( n.activation(1.) == Approx(0.7219864828).epsilon(0.000001));
        REQUIRE( n.activation(-1.) == Approx(0.2780135172).epsilon(0.000001));

        WHEN("State is updated - impossible to spike") {
            n.update_state(-50.);
            REQUIRE( n.get_internalstate() == 100 );
            REQUIRE( n.get_state() == 0 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = std::exp(-1.)/(1.-std::exp(-1.));
            REQUIRE( n.get_interaction() == f_int );
        }
        WHEN("State is updated - inevitable to spike") {
            n.update_state(50.);
            REQUIRE( n.get_internalstate() == 0 );
            REQUIRE( n.get_state() == 1 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = 1./(1.-std::exp(-1.));
            REQUIRE( n.get_interaction() == f_int );
        }

    }

    GIVEN("Tail erf neuron - active") {
        Neuron n(100, 100, 1, 99, Erf, Tail);

        REQUIRE( n.get_internalstate()==99 );
        REQUIRE( n.get_state()== 1);
        double f_int = 1.;
        REQUIRE( n.get_interaction() == f_int );
        REQUIRE( n.activation(0.) == 0.5 );
        REQUIRE( n.activation(1.) == Approx(0.7219864828).epsilon(0.000001));
        REQUIRE( n.activation(-1.) == Approx(0.2780135172).epsilon(0.000001));

        WHEN("State is updated - impossible to spike") {
            n.update_state(-50.);
            REQUIRE( n.get_internalstate() == 100 );
            REQUIRE( n.get_state() == 0 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = std::exp(-1.)/(1.-std::exp(-1.));
            REQUIRE( n.get_interaction() == f_int );
        }
        WHEN("State is updated - inevitable to spike") {
            n.update_state(50.);
            REQUIRE( n.get_internalstate() == 0 );
            REQUIRE( n.get_state() == 1 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = 1.;
            REQUIRE( n.get_interaction() == f_int );
        }

    }

    GIVEN("Cuto erf neuron - active") {
        Neuron n(100, 100, 1, 99, Erf, Cuto);

        REQUIRE( n.get_internalstate()==99 );
        REQUIRE( n.get_state()== 1);
        double f_int = std::exp(-0.99)/(1.-std::exp(-1.));
        REQUIRE( n.get_interaction() == f_int );
        REQUIRE( n.activation(0.) == 0.5 );
        REQUIRE( n.activation(1.) == Approx(0.7219864828).epsilon(0.000001));
        REQUIRE( n.activation(-1.) == Approx(0.2780135172).epsilon(0.000001));

        WHEN("State is updated - impossible to spike") {
            n.update_state(-50.);
            REQUIRE( n.get_internalstate() == 100 );
            REQUIRE( n.get_state() == 0 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = 0.;
            REQUIRE( n.get_interaction() == f_int );
        }
        WHEN("State is updated - inevitable to spike") {
            n.update_state(50.);
            REQUIRE( n.get_internalstate() == 0 );
            REQUIRE( n.get_state() == 1 );
            REQUIRE( n.get_interaction() == f_int );
            n.update_interaction();
            f_int = 1./(1.-std::exp(-1.));
            REQUIRE( n.get_interaction() == f_int );
        }

    }

}




