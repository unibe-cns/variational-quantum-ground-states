#ifndef MYRANDOM_H
#define MYRANDOM_H

#include <random>

std::mt19937_64 mt_random;
std::uniform_real_distribution<double> random_double(0.0, 1.0);
std::normal_distribution<double> random_normal(0.0, 1.0);
#endif // MYRANDOM_H
