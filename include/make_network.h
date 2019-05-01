#ifndef MAKE_NETWORK_H
#define MAKE_NETWORK_H

#include <vector>
#include "poisson_noise.h"

void make_connections_source2target(int N_source, int N_target, double p, double Gmax, std::vector<std::vector<int>>& targets_ID, std::vector<std::vector<double>>& weights, Poisson_noise* noise_generator); // make connections from one type of neurons to another one with probability p

void make_chain(int N, int num_neurons_in_layer, double p, double Gmax, std::vector<std::vector<int>>& targets_ID, std::vector<std::vector<double>>& weights, Poisson_noise* noise_generator); // make chain with num_neurons_in_layer neurons and probability to connect neurons in the next layer p

#endif
