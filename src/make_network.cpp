#include "make_network.h"
#include <iostream>

void make_connections_source2target(int N_source, int N_target, double p, double Gmax, std::vector<std::vector<int>>& targets_ID, std::vector<std::vector<double>>& weights, Poisson_noise* noise_generator)
{
	targets_ID.resize(N_source);
	weights.resize(N_source);

	for (int i = 0; i < N_source; i++)
	{
		for (int j = 0; j < N_target; j++)
		{
			if (noise_generator->random(1.0) < p)
			{
				targets_ID[i].push_back(j);
				weights[i].push_back(noise_generator->random(Gmax));
			}
		}
	}
}

void make_chain(int N, int num_neurons_in_layer, double p, double Gmax, std::vector<std::vector<int>>& targets_ID, std::vector<std::vector<double>>& weights, Poisson_noise* noise_generator)
{
	if (N % num_neurons_in_layer != 0)
		std::cerr << " Number of neurons is not multiple of number of neurons in layer! Some neurons remain unconnected\n" << std::endl;

	targets_ID.resize(N);
	weights.resize(N);

	int num_layers = N / num_neurons_in_layer;

	for (int i = 0; i < num_layers-1; i++)
	{
		for (int j = 0; j < num_neurons_in_layer; j++)
		{
			for (int k = 0; k < num_neurons_in_layer; k++)
			{
				if (noise_generator->random(1.0) < p)
				{
					targets_ID[i*num_neurons_in_layer+j].push_back((i+1)*num_neurons_in_layer+k);
					weights[i*num_neurons_in_layer+j].push_back(noise_generator->random(Gmax));
				}
			}
		}
	}
}

