#include "poisson_noise.h"

void Poisson_noise::set_seed(unsigned s)
{
	seed  = s;
	generator.seed(seed);
}

double Poisson_noise::get_spike_time(double lambda)
{
	double random;
	
	random = (double)generator() / generator.max();	//	create random real number from 0 to 1;
	
	while (random >=1.0)
		random = (double)generator() / generator.max();	//	create random real number from 0 to 1;
	
	return (- log(1.0 - random) / lambda);
	
}

double Poisson_noise::random(double G)
{
	return (double) G * generator() / generator.max();
}

int Poisson_noise::sample_index_for_point_distribution()
{
	return dis_int(generator);
}

void Poisson_noise::set_normal_distribution(double mu, double sigma)
{
	mean = mu;
	sd = sigma;
}

double Poisson_noise::normal_distribution()
{
	return mean + d(generator)*sd;
}

