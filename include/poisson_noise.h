#pragma once
#ifndef POISSON_NOISE_H
#define POISSON_NOISE_H
#endif

#include <random>
#include <tgmath.h>
#include <iostream>

class Poisson_noise
{
public:
	Poisson_noise() : d(0.0, 1.0), dis_int(0, 29), mean(0.0), sd(1.0){};
	void set_seed(unsigned s);	//	function to set seed to randon number generator
	double get_spike_time(double lambda);	//	get time for the next noisy spike
	double random(double G);	//	get random number in range (0; G)
	int sample_index_for_point_distribution(); // get index from 0 to 29 for point distribution
    int sample_integer(int min, int max){return std::uniform_int_distribution<int>{min, max}(generator);}; // sample integer number between min and max
	void set_normal_distribution(double mu, double sigma); // set parameters of normal distribution
	double normal_distribution(); // get number sampled from normal distribution
private:
	unsigned seed;	//	seed for random number generator
	std::mt19937 generator;	//	marsene-twister generator
	std::normal_distribution<double> d; // normal distribution with zero mean and unit variance
	std::uniform_int_distribution<int> dis_int; // integer distribution for sampling random numbers for Debraband step in HH2_final.cpp
    double mean; // mean for normal distribution
	double sd; // standard deviation for noraml distribution
};
