#ifndef HVCI_H
#define HVCI_H
#include <curand.h>
#include <curand_kernel.h>
#
namespace hvcIConstants
{		
	const double cm = 1.0;	// micro F / cm2
	const double A = 6000;	// microns2

	const double Ena = 55.0;	// mV
	const double Ek = -80.0;
	const double El = -65.0;
	const double Ei = -75.0;

	const double gNa = 100.0;	// mS/cm2
	const double gKdr = 20.0;
	const double gKHT = 500.0;
	const double gL = 0.1;

	const double tExc = 2.0;	//	ms
	const double tInh = 5.0;

	// spike params

	const double THRESHOLD_SPIKE = -20.0;	//	mV
	const double SPIKE_MARGIN = 20.0; // mV
	
	const int MAX_NUM_OF_SPIKES  = 50;


	// noise parameters
	const double G_noise = 0.45;	//	maximum noise conductance
	const double lambda = 250.0; // intensity parameter for Poisson noise
}

__global__ void print_buffer_HVCI(double* buffer, int buffer_size);
__global__ void set_record_HVCI(bool* record, int neuron_id);


__global__ void initialize_noise_I(double *vars, int N, int seed, curandState* states);

__global__ void calculate_next_step_I(double *vars, bool* flags, int N, double timestep, bool* record, double* buffer, int step, double ampl, bool *spiked, int *num_spikes, double *spike_times, curandState *states);

__device__ inline double an(double v){return 0.15*(v + 15) / (1 - exp(-(v + 15) / 10));} // was 0.05; original value = 0.15
__device__ inline double bn(double v){return 0.2 * exp(-(v + 25) / 80);} // was 0.1; original value = 0.2
__device__ inline double am(double v){return (v + 22) / (1 - exp(-(v + 22) / 10));}
__device__ inline double bm(double v){return 40 * exp(-(v + 47) / 18);}
__device__ inline double ah(double v){return 0.7 * exp(-(v + 34) / 20);}
__device__ inline double bh(double v){return 10 / (1 + exp(-(v + 4) / 10));}
__device__ inline double wInf(double v){return 1 / (1 + exp(-v / 5));}
__device__ inline double tauW(double v){return 1;} // was 2; original value = 1

#endif
