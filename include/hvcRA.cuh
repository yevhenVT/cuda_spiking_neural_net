#ifndef HVCRA_H
#define HVCRA_H
#include <curand.h>
#include <curand_kernel.h>

namespace hvcRaConstants
{
	const double Gs_noise_inh = 0.035;
	const double Gd_noise_inh = 0.045;
	const double Gs_noise_exc = 0.035;
	const double Gd_noise_exc = 0.045;

	// noise spike frequencies
	const double lambda_exc = 100.0;
	const double lambda_inh = 100.0;

	// neuron model parameters
	const double cm = 1;
	const double Rc = 55; // original 55 coupling between dendrite and soma in MOhms 
	const double As = 5000; // soma surface area in micro meters squared
	const double GsL = 0.1;
	const double GsNa = 60;
	const double GsK = 8;
	const double EsL = -80;
	const double EsNa = 55;
	const double EsK = -90;
	const double Ad = 10000;
	const double GdL = 0.1;
	const double GdCa = 55; // original 55
	const double GdCaK = 150; // original 150
	const double EdL = -80;
	const double EdCa = 120;
	const double EdK = -90;
	const double Egaba = -80;
	const double tExc = 5;
	const double tInh = 5;

	// thresholds
	const double threshold_spike = 0;
	const double threshold_burst = 0;
	const double spike_margin = 5.0;
	const double burst_margin = 5.0;	
}

__global__ void print_buffer_HVCRA(double* buffer, int buffer_size);
__global__ void set_record_HVCRA(bool* record, int neuron_id);


__global__ void initialize_noise_RA(double *vars, int N, int seed, curandState* states);

__global__ void calculate_next_step_RA(double *vars, bool *flags, int N, double timestep, bool* record, double* buffer, int step, double ampl_s, double ampl_d, int training, bool* spiked, int* num_spikes, double* spike_times, curandState *states);

__device__ inline double nInf(double v){return 1 / (1 + exp(-(v + 35) / 10));} // +35
__device__ inline double tauN(double v){return 0.1 + 0.5 / (1 + exp((v + 27) / 15));} //+27
__device__ inline double hInf(double v){return 1 / (1 + exp((v + 45) / 7));} //+45
__device__ inline double tauH(double v){return 0.1 + 0.75 / (1 + exp((v + 40.5) / 6));} //+40.5
__device__ inline double mInf(double v){return 1 / (1 + exp(-(v + 30) / 9.5));} //+30
__device__ inline double rInf(double v){return 1 / (1 + exp(-(v + 5) / 10));} //+15
__device__ inline double tauR(double v){return 1;}
__device__ inline double cInf(double v){return 1 / (1 + exp(-(v - 10) / 7));} // -0
__device__ inline double tauC(double v){return 10;}

#endif
