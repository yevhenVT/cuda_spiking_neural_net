#include <iostream>
#include <stdio.h>
#include "hvcRA.cuh"
#include <curand.h>
#include <curand_kernel.h>

#define MAX_NUM_OF_SPIKES 50

using namespace hvcRaConstants;

typedef double (*current)(double);

const static double THRESHOLD_SPIKE = 0.0;
const static double SPIKE_MARGIN = 20.0;
const static double THRESHOLD_BURST = 0.0;


__global__ void print_buffer_HVCRA(double* buffer, int buffer_size)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread_id == 0)
	{
		printf("Buffer on device before copying to host\n");
		for (int i = 0; i < buffer_size; i++)
			printf("time = %f; Vs = %f; Vd = %f\n", buffer[i*3], buffer[i*3+1], buffer[i*3+2]);

	}
}

__global__ void set_record_HVCRA(bool* record, int neuron_id)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	//printf("neuron_id = %d\n", neuron_id);

	if (thread_id == neuron_id)
	{
		printf("Set neuron %d to be recorded\n", neuron_id);
		record[neuron_id] = true;
	}
}


__device__ static double IdExt(double ampl, double t)
{
	if ( (t >= 10.0 ) && (t <= 30) )
		return ampl;
	else 
		return 0.0;
}

__device__ static double IsExt(double ampl, double t)
{
	if ( (t >= 10.0 ) && (t <= 30) )
		return ampl;
	else 
		return 0.0;
}

__device__ static double Gi(double G_cur, double t_cur, double t){return G_cur * exp(-(t - t_cur) / tInh);}
__device__ static double Ge(double G_cur, double t_cur, double t){return G_cur * exp(-(t - t_cur) / tExc);}


__device__ static double kVs(double vs, double vd, double n, double h, double t, double t_cur, double ge_s, double gi_s, double ampl_s)
{
	double m3, n4;

	m3 = mInf(vs) * mInf(vs) * mInf(vs);
	n4 = n * n * n * n;
	return (-GsL * (vs - EsL) - GsNa * m3 * h * (vs - EsNa) - GsK * n4 * (vs - EsK)
		 - Gi(gi_s, t_cur, t) * (vs - Egaba) - Ge(ge_s, t_cur, t) * vs + 100000 * IsExt(ampl_s, t) / As + 100000 * (vd - vs) / (Rc * As)) / cm;
}

__device__ static double kVd(double vs, double vd, double r, double c, double ca, double t, double t_cur, double ge_d, double gi_d, double ampl_d)
{
	double r2;

	r2 = r * r;
	return (-GdL * (vd - EdL) - GdCa * r2 * (vd - EdCa) - GdCaK * (vd - EdK) * c / (1 + 6 / ca)
		- Ge(ge_d, t_cur, t) * vd - Gi(gi_d, t_cur, t) * (vd - Egaba) + 100000 * IdExt(ampl_d, t) / Ad + 100000 * (vs - vd) / (Rc * Ad)) / cm;
}

__device__ static double kn(double vs, double n){return (nInf(vs) - n) / tauN(vs);}
__device__ static double kh(double vs, double h){return (hInf(vs) - h) / tauH(vs);}
__device__ static double kr(double vd, double r){return (rInf(vd) - r) / tauR(vd);}
__device__ static double kc(double vd, double c){return (cInf(vd) - c) / tauC(vd);}
__device__ static double kCa(double vd, double r, double ca)
	{return (-0.1 * GdCa * r * r * (vd - EdCa) - 0.02 * ca);}


__global__ void initialize_noise_RA(double *vars, int N, int seed, curandState* states)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (thread_id < N)
	{
		// initialize noise generators
		curand_init(seed, thread_id, 0, &states[thread_id]);
		
		// Ge_d noise time
		double random = curand_uniform(&states[thread_id]);
		vars[thread_id + 14*N] = 1000.0 * (- log(1.0 - random) / lambda_exc);
		
		// Ge_s noise time
		random = curand_uniform(&states[thread_id]);
		vars[thread_id + 15*N] = 1000.0 * (- log(1.0 - random) / lambda_exc);
		
		// Gi_d noise time
		random = curand_uniform(&states[thread_id]);
		vars[thread_id + 16*N] = 1000.0 * (- log(1.0 - random) / lambda_inh);
		
		// Gi_s noise time
		random = curand_uniform(&states[thread_id]);
		vars[thread_id + 17*N] = 1000.0 * (- log(1.0 - random) / lambda_inh);
	}
}

__global__ void calculate_next_step_RA(double *vars, bool* flags, int N, double timestep, bool* record, 
									   double* buffer, int step, double ampl_s, double ampl_d, int training, 
									   bool *spiked, int *num_spikes, double *spike_times, curandState *states)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread_id < N)
	{
		if (thread_id >= training)
		{
			ampl_s = 0;
			ampl_d = 0;
		}

		double n1, h1, c1, r1;	//	temporary values of gating variables
		double vts, vtd, Cat;	//	temporary values of variables
		double k1Vs, k2Vs, k3Vs, k4Vs;
		double k1Vd, k2Vd, k3Vd, k4Vd;
		double k1Ca, k2Ca, k3Ca, k4Ca;
		double k1n, k2n, k3n, k4n;
		double k1h, k2h, k3h, k4h;
		double k1c, k2c, k3c, k4c;
		double k1r, k2r, k3r, k4r;
		double t;

		double t_cur    = vars[thread_id];
		double Ge_d_cur = vars[thread_id + 8*N];
		double Ge_s_cur = vars[thread_id + 9*N];
		double Gi_d_cur = vars[thread_id + 10*N];
		double Gi_s_cur = vars[thread_id + 11*N];

		t   = t_cur;
		vts = vars[thread_id + N];
		n1  = vars[thread_id + 2*N];
		h1  = vars[thread_id + 3*N];
		vtd = vars[thread_id + 4*N];
		r1  = vars[thread_id + 5*N];
		c1  = vars[thread_id + 6*N];
		Cat = vars[thread_id + 7*N];

		k1Vs = kVs(vts, vtd, n1, h1, t, t_cur, Ge_s_cur, Gi_s_cur, ampl_s);
		k1n = kn(vts, n1);
		k1h = kh(vts, h1);
		k1Vd = kVd(vts, vtd, r1, c1, Cat, t, t_cur, Ge_d_cur, Gi_d_cur, ampl_d);
		k1r = kr(vtd, r1);
		k1c = kc(vtd, c1);
		k1Ca = kCa(vtd, r1, Cat);

		t   = vars[thread_id] + timestep / 3;
		vts = vars[thread_id + 1*N] + timestep * k1Vs / 3;
		n1  = vars[thread_id + 2*N] + timestep * k1n / 3;
		h1  = vars[thread_id + 3*N] + timestep * k1h / 3;
		vtd = vars[thread_id + 4*N] + timestep * k1Vd / 3;
		r1  = vars[thread_id + 5*N] + timestep * k1r / 3;
		c1  = vars[thread_id + 6*N] + timestep * k1c / 3;
		Cat = vars[thread_id + 7*N] + timestep * k1Ca / 3;

		k2Vs = kVs(vts, vtd, n1, h1, t, t_cur, Ge_s_cur, Gi_s_cur, ampl_s);
		k2n = kn(vts, n1);
		k2h = kh(vts, h1);
		k2Vd = kVd(vts, vtd, r1, c1, Cat, t, t_cur, Ge_d_cur, Gi_d_cur, ampl_d);
		k2r = kr(vtd, r1);
		k2c = kc(vtd, c1);
		k2Ca = kCa(vtd, r1, Cat);

		t   = vars[thread_id] + 2 * timestep / 3;
		vts = vars[thread_id + 1*N] + timestep * (-k1Vs / 3 + k2Vs);
		n1  = vars[thread_id + 2*N] + timestep * (-k1n / 3 + k2n);
		h1  = vars[thread_id + 3*N] + timestep * (-k1h / 3 + k2h);
		vtd = vars[thread_id + 4*N] + timestep * (-k1Vd / 3 + k2Vd);
		r1  = vars[thread_id + 5*N] + timestep * (-k1r / 3 + k2r);
		c1  = vars[thread_id + 6*N] + timestep * (-k1c / 3 + k2c);
		Cat = vars[thread_id + 7*N] + timestep * (-k1Ca / 3 + k2Ca);


		k3Vs = kVs(vts, vtd, n1, h1, t, t_cur, Ge_s_cur, Gi_s_cur, ampl_s);
		k3n = kn(vts, n1);
		k3h = kh(vts, h1);
		k3Vd = kVd(vts, vtd, r1, c1, Cat, t, t_cur, Ge_d_cur, Gi_d_cur, ampl_d);
		k3r = kr(vtd, r1);
		k3c = kc(vtd, c1);
		k3Ca = kCa(vtd, r1, Cat);

		t   = vars[thread_id] + timestep;
		vts = vars[thread_id + 1*N] + timestep * (k1Vs - k2Vs + k3Vs);
		n1  = vars[thread_id + 2*N] + timestep * (k1n - k2n + k3n);
		h1  = vars[thread_id + 3*N] + timestep * (k1h - k2h + k3h);
		vtd = vars[thread_id + 4*N] + timestep * (k1Vd - k2Vd + k3Vd);
		r1  = vars[thread_id + 5*N] + timestep * (k1r - k2r + k3r);
		c1  = vars[thread_id + 6*N] + timestep * (k1c - k2c + k3c);
		Cat = vars[thread_id + 7*N] + timestep * (k1Ca - k2Ca + k3Ca);


		k4Vs = kVs(vts, vtd, n1, h1, t, t_cur, Ge_s_cur, Gi_s_cur, ampl_s);
		k4n = kn(vts, n1);
		k4h = kh(vts, h1);
		k4Vd = kVd(vts, vtd, r1, c1, Cat, t, t_cur, Ge_d_cur, Gi_d_cur, ampl_d);
		k4r = kr(vtd, r1);
		k4c = kc(vtd, c1);
		k4Ca = kCa(vtd, r1, Cat);

		vars[thread_id + 8*N]  = Ge(vars[thread_id + 8*N],  vars[thread_id], vars[thread_id] + timestep); // Gexc_d
		vars[thread_id + 9*N]  = Ge(vars[thread_id + 9*N],  vars[thread_id], vars[thread_id] + timestep); // Gexc_s
		vars[thread_id + 10*N] = Gi(vars[thread_id + 10*N], vars[thread_id], vars[thread_id] + timestep); // Ginh_d
		vars[thread_id + 11*N] = Gi(vars[thread_id + 11*N], vars[thread_id], vars[thread_id] + timestep); // Ginh_s
		
		vars[thread_id + 1*N] += timestep * (k1Vs + 3 * k2Vs + 3 * k3Vs + k4Vs) / 8;
		vars[thread_id + 2*N] += timestep * (k1n + 3 * k2n + 3 * k3n + k4n) / 8;
		vars[thread_id + 3*N] += timestep * (k1h + 3 * k2h + 3 * k3h + k4h) / 8;
		vars[thread_id + 4*N] += timestep * (k1Vd + 3 * k2Vd + 3 * k3Vd + k4Vd) / 8;
		vars[thread_id + 5*N] += timestep * (k1r + 3 * k2r + 3 * k3r + k4r) / 8;
		vars[thread_id + 6*N] += timestep * (k1c + 3 * k2c + 3 * k3c + k4c) / 8;
		vars[thread_id + 7*N] += timestep * (k1Ca + 3 * k2Ca + 3 * k3Ca + k4Ca) / 8;
		
		vars[thread_id] += timestep;
		
		vars[thread_id + 12*N] = IdExt(ampl_d, vars[thread_id]); // Id
		vars[thread_id + 13*N] = IsExt(ampl_s, vars[thread_id]); // Is
	
		if ( record[thread_id] )
		{
			buffer[step*3] = vars[thread_id];
			buffer[step*3 + 1] = vars[thread_id + N];
			buffer[step*3 + 2] = vars[thread_id + 4*N];

			//printf("time = %f; Vs = %f; Vd = %f\n", vars[thread_id], vars[thread_id +N], vars[thread_id + 4*N]);
			//printf("step = %d; time = %f; Vs = %f; Vd = %f\n", step, buffer[step], buffer[step+1], buffer[step+2]);
		}

		// check if neuron has spiked
		if ( ( vars[thread_id + 1*N] >= THRESHOLD_SPIKE) && ( flags[thread_id] == false ) )
		{
			spiked[thread_id] = true;
			flags[thread_id] = true;
			spike_times[thread_id*MAX_NUM_OF_SPIKES + num_spikes[thread_id]] = vars[thread_id];
			num_spikes[thread_id] += 1;
			
			//if (thread_id < 200)
			//	printf("Neuron %d spiked at %f; spike num = %d\n", thread_id, vars[thread_id], num_spikes[thread_id]);
		}
		else if ( (flags[thread_id] == true) && ( vars[thread_id + 1*N] < THRESHOLD_SPIKE - SPIKE_MARGIN ) )
			flags[thread_id] = false;
			
			
		// check if noise input has arrives
		// Ge_d input
		if ( vars[thread_id] >=  vars[thread_id + 14*N] )
		{
			vars[thread_id + 8*N] += curand_uniform(&states[thread_id]) * Gd_noise_exc; // update conductance
			
			// sample new noise input time
			double random = curand_uniform(&states[thread_id]);
			vars[thread_id + 14*N] += 1000.0 * (- log(1.0 - random) / lambda_exc);
		}
		
		// Ge_s input
		if ( vars[thread_id] >=  vars[thread_id + 15*N] )
		{
			vars[thread_id + 9*N] += curand_uniform(&states[thread_id]) * Gs_noise_exc; // update conductance
			
			// sample new noise input time
			double random = curand_uniform(&states[thread_id]);
			vars[thread_id + 15*N] += 1000.0 * (- log(1.0 - random) / lambda_exc);
		}
		
		// Gi_d input
		if ( vars[thread_id] >=  vars[thread_id + 16*N] )
		{
			vars[thread_id + 10*N] += curand_uniform(&states[thread_id]) * Gd_noise_inh; // update conductance
			
			// sample new noise input time
			double random = curand_uniform(&states[thread_id]);
			vars[thread_id + 16*N] += 1000.0 * (- log(1.0 - random) / lambda_inh);
		}
		
		// Gi_s input
		if ( vars[thread_id] >=  vars[thread_id + 17*N] )
		{
			vars[thread_id + 11*N] += curand_uniform(&states[thread_id]) * Gs_noise_inh; // update conductance
			
			// sample new noise input time
			double random = curand_uniform(&states[thread_id]);
			vars[thread_id + 17*N] += 1000.0 * (- log(1.0 - random) / lambda_inh);
		}
	}
}

