#include <iostream>
#include <stdio.h>
#include "hvcI.cuh"
#include <curand.h>
#include <curand_kernel.h>

using namespace hvcIConstants;

__global__ void print_buffer_HVCI(double* buffer, int buffer_size)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread_id == 0)
	{
		printf("Buffer on device before copying to host\n");
		for (int i = 0; i < buffer_size; i++)
			printf("time = %f; V = %f; Ge = %f\n", buffer[i*3], buffer[i*3+1], buffer[i*3+2]);

	}
}

__global__ void set_record_HVCI(bool* record, int neuron_id)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	//printf("neuron_id = %d\n", neuron_id);

	if (thread_id == neuron_id)
	{
		printf("Set neuron %d to be recorded\n", neuron_id);
		record[neuron_id] = true;
	}
}

__device__ static double Gi(double G_cur, double t_cur, double t){return G_cur * exp(-(t - t_cur) / tInh);}
__device__ static double Ge(double G_cur, double t_cur, double t){return G_cur * exp(-(t - t_cur) / tExc);}

__device__ static double I(double ampl, double t)
{
	if ( (t >= 10.0 ) && (t <= 110) )
		return ampl;
	else 
		return 0.0;
}

__device__ static double kV(double v, double t, double t_cur, double h, double w, double m3, double n4, 
					 double ge, double gi, double ampl){
	return (-gL * (v - El) - gNa * h * m3 * (v - Ena) - gKdr * n4 * (v - Ek)
		- gKHT * w * (v - Ek) - Ge(ge, t_cur, t) * v - Gi(gi, t_cur, t) * (v - Ei) + 100000 * I(ampl, t) / A) / cm;}

__device__ static double kn(double v, double n){return an(v)*(1 - n) - bn(v)*n;}
__device__ static double km(double v, double m){return am(v)*(1 - m) - bm(v)*m;}
__device__ static double kh(double v, double h){return ah(v)*(1 - h) - bh(v)*h;}
__device__ static double kw(double v, double w){return (wInf(v) - w) / tauW(v);}

__global__ void initialize_noise_I(double *vars, int N, int seed, curandState* states)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (thread_id < N)
	{
		// initialize noise generators
		curand_init(seed, thread_id, 0, &states[thread_id]);
		
		// Ge noise time
		double random = curand_uniform(&states[thread_id]);
		vars[thread_id + 9*N] = 1000.0 * (- log(1.0 - random) / lambda);
		
		// Gi noise time
		random = curand_uniform(&states[thread_id]);
		vars[thread_id + 10*N] = 1000.0 * (- log(1.0 - random) / lambda);
	}
}

__global__ void calculate_next_step_I(double *vars, bool* flags, int N, double timestep, bool* record, 
									  double* buffer, int step, double ampl, bool *spiked, 
									  int *num_spikes, double *spike_times, curandState* states)
{
	double m1, n1, h1, w1, m3, n4;
	double v, t;
	double k1V, k2V, k3V, k4V;
	double k1n, k2n, k3n, k4n;
	double k1m, k2m, k3m, k4m;
	double k1h, k2h, k3h, k4h;
	double k1w, k2w, k3w, k4w;

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread_id < N)
	{	
		double t_cur  = vars[thread_id];
		double Ge_cur = vars[thread_id + 6*N];
		double Gi_cur = vars[thread_id + 7*N];
	
		t  = vars[thread_id];
		v  = vars[thread_id + 1*N];
		m1 = vars[thread_id + 2*N];
		n1 = vars[thread_id + 3*N];
		h1 = vars[thread_id + 4*N];
		w1 = vars[thread_id + 5*N];
		n4 = n1 * n1 * n1 * n1;
		m3 = m1 * m1 * m1;
		
		k1V = kV(v, t, t_cur, h1, w1, m3, n4, Ge_cur, Gi_cur, ampl);
		k1n = kn(v, n1);
		k1m = km(v, m1);
		k1h = kh(v, h1);
		k1w = kw(v, w1);
		
		t  = vars[thread_id] + timestep / 3;
		v  = vars[thread_id + 1*N] + timestep * k1V / 3;
		m1 = vars[thread_id + 2*N] + timestep * k1m / 3;
		n1 = vars[thread_id + 3*N] + timestep * k1n / 3;
		h1 = vars[thread_id + 4*N] + timestep * k1h / 3;
		w1 = vars[thread_id + 5*N] + timestep * k1w / 3;
		n4 = n1 * n1 * n1 * n1;
		m3 = m1 * m1 * m1;
		
		k2V = kV(v, t, t_cur, h1, w1, m3, n4, Ge_cur, Gi_cur, ampl);
		k2n = kn(v, n1);
		k2m = km(v, m1);
		k2h = kh(v, h1);
		k2w = kw(v, w1);

		t  = vars[thread_id] + 2 * timestep / 3;
		v  = vars[thread_id + 1*N] + timestep * (-k1V / 3 + k2V);
		m1 = vars[thread_id + 2*N] + timestep * (-k1m / 3 + k2m);
		n1 = vars[thread_id + 3*N] + timestep * (-k1n / 3 + k2n);
		h1 = vars[thread_id + 4*N] + timestep * (-k1h / 3 + k2h);
		w1 = vars[thread_id + 5*N] + timestep * (-k1w / 3 + k2w);
		n4 = n1 * n1 * n1 * n1;
		m3 = m1 * m1 * m1;
		
		k3V = kV(v, t, t_cur, h1, w1, m3, n4, Ge_cur, Gi_cur, ampl);
		k3n = kn(v, n1);
		k3m = km(v, m1);
		k3h = kh(v, h1);
		k3w = kw(v, w1);

		t  = vars[thread_id] + timestep;
		v  = vars[thread_id + 1*N] + timestep * (k1V - k2V + k3V);
		m1 = vars[thread_id + 2*N] + timestep * (k1m - k2m + k3m);
		n1 = vars[thread_id + 3*N] + timestep * (k1n - k2n + k3n);
		h1 = vars[thread_id + 4*N] + timestep * (k1h - k2h + k3h);
		w1 = vars[thread_id + 5*N] + timestep * (k1w - k2w + k3w);
		n4 = n1 * n1 * n1 * n1;
		m3 = m1 * m1 * m1;
		
		k4V = kV(v, t, t_cur, h1, w1, m3, n4, Ge_cur, Gi_cur, ampl);
		k4n = kn(v, n1);
		k4m = km(v, m1);
		k4h = kh(v, h1);
		k4w = kw(v, w1);

		//	update all values for next time point
		vars[thread_id + 6*N]  = Ge(vars[thread_id + 6*N],  vars[thread_id], vars[thread_id] + timestep); // Gexc_d
		vars[thread_id + 7*N]  = Gi(vars[thread_id + 7*N],  vars[thread_id], vars[thread_id] + timestep); // Gexc_s
		
		vars[thread_id]  = vars[thread_id] + timestep;
		vars[thread_id + 1*N]  = vars[thread_id + 1*N] + timestep * (k1V + 3 * k2V + 3 * k3V + k4V) / 8;
		vars[thread_id + 2*N] = vars[thread_id + 2*N] + timestep * (k1m + 3 * k2m + 3 * k3m + k4m) / 8;
		vars[thread_id + 3*N] = vars[thread_id + 3*N] + timestep * (k1n + 3 * k2n + 3 * k3n + k4n) / 8;
		vars[thread_id + 4*N] = vars[thread_id + 4*N] + timestep * (k1h + 3 * k2h + 3 * k3h + k4h) / 8;
		vars[thread_id + 5*N] = vars[thread_id + 5*N] + timestep * (k1w + 3 * k2w + 3 * k3w + k4w) / 8;
		
		vars[thread_id + 8*N] = I(ampl, vars[thread_id]); // Id
		
		if ( record[thread_id] )
		{
			buffer[step*3] = vars[thread_id];
			buffer[step*3 + 1] = vars[thread_id + N];
			buffer[step*3 + 2] = vars[thread_id + 6*N];

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
		// Ge input
		if ( vars[thread_id] >=  vars[thread_id + 9*N] )
		{
			vars[thread_id + 6*N] += curand_uniform(&states[thread_id]) * G_noise; // update conductance
			
			// sample new noise input time
			double random = curand_uniform(&states[thread_id]);
			vars[thread_id + 9*N] += 1000.0 * (- log(1.0 - random) / lambda);
		}
		
		// Gi input
		if ( vars[thread_id] >=  vars[thread_id + 10*N] )
		{
			vars[thread_id + 7*N] += curand_uniform(&states[thread_id]) * G_noise; // update conductance
			
			// sample new noise input time
			double random = curand_uniform(&states[thread_id]);
			vars[thread_id + 10*N] += 1000.0 * (- log(1.0 - random) / lambda);
		}
	}
}
