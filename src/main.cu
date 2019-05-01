#include "hvcRA.cuh"
#include "hvcI.cuh"
#include "make_network.h"
#include "poisson_noise.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define NUM_THREADS_IN_BLOCK 1024
#define BUFFER_SIZE 2000
#define MAX_NUM_OF_SPIKES 50


struct NetworkParameters
{
	int N_RA;
	int N_I;
	int num_neurons_in_layer;

	double p_RA2RA;
	double Gmax_RA2RA;

	double p_RA2I;
	double Gmax_RA2I;

	double p_I2RA;
	double Gmax_I2RA;
};

double myDiffTime(struct timeval &start, struct timeval &end)
{
	double d_start = (double) (start.tv_sec + start.tv_usec/1000000.0);
	double d_end   = (double) (end.tv_sec + end.tv_usec/1000000.0);

	return (d_end - d_start);
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
    {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) exit(code);
	}
}

void write_spikes(const double* spike_times, const int* num_spikes, int N, const char *filename)
{
	std::ofstream out;
	out.open(filename, std::ios::out | std::ios::binary);

	out.write(reinterpret_cast<char*>(&N), sizeof(int));

	for (int i = 0; i < N; i++)
	{
		out.write(reinterpret_cast<const char*>(&num_spikes[i]), sizeof(int));

		for (int j = 0; j < num_spikes[i]; j++)
			out.write(reinterpret_cast<const char*>(&spike_times[i*MAX_NUM_OF_SPIKES + j]), sizeof(double));

	}

	out.close();

}

void write_data(double* buffer, int num_iter, const char *filename)
{
	std::ofstream out;
	out.open(filename, std::ios::out | std::ios::binary);

	out.write(reinterpret_cast<char*>(&num_iter), sizeof(int));

	for (int i = 0; i < num_iter; i++)
	{
		//std::cout << "time = " << buffer[i*3] 
		//		  << " Vs = "  << buffer[i*3+1]
		//		  << " Vd = "  << buffer[i*3+2] << std::endl; 

		out.write(reinterpret_cast<char*>(&buffer[i*3]), sizeof(double));
		out.write(reinterpret_cast<char*>(&buffer[i*3+1]), sizeof(double));
		out.write(reinterpret_cast<char*>(&buffer[i*3+2]), sizeof(double));

	}

	out.close();
}

__device__ double my_atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = 
									(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
						__double_as_longlong(val + 
						__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);

}

__global__ void update_target_conductances_HVCRA(double *vars_HVCRA, double *vars_HVCI, int N_RA, int N_I, bool *spiked, 
											     int *targets_id_RA2RA, double *weights_RA2RA, int *cum_num_targets_RA2RA, int *num_targets_RA2RA,
											     int *targets_id_RA2I,  double *weights_RA2I,  int *cum_num_targets_RA2I,  int *num_targets_RA2I)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread_id < N_RA)
	{
		if ( spiked[thread_id] )
		{
			for (int i = 0; i < num_targets_RA2RA[thread_id]; i++)
			{
				int target_id = targets_id_RA2RA[cum_num_targets_RA2RA[thread_id] + i];
				double weight = weights_RA2RA[cum_num_targets_RA2RA[thread_id] + i];

				//printf("target_id = %d; weight = %f; Ge before update = %f\n", target_id, weight, vars_HVCRA[target_id + 8*N_RA]);
				
				my_atomicAdd(&vars_HVCRA[target_id + 8*N_RA], weight);
				
				//printf("target_id = %d; weight = %f; Ge after update = %f\n", target_id, weight, &vars_HVCRA[target_id + 8*N_RA]);
			}

			for (int i = 0; i < num_targets_RA2I[thread_id]; i++)
			{
				int target_id = targets_id_RA2I[cum_num_targets_RA2I[thread_id] + i];
				double weight = weights_RA2I[cum_num_targets_RA2I[thread_id] + i];

				//printf("target_id = %d; weight = %f; Ge before update = %f\n", target_id, weight, vars_HVCI[target_id + 6*N_I]);

				my_atomicAdd(&vars_HVCI[target_id + 6*N_I], weight);
				
				//printf("target_id = %d; weight = %f; Ge after update = %f\n", target_id, weight, vars_HVCI[target_id + 6*N_I]);


			}

			spiked[thread_id] = false;
		}
	}
}

__global__ void update_target_conductances_RA2RA(double *vars, int N, bool *spiked, int *targets_id, double *weights, int *cum_num_targets, int *num_targets)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread_id < N)
	{
		if ( spiked[thread_id] )
		{
			for (int i = 0; i < num_targets[thread_id]; i++)
			{
				int target_id = targets_id[cum_num_targets[thread_id] + i];
				double weight = weights[cum_num_targets[thread_id] + i];

				my_atomicAdd(&vars[target_id + 8*N], weight);
			}

			//spiked[thread_id] = false;
		}
	}
}

__global__ void update_target_conductances_RA2I(double *vars, int N_RA, int N_I, bool *spiked, int *targets_id, double *weights, int *cum_num_targets, int *num_targets)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread_id < N_RA)
	{
		if ( spiked[thread_id] )
		{
			for (int i = 0; i < num_targets[thread_id]; i++)
			{
				int target_id = targets_id[cum_num_targets[thread_id] + i];
				double weight = weights[cum_num_targets[thread_id] + i];

				my_atomicAdd(&vars[target_id + 6*N_I], weight);
			}

			spiked[thread_id] = false;
		}
	}
}

__global__ void update_target_conductances_I2RA(double *vars, int N_I, int N_RA, bool *spiked, int *targets_id, double *weights, int *cum_num_targets, int *num_targets)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread_id < N_I)
	{
		if ( spiked[thread_id] )
		{
			for (int i = 0; i < num_targets[thread_id]; i++)
			{
				int target_id = targets_id[cum_num_targets[thread_id] + i];
				double weight = weights[cum_num_targets[thread_id] + i];

				my_atomicAdd(&vars[target_id + 10*N_RA], weight);
			}

			spiked[thread_id] = false;
		}
	}
}



void populate_connections(const std::vector<std::vector<int>>& targets_ID, 
						  const std::vector<std::vector<double>>& weights,
						  int **d_targets_id, double **d_weights, int **d_num_targets, int **d_cum_num_targets)
{
	int N = static_cast<int>(targets_ID.size());

	int total_num_of_targets = 0;

	for (int i = 0; i < N; i++)
		total_num_of_targets += static_cast<int>(targets_ID[i].size());

	std::cout << "Total number of targets for " << N << " source neurons = " << total_num_of_targets << std::endl;

	int *h_targets_id = new int[total_num_of_targets];
	double *h_weights = new double[total_num_of_targets];
	int *h_cum_num_targets = new int[N];
	int *h_num_targets = new int[N];

	// populate arrays with connections
	// populate separately for neuron with id 0
	h_cum_num_targets[0] = 0;
	h_num_targets[0] = static_cast<int>(targets_ID[0].size());
	for (size_t i = 0; i < targets_ID[0].size(); i++)
	{
		h_targets_id[i] = targets_ID[0][i];
		h_weights[i] = weights[0][i];
	}
	
	for (int i = 1; i < N; i++)
	{
		int num_targets = static_cast<int>(targets_ID[i].size());
		
		h_cum_num_targets[i] = h_cum_num_targets[i-1] + static_cast<int>(targets_ID[i-1].size());
		h_num_targets[i] = num_targets;
		
		for (int j = 0; j < num_targets; j++)
		{
			h_targets_id[h_cum_num_targets[i] + j] = targets_ID[i][j];
			h_weights[h_cum_num_targets[i] + j] = weights[i][j];


		}
	}

	// allocate memory on device
	gpuErrchk(cudaMalloc(d_targets_id, total_num_of_targets*sizeof(int)));
	gpuErrchk(cudaMalloc(d_weights, total_num_of_targets*sizeof(double)));
	gpuErrchk(cudaMalloc(d_cum_num_targets, N*sizeof(int)));
	gpuErrchk(cudaMalloc(d_num_targets, N*sizeof(int)));

	// copy synapses
	gpuErrchk(cudaMemcpy(*d_targets_id, h_targets_id, total_num_of_targets*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_weights, h_weights, total_num_of_targets*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_cum_num_targets, h_cum_num_targets, N*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_num_targets, h_num_targets, N*sizeof(int), cudaMemcpyHostToDevice));

	// free memory
	delete[] h_targets_id; delete[] h_weights; delete[] h_num_targets; delete[] h_cum_num_targets;
}

void generate_network(const struct NetworkParameters& params, Poisson_noise* noise_generator,
					  int **d_targets_id_RA2RA, double **d_weights_RA2RA, int **d_num_targets_RA2RA, int **d_cum_num_targets_RA2RA,
					  int **d_targets_id_RA2I,  double **d_weights_RA2I,  int **d_num_targets_RA2I,  int **d_cum_num_targets_RA2I,
					  int **d_targets_id_I2RA,  double **d_weights_I2RA,  int **d_num_targets_I2RA,  int **d_cum_num_targets_I2RA)
{
	std::vector<std::vector<int>> targets_ID;
	std::vector<std::vector<double>> weights;

	make_chain(params.N_RA, params.num_neurons_in_layer, params.p_RA2RA, params.Gmax_RA2RA, targets_ID, weights, noise_generator); 
	populate_connections(targets_ID, weights, d_targets_id_RA2RA, d_weights_RA2RA, d_num_targets_RA2RA, d_cum_num_targets_RA2RA);

	std::vector<std::vector<int>>().swap(targets_ID);
	std::vector<std::vector<double>>().swap(weights);

	make_connections_source2target(params.N_RA, params.N_I, params.p_RA2I, params.Gmax_RA2I, targets_ID, weights, noise_generator);
	populate_connections(targets_ID, weights, d_targets_id_RA2I, d_weights_RA2I, d_num_targets_RA2I, d_cum_num_targets_RA2I);

	std::vector<std::vector<int>>().swap(targets_ID);
	std::vector<std::vector<double>>().swap(weights);

	make_connections_source2target(params.N_I, params.N_RA, params.p_I2RA, params.Gmax_I2RA, targets_ID, weights, noise_generator);
	populate_connections(targets_ID, weights, d_targets_id_I2RA, d_weights_I2RA, d_num_targets_I2RA, d_cum_num_targets_I2RA);

}

void initialize_neurons(int N_RA,                   int N_I, 
						double **d_vars_HVCRA,      double **d_vars_HVCI,
						double **d_buffer_HVCRA,    double **d_buffer_HVCI,
						bool **d_record_HVCRA,      bool **d_record_HVCI,
						bool **d_bool_spiked_HVCRA, bool **d_bool_spiked_HVCI,
						bool **d_flags_HVCRA,       bool **d_flags_HVCI)
{
	// HVC-RA neurons

	double *h_vars_HVCRA = new double[N_RA*18];
	double *h_buffer_HVCRA = new double[BUFFER_SIZE*3];
	bool *h_record_HVCRA = new bool[N_RA];
	bool *h_bool_spiked_HVCRA = new bool[N_RA];
	bool *h_flags_HVCRA = new bool[N_RA];

	// initialize variables for neurons
	for (int i = 0; i < N_RA; i++)
	{
		h_vars_HVCRA[i] = 0; // time
		h_vars_HVCRA[i + 1*N_RA] = -79.97619025; // Vs
		h_vars_HVCRA[i + 2*N_RA] = 0.01101284; // n
		h_vars_HVCRA[i + 3*N_RA] = 0.9932845; // h
		h_vars_HVCRA[i + 4*N_RA] = -79.97268759; // Vd
		h_vars_HVCRA[i + 5*N_RA] = 0.00055429; // r
		h_vars_HVCRA[i + 6*N_RA] = 0.00000261762353; // c
		h_vars_HVCRA[i + 7*N_RA] = 0.01689572; // Ca
		h_vars_HVCRA[i + 8*N_RA] = 0; // Gexc_d
		h_vars_HVCRA[i + 9*N_RA] = 0; // Gexc_s
		h_vars_HVCRA[i + 10*N_RA] = 0; // Ginh_d
		h_vars_HVCRA[i + 11*N_RA] = 0; // Ginh_s
		h_vars_HVCRA[i + 12*N_RA] = 0; // Id
		h_vars_HVCRA[i + 13*N_RA] = 0; // Is
		h_vars_HVCRA[i + 14*N_RA] = 0; // noise input time Gexc_d
		h_vars_HVCRA[i + 15*N_RA] = 0; // noise input time Gexc_s
		h_vars_HVCRA[i + 16*N_RA] = 0; // noise input time Ginh_d
		h_vars_HVCRA[i + 17*N_RA] = 0; // noise input time Ginh_s

		h_record_HVCRA[i] = false;
		h_bool_spiked_HVCRA[i] = false;
		h_flags_HVCRA[i] = false;
	}

	for (int i = 0; i < 3*BUFFER_SIZE; i++)
	{
		h_buffer_HVCRA[i] = 0.0;
	}

	// copy data
	// neuron variables
	gpuErrchk(cudaMalloc(d_vars_HVCRA, N_RA*18*sizeof(double)));
	gpuErrchk(cudaMalloc(d_record_HVCRA, N_RA*sizeof(bool)));
	gpuErrchk(cudaMalloc(d_buffer_HVCRA, BUFFER_SIZE*3*sizeof(double)));
	
	// dynamics
	gpuErrchk(cudaMalloc(d_bool_spiked_HVCRA, N_RA*sizeof(bool)));
	gpuErrchk(cudaMalloc(d_flags_HVCRA, N_RA*sizeof(bool)));
	
	// copy memory from host to device
	// copy neuron varoables
	gpuErrchk(cudaMemcpy(*d_vars_HVCRA, h_vars_HVCRA, N_RA*18*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_record_HVCRA, h_record_HVCRA, N_RA*sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_buffer_HVCRA, h_buffer_HVCRA, BUFFER_SIZE*3*sizeof(double), cudaMemcpyHostToDevice));
	
	// copy dynamics variables
	gpuErrchk(cudaMemcpy(*d_bool_spiked_HVCRA, h_bool_spiked_HVCRA, N_RA*sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_flags_HVCRA, h_flags_HVCRA, N_RA*sizeof(bool), cudaMemcpyHostToDevice));

	// free memory
	delete[] h_buffer_HVCRA; delete[] h_record_HVCRA; delete[] h_vars_HVCRA; delete[] h_bool_spiked_HVCRA; delete[] h_flags_HVCRA;


	// HVC-I neurons

	double *h_vars_HVCI = new double[N_I*11];
	double *h_buffer_HVCI = new double[BUFFER_SIZE*3];
	bool *h_record_HVCI = new bool[N_I];
	bool *h_bool_spiked_HVCI = new bool[N_I];
	bool *h_flags_HVCI = new bool[N_I];

	// initialize variables for neurons
	for (int i = 0; i < N_I; i++)
	{
		h_vars_HVCI[i] = 0.0; // time
		h_vars_HVCI[i + 1*N_I] = -66; // v
		h_vars_HVCI[i + 2*N_I] = 0.0; // m
		h_vars_HVCI[i + 3*N_I] = 0.125; // n
		h_vars_HVCI[i + 4*N_I] = 0.99; // h
		h_vars_HVCI[i + 5*N_I] = 0.0; // w
		h_vars_HVCI[i + 6*N_I] = 0.0; // Ge
		h_vars_HVCI[i + 7*N_I] = 0.0; // Gi
		h_vars_HVCI[i + 8*N_I] = 0.0; // I
		h_vars_HVCI[i + 9*N_I] = 0.0; // noise input time Ge
		h_vars_HVCI[i + 10*N_I] = 0.0; // noise input time Gi

		h_record_HVCI[i] = false;
		h_bool_spiked_HVCI[i] = false;
		h_flags_HVCI[i] = false;
	}

	for (int i = 0; i < 3*BUFFER_SIZE; i++)
	{
		h_buffer_HVCI[i] = 0.0;
	}

	// copy data
	// neuron variables
	gpuErrchk(cudaMalloc(d_vars_HVCI, N_I*11*sizeof(double)));
	gpuErrchk(cudaMalloc(d_record_HVCI, N_I*sizeof(bool)));
	gpuErrchk(cudaMalloc(d_buffer_HVCI, BUFFER_SIZE*3*sizeof(double)));
	
	// dynamics
	gpuErrchk(cudaMalloc(d_bool_spiked_HVCI, N_I*sizeof(bool)));
	gpuErrchk(cudaMalloc(d_flags_HVCI, N_I*sizeof(bool)));
	
	// copy memory from host to device
	// copy neuron varoables
	gpuErrchk(cudaMemcpy(*d_vars_HVCI, h_vars_HVCI, N_I*11*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_record_HVCI, h_record_HVCI, N_I*sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_buffer_HVCI, h_buffer_HVCI, BUFFER_SIZE*3*sizeof(double), cudaMemcpyHostToDevice));
	
	// copy dynamics variables
	gpuErrchk(cudaMemcpy(*d_bool_spiked_HVCI, h_bool_spiked_HVCI, N_I*sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_flags_HVCI, h_flags_HVCI, N_I*sizeof(bool), cudaMemcpyHostToDevice));

	// free memory
	delete[] h_buffer_HVCI; delete[] h_record_HVCI; delete[] h_vars_HVCI; delete[] h_bool_spiked_HVCI; delete[] h_flags_HVCI;
}

void initialize_spike_info(int N_RA, int N_I,
						   int **h_num_spikes_HVCRA, int **d_num_spikes_HVCRA,
						   double **h_spike_times_HVCRA, double **d_spike_times_HVCRA,
						   int **h_num_spikes_HVCI, int **d_num_spikes_HVCI,
						   double **h_spike_times_HVCI, double **d_spike_times_HVCI)
{
	// HVC-RA
	*h_num_spikes_HVCRA = new int[N_RA];
	*h_spike_times_HVCRA = new double[N_RA * MAX_NUM_OF_SPIKES];


	for (int i = 0; i < N_RA; i++)
	{
		(*h_num_spikes_HVCRA)[i] = 0;

		for (int j = 0; j < MAX_NUM_OF_SPIKES; j++)
			(*h_spike_times_HVCRA)[i*MAX_NUM_OF_SPIKES + j] = -1.0;
	}
	
	gpuErrchk(cudaMalloc(d_num_spikes_HVCRA, N_RA*sizeof(int)));
	gpuErrchk(cudaMalloc(d_spike_times_HVCRA, N_RA*MAX_NUM_OF_SPIKES*sizeof(double)));
	
	// copy memory from host to device
	gpuErrchk(cudaMemcpy(*d_num_spikes_HVCRA, *h_num_spikes_HVCRA, N_RA*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_spike_times_HVCRA, *h_spike_times_HVCRA, N_RA*MAX_NUM_OF_SPIKES*sizeof(double), cudaMemcpyHostToDevice));
	
	// HVC-I
	
	*h_num_spikes_HVCI = new int[N_I];
	*h_spike_times_HVCI = new double[N_I * MAX_NUM_OF_SPIKES];


	for (int i = 0; i < N_I; i++)
	{
		(*h_num_spikes_HVCI)[i] = 0;

		for (int j = 0; j < MAX_NUM_OF_SPIKES; j++)
			(*h_spike_times_HVCI)[i*MAX_NUM_OF_SPIKES + j] = -1.0;
	}
	
	gpuErrchk(cudaMalloc(d_num_spikes_HVCI, N_I*sizeof(int)));
	gpuErrchk(cudaMalloc(d_spike_times_HVCI, N_I*MAX_NUM_OF_SPIKES*sizeof(double)));
	
	// copy memory from host to device
	gpuErrchk(cudaMemcpy(*d_num_spikes_HVCI,  *h_num_spikes_HVCI, N_I*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*d_spike_times_HVCI, *h_spike_times_HVCI, N_I*MAX_NUM_OF_SPIKES*sizeof(double), cudaMemcpyHostToDevice));
}

int main(int argc, char** argv)
{
	Poisson_noise noise_generator;

	unsigned seed = 1991;
	int N_RA = 20000;
	int N_I = 8000;
	int num_neurons_in_layer = 200;

	double p_RA2RA = 1.0;
	double Gmax_RA2RA = 0.020;

	double p_RA2I = 0.002;
	double Gmax_RA2I = 0.50;

	double p_I2RA = 0.01;
	double Gmax_I2RA = 0.10;

	NetworkParameters params = {N_RA, N_I, num_neurons_in_layer, p_RA2RA, Gmax_RA2RA, p_RA2I, Gmax_RA2I, p_I2RA, Gmax_I2RA};

	std::string filename_buffer_HVCRA = "/storage/home/yzt116/ConcurrentMatrixComputation/mini_project/neuron_dynamics_HVCRA.bin";
	std::string filename_spikes_HVCRA = "/storage/home/yzt116/ConcurrentMatrixComputation/mini_project/spikes_HVCRA.bin";
	
	std::string filename_buffer_HVCI = "/storage/home/yzt116/ConcurrentMatrixComputation/mini_project/neuron_dynamics_HVCI.bin";
	std::string filename_spikes_HVCI = "/storage/home/yzt116/ConcurrentMatrixComputation/mini_project/spikes_HVCI.bin";
	
	if (argc > 4)
	{	
		filename_buffer_HVCRA = argv[1];
		filename_spikes_HVCRA = argv[2];
		filename_buffer_HVCI = argv[3];
		filename_spikes_HVCI = argv[4];

		std::cout << "filename_buffer_HVCRA = " << filename_buffer_HVCRA << std::endl;
		std::cout << "filename_spikes_HVCRA = " << filename_spikes_HVCRA << std::endl;
		std::cout << "filename_buffer_HVCI = " << filename_buffer_HVCI << std::endl;
		std::cout << "filename_spikes_HVCI = " << filename_spikes_HVCI << std::endl;
	}

	noise_generator.set_seed(seed);

	// initialize arrays
	double *d_vars_HVCRA, *d_vars_HVCI; 
	double *d_buffer_HVCRA, *d_buffer_HVCI;
	bool *d_record_HVCRA, *d_bool_spiked_HVCRA, *d_flags_HVCRA, *d_record_HVCI, *d_bool_spiked_HVCI, *d_flags_HVCI;

	int *d_targets_id_RA2RA, *d_num_targets_RA2RA, *d_cum_num_targets_RA2RA;
	int *d_targets_id_RA2I,  *d_num_targets_RA2I,  *d_cum_num_targets_RA2I;
	int *d_targets_id_I2RA,  *d_num_targets_I2RA,  *d_cum_num_targets_I2RA;
	
	
	double *d_weights_RA2RA, *d_weights_RA2I, *d_weights_I2RA;
	
	int *d_num_spikes_HVCRA, *h_num_spikes_HVCRA, *d_num_spikes_HVCI, *h_num_spikes_HVCI;
	double *d_spike_times_HVCRA, *h_spike_times_HVCRA, *d_spike_times_HVCI, *h_spike_times_HVCI;

	timeval start, start_calc, end;

	gettimeofday(&start, NULL);

	generate_network(params, &noise_generator,
					 &d_targets_id_RA2RA, &d_weights_RA2RA, &d_num_targets_RA2RA, &d_cum_num_targets_RA2RA,
					 &d_targets_id_RA2I,  &d_weights_RA2I,  &d_num_targets_RA2I,  &d_cum_num_targets_RA2I,
					 &d_targets_id_I2RA,  &d_weights_I2RA,  &d_num_targets_I2RA,  &d_cum_num_targets_I2RA);

	

	initialize_neurons(N_RA, N_I, 
					   &d_vars_HVCRA,        &d_vars_HVCI,
					   &d_buffer_HVCRA,      &d_buffer_HVCI,
					   &d_record_HVCRA,      &d_record_HVCI,
					   &d_bool_spiked_HVCRA, &d_bool_spiked_HVCI,
					   &d_flags_HVCRA,       &d_flags_HVCI);
	
	
	initialize_spike_info(N_RA, N_I,
						  &h_num_spikes_HVCRA,  &d_num_spikes_HVCRA,
						  &h_spike_times_HVCRA, &d_spike_times_HVCRA,
						  &h_num_spikes_HVCI,   &d_num_spikes_HVCI,
						  &h_spike_times_HVCI,  &d_spike_times_HVCI);

	
	double timestep = 0.01;
	double trial = 100;
	int num_iter = static_cast<int>(trial / timestep);
	int num_iter_without_sync = 10;

	int num_blocks_HVCRA = N_RA / NUM_THREADS_IN_BLOCK + 1;
	int num_blocks_HVCI = N_I / NUM_THREADS_IN_BLOCK + 1;

	int neuron_to_record = 1;
	int training = 200;

	double *big_buffer_HVCRA, *big_buffer_HVCI;
	
	big_buffer_HVCRA = new double[3*(num_iter / BUFFER_SIZE)* BUFFER_SIZE];
	big_buffer_HVCI = new double[3*(num_iter / BUFFER_SIZE)* BUFFER_SIZE];
	
	set_record_HVCRA<<<num_blocks_HVCRA, NUM_THREADS_IN_BLOCK >>>(d_record_HVCRA, neuron_to_record);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	set_record_HVCI<<<num_blocks_HVCI, NUM_THREADS_IN_BLOCK >>>(d_record_HVCI, neuron_to_record);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	

	// set up noise
	curandState *d_states_HVCI, *d_states_HVCRA;
	
	gpuErrchk(cudaMalloc( &d_states_HVCRA, N_RA*sizeof( curandState )));
	gpuErrchk(cudaMalloc( &d_states_HVCI,  N_I*sizeof( curandState )));
	
	initialize_noise_RA<<< num_blocks_HVCRA, NUM_THREADS_IN_BLOCK >>>(d_vars_HVCRA, N_RA, seed, d_states_HVCRA);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	initialize_noise_I<<< num_blocks_HVCI, NUM_THREADS_IN_BLOCK >>>(d_vars_HVCI, N_I, seed + N_RA, d_states_HVCI);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	int step = 0;

	std::cout << "Before starting iterations\n" << std::endl;

	int i = 0;

	double ampl_s = 0.0;
	double ampl_d = 2.0;
	double ampl = 0.0;

	gettimeofday(&start_calc, NULL);
	
	while (i < num_iter)
	{
		//std::cout << "time = " << static_cast<double>(i) * timestep << std::endl;

		for (int j = 0; j < num_iter_without_sync; j++)
		{

	 		calculate_next_step_RA<<<num_blocks_HVCRA, NUM_THREADS_IN_BLOCK >>>(d_vars_HVCRA, d_flags_HVCRA, N_RA, timestep, d_record_HVCRA, 
																				d_buffer_HVCRA, step, ampl_s, ampl_d, training, 
																				d_bool_spiked_HVCRA, d_num_spikes_HVCRA, d_spike_times_HVCRA,
																				d_states_HVCRA);

	 		calculate_next_step_I<<<num_blocks_HVCI, NUM_THREADS_IN_BLOCK >>>(d_vars_HVCI, d_flags_HVCI, N_I, timestep, d_record_HVCI, 
																				d_buffer_HVCI, step, ampl, 
																				d_bool_spiked_HVCI, d_num_spikes_HVCI, d_spike_times_HVCI,
																				d_states_HVCI);
			cudaDeviceSynchronize();

			if ( (i+1) % BUFFER_SIZE == 0)
			{
				step = 0;
			
				//print_buffer<<<num_blocks, NUM_THREADS_IN_BLOCK >>>(d_buffer, BUFFER_SIZE);
			
				int ind = ( (i+1)/BUFFER_SIZE - 1 ) * BUFFER_SIZE * 3;
			
				cudaMemcpy(&big_buffer_HVCRA[ind], d_buffer_HVCRA, BUFFER_SIZE*3*sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(&big_buffer_HVCI[ind], d_buffer_HVCI, BUFFER_SIZE*3*sizeof(double), cudaMemcpyDeviceToHost);
			
				
				//std::cout << "ind = " << ind 
				//	  	<< " time = " << big_buffer_HVCRA[ind]
				//	  	<< " Vs = " << big_buffer_HVCRA[ind + 1]
				//	  	<< " Vd = " << big_buffer_HVCRA[ind + 2] << std::endl;
				

				std::cout << "ind = " << ind 
					  	<< " time = " << big_buffer_HVCI[ind]
					  	<< " V = " << big_buffer_HVCI[ind + 1]
					  	<< " Ge = " << big_buffer_HVCI[ind + 2] << std::endl;
			}
			else
				step += 1;

			i++;
		}

		//std::cout << "Before update of target conductances\n" << std::endl;

	 	update_target_conductances_HVCRA<<<num_blocks_HVCRA, NUM_THREADS_IN_BLOCK >>>(d_vars_HVCRA, d_vars_HVCI, N_RA, N_I, d_bool_spiked_HVCRA, 
															d_targets_id_RA2RA, d_weights_RA2RA, d_cum_num_targets_RA2RA, d_num_targets_RA2RA,
															d_targets_id_RA2I, d_weights_RA2I, d_cum_num_targets_RA2I, d_num_targets_RA2I);

	 	//update_target_conductances_RA2I<<<num_blocks_HVCRA, NUM_THREADS_IN_BLOCK >>>(d_vars_HVCI, N_RA, N_I, d_bool_spiked_HVCRA, 
		//													d_targets_id_RA2I, d_weights_RA2I, d_cum_num_targets_RA2I, d_num_targets_RA2I);
	 	
		update_target_conductances_I2RA<<<num_blocks_HVCI, NUM_THREADS_IN_BLOCK >>>(d_vars_HVCRA, N_I, N_RA, d_bool_spiked_HVCI, 
															d_targets_id_I2RA, d_weights_I2RA, d_cum_num_targets_I2RA, d_num_targets_I2RA);
	}

	//cudaDeviceSynchronize();

	//std::cout << "Before memcpy\n" << std::endl;

	cudaMemcpy(h_spike_times_HVCRA, d_spike_times_HVCRA, N_RA*MAX_NUM_OF_SPIKES*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_num_spikes_HVCRA, d_num_spikes_HVCRA, N_RA*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaMemcpy(h_spike_times_HVCI, d_spike_times_HVCI, N_I*MAX_NUM_OF_SPIKES*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_num_spikes_HVCI, d_num_spikes_HVCI, N_I*sizeof(int), cudaMemcpyDeviceToHost);
	//std::cout << "After memcpy\n" << std::endl;

	//std::cout << "Number of spikes:\n";

	//for (int i = 0; i < N_RA; i++)
	//	std::cout << h_num_spikes[i] << " ";
	//std::cout << std::endl;
	


	write_data(big_buffer_HVCRA, (num_iter/BUFFER_SIZE) * BUFFER_SIZE, filename_buffer_HVCRA.c_str());
	write_spikes(h_spike_times_HVCRA, h_num_spikes_HVCRA, N_RA, filename_spikes_HVCRA.c_str());
	
	write_data(big_buffer_HVCI, (num_iter/BUFFER_SIZE) * BUFFER_SIZE, filename_buffer_HVCI.c_str());
	write_spikes(h_spike_times_HVCI, h_num_spikes_HVCI, N_I, filename_spikes_HVCI.c_str());
	
	gettimeofday(&end, NULL);

	std::cout << "Time for allocation and calculation: " << myDiffTime(start, end) << "\n";
	std::cout << "Time for calculation: " << myDiffTime(start_calc, end) << "\n";

	// free memory on host
	delete[] big_buffer_HVCRA; delete[] big_buffer_HVCI;
	delete[] h_num_spikes_HVCRA; delete h_num_spikes_HVCI;
	delete[] h_spike_times_HVCRA; delete[] h_spike_times_HVCI;

	// free memory on device
	cudaFree(d_buffer_HVCRA); cudaFree(d_vars_HVCRA); cudaFree(d_record_HVCRA); cudaFree(d_bool_spiked_HVCRA);
	cudaFree(d_buffer_HVCI); cudaFree(d_vars_HVCI); cudaFree(d_record_HVCI); cudaFree(d_bool_spiked_HVCI);
	
	cudaFree(d_targets_id_RA2RA); cudaFree(d_weights_RA2RA); cudaFree(d_cum_num_targets_RA2RA); cudaFree(d_num_targets_RA2RA);
	cudaFree(d_targets_id_RA2I);  cudaFree(d_weights_RA2I);  cudaFree(d_cum_num_targets_RA2I);  cudaFree(d_num_targets_RA2I);
	cudaFree(d_targets_id_I2RA);  cudaFree(d_weights_I2RA);  cudaFree(d_cum_num_targets_I2RA);  cudaFree(d_num_targets_I2RA);
	

	cudaFree(d_num_spikes_HVCRA); cudaFree(d_spike_times_HVCRA); cudaFree(d_flags_HVCRA);
	cudaFree(d_num_spikes_HVCI);  cudaFree(d_spike_times_HVCI);  cudaFree(d_flags_HVCI);
	
	cudaFree(d_states_HVCRA); cudaFree(d_states_HVCI);
	
		
}
